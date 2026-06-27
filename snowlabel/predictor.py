"""SnowPredictor — orchestrates SAM + DINO + Memory for one annotation request.

The predictor is stateful (it holds open model handles and the memory store),
which is why we initialize one per process and share it across requests.

Confidence tiers (configurable in config.py). "conf" is the winner class's
ABSOLUTE rerank blend score (0.55*cos_whole + 0.35*parts_maxsim +
0.10*geom_sim), NOT a share over classes — shares dilute as the label set
grows. Auto additionally requires a class margin and support:
    conf >= conf_auto_apply (+margin+support) : "auto"    → pre-fill label
    conf >= conf_suggest    (+support)        : "suggest" → require a click
    otherwise                                 : "silent"  → mask only
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .config import SnowConfig, get_config
from .features import MaskFeatureExtractor, geom_sim, parts_maxsim
from .imgcache import make_image_cache
from .memory import SnowballMemory, get_memory
from .models import DinoFeatureExtractor, SamMask, SamSegmenter

log = logging.getLogger("snowlabel.predictor")


Tier = Literal["auto", "suggest", "silent"]

# Stage-2 rerank blend weights. Module constants (not config) because changing
# them invalidates tuned thresholds; the eval harness snapshots them per run.
RERANK_W_WHOLE = 0.55
RERANK_W_PARTS = 0.35
RERANK_W_GEOM = 0.10


@dataclass
class Prediction:
    """One predicted annotation for a single mask."""
    mask: np.ndarray              # bool H×W
    bbox: tuple[int, int, int, int] | None
    label: str | None             # predicted class, or None if memory empty / low conf
    confidence: float             # 0.0 .. 1.0
    tier: Tier                    # auto / suggest / silent
    top_k: list[tuple[str, float]]  # full breakdown for debugging / UI
    cw: float = 0.0               # best stage-1 v_whole cosine (raw NN sim, pre-rerank)


class SnowPredictor:
    """The brains of SnowLabel.

    One instance per server process. Loads SAM + DINO + Memory lazily.
    """

    def __init__(self, config: SnowConfig | None = None) -> None:
        self.cfg = config or get_config()
        self.sam = SamSegmenter()
        self.dino = DinoFeatureExtractor()
        self.feat = MaskFeatureExtractor(self.dino)
        self._memory: SnowballMemory | None = None
        # 三库缓存(MinIO+MySQL+Milvus):开启后 everything 走持久化缓存、跳过重复 SAM。
        # 关闭(默认)或连接失败时为 None,所有缓存分支自动退化为直接计算。
        self.cache = make_image_cache(self.cfg)
        # GPU 串行锁:后台预处理 worker 与前台请求共用同一份 SAM/DINO,加锁防并发崩溃。
        # 只锁真正跑 GPU 的计算段;命中缓存的快路径不持锁,保证秒出。
        self.gpu_lock = threading.Lock()

    @property
    def memory(self):
        """Returns whichever memory backend is configured (numpy or milvus)."""
        if self._memory is None:
            self._memory = get_memory(self.cfg.memory_dir, dim=self.dino.embed_dim)
        return self._memory

    # ---------- prediction paths ----------
    def predict_from_point(
        self,
        img_pil: Image.Image,
        x: int,
        y: int,
        *,
        user_id: str | None = None,
    ) -> Prediction | None:
        """Interactive: user clicks (x, y) on the image → mask + predicted label."""
        with self.gpu_lock:
            sm = self.sam.predict_from_point(img_pil, x, y)
            if sm is None:
                return None
            return self._enrich_with_label(img_pil, sm, user_id=user_id)

    def predict_from_points(
        self,
        img_pil: Image.Image,
        points: list[list[int]],
        labels: list[int],
        *,
        user_id: str | None = None,
    ) -> Prediction | None:
        """Interactive multi-point (positive + negative clicks) → mask + label."""
        with self.gpu_lock:
            sm = self.sam.predict_from_points(img_pil, points, labels)
            if sm is None:
                return None
            return self._enrich_with_label(img_pil, sm, user_id=user_id)

    def predict_from_box(
        self,
        img_pil: Image.Image,
        bbox_xyxy: tuple[int, int, int, int],
        *,
        user_id: str | None = None,
    ) -> Prediction | None:
        with self.gpu_lock:
            sm = self.sam.predict_from_box(img_pil, bbox_xyxy)
            if sm is None:
                return None
            return self._enrich_with_label(img_pil, sm, user_id=user_id)

    def predict_all(
        self,
        img_pil: Image.Image,
        max_masks: int = 16,
        *,
        user_id: str | None = None,
    ) -> list[Prediction]:
        """Auto mode: SAM proposes all masks, we tag each with a predicted label."""
        with self.gpu_lock:
            sms = self.sam.predict_all(img_pil, max_masks=max_masks)
            if not sms:
                return []
            # One BATCHED crop-encode over all masks — keeps "everything" fast.
            mfs = self.feat.features_batch(img_pil, [sm.mask for sm in sms])
            return [self._prediction_from_features(sm, mf, user_id=user_id)
                    for sm, mf in zip(sms, mfs)]

    def segment_all_cached(
        self,
        img_bytes: bytes,
        img_pil: Image.Image,
        max_masks: int = 16,
        *,
        user_id: str | None = None,
        ext: str = ".jpg",
        orig_name: str = "",
    ) -> tuple[list[Prediction], bool]:
        """带三库缓存的 everything。返回 (preds, from_cache)。

        命中:从库读 SAM mask + DINO 特征,跳过最贵的 SAM/DINO;label/tier 仍用最新
        记忆库对缓存特征实时重判(便宜),保证不因记忆库增长而过时。未命中:正常算一遍
        再写库。缓存关闭/不可用(self.cache is None)时,行为等价于 predict_all。
        """
        if self.cache is None:
            return (self.predict_all(img_pil, max_masks=max_masks,
                                     user_id=user_id), False)
        from .imgcache.cache import image_sha256
        h = image_sha256(img_bytes)
        cached = self.cache.lookup(h)
        if cached is not None:
            preds = []
            for cm in cached:   # CachedMask 自带 v_whole/v_parts/geom,可直接当特征用
                label, conf, tier, top_k, cw = self._predict_from_features(
                    cm, user_id=user_id)
                preds.append(Prediction(
                    mask=cm.mask, bbox=cm.bbox, label=label, confidence=conf,
                    tier=tier, top_k=top_k, cw=cw))
            log.info("everything cache HIT %s (%d masks)", h[:12], len(preds))
            return preds, True
        # miss → 算 SAM+DINO(持 GPU 锁),再在锁外持久化
        with self.gpu_lock:
            sms = self.sam.predict_all(img_pil, max_masks=max_masks)
            if not sms:
                return [], False
            mfs = self.feat.features_batch(img_pil, [sm.mask for sm in sms])
            preds = [self._prediction_from_features(sm, mf, user_id=user_id)
                     for sm, mf in zip(sms, mfs)]
        items = [dict(mask=sm.mask, bbox=sm.bbox, v_whole=mf.v_whole,
                      v_parts=mf.v_parts, geom=mf.geom,
                      pred_label=p.label or "", tier=p.tier,
                      confidence=p.confidence)
                 for sm, mf, p in zip(sms, mfs, preds)]
        w, hgt = img_pil.size
        self.cache.store(h, items, img_bytes=img_bytes, ext=ext,
                         orig_name=orig_name, width=w, height=hgt)
        log.info("everything cache MISS %s → stored %d masks", h[:12], len(preds))
        return preds, False

    # ---------- memory write ----------
    def remember(
        self,
        img_pil: Image.Image,
        mask: np.ndarray,
        label: str,
        *,
        image_uri: str = "",
        verified: bool = True,
        user_id: str = "",
    ) -> int:
        """User confirmed/added a label — push it into the snowball memory."""
        mf = self.feat.features(img_pil, mask)
        entry_id = self.memory.add(
            mf.v_whole,
            label=label,
            image_uri=image_uri,
            verified=verified,
            user_id=user_id,
            parts=mf.v_parts,
            geom=mf.geom,
        )
        log.info("Remembered '%s' (entry=%d, memory_size=%d)",
                 label, entry_id, len(self.memory))
        return entry_id

    def remember_feature(self, feat: np.ndarray, label: str, *, image_uri: str = "",
                         user_id: str = "") -> int:
        return self.memory.add(feat, label=label, image_uri=image_uri,
                               verified=True, user_id=user_id)

    def remember_batch(
        self,
        img_pil: Image.Image,
        items: list[tuple[np.ndarray, str]],
        *,
        image_uri: str = "",
        user_id: str = "",
    ) -> list[int]:
        """Commit several (mask, label) pairs from ONE image into memory.

        Encodes the image with DINO a single time and pools every mask from the
        shared patch tokens — far cheaper than calling remember() per mask
        (which would re-encode the whole image each time). Used by the
        "segment everything → bulk accept" flow.
        """
        if not items:
            return []
        masks = [m for m, _ in items]
        labels = [l for _, l in items]
        mfs = self.feat.features_batch(img_pil, masks)   # one batched crop-encode
        v = np.stack([mf.v_whole for mf in mfs])
        parts = np.stack([mf.v_parts for mf in mfs])
        geom = np.stack([mf.geom for mf in mfs])
        ids = self.memory.add_batch(
            v, labels, image_uri=image_uri, user_id=user_id, parts=parts, geom=geom)
        log.info("Remembered batch of %d (memory_size=%d)", len(ids), len(self.memory))
        return ids

    # ---------- internals ----------
    def _enrich_with_label(self, img_pil: Image.Image, sm: SamMask,
                           user_id: str | None = None) -> Prediction:
        mf = self.feat.features(img_pil, sm.mask)
        return self._prediction_from_features(sm, mf, user_id=user_id)

    def _prediction_from_features(self, sm: SamMask, mf,
                                  user_id: str | None = None) -> Prediction:
        label, conf, tier, top_k, cw = self._predict_from_features(mf, user_id=user_id)
        return Prediction(
            mask=sm.mask, bbox=sm.bbox, label=label,
            confidence=conf, tier=tier, top_k=top_k, cw=cw,
        )

    def _predict_from_features(self, mf, user_id: str | None = None):
        """Two-stage retrieval: Milvus ANN on v_whole (stage 1), then a Python
        rerank by part-level MaxSim + geometry (stage 2), with an open-set guard
        and margin-gated tiering. Returns (label|None, conf, tier, top_k, cw)
        where cw is the best raw v_whole cosine (open-set / diagnostics)."""
        entries, sims = self.memory.query(mf.v_whole, k=self.cfg.rerank_n,
                                          user_id=user_id)
        if len(entries) == 0:
            return None, 0.0, "silent", [], 0.0
        best_cw = float(np.max(sims))
        # Open-set guard: nothing in memory actually looks like this object.
        if best_cw < self.cfg.open_set_min_sim:
            return None, round(best_cw, 4), "silent", [], round(best_cw, 4)
        scored: list[tuple[str, float]] = []
        for e, s in zip(entries, sims):
            cw = float(s)
            pm = parts_maxsim(mf.v_parts, e.parts) if e.parts is not None else cw
            gs = geom_sim(mf.geom, e.geom) if e.geom is not None else 0.5
            scored.append((e.label,
                           RERANK_W_WHOLE * cw + RERANK_W_PARTS * pm + RERANK_W_GEOM * gs))
        class_score: dict[str, float] = {}
        for lbl, sc in scored:
            if sc > class_score.get(lbl, -1.0):
                class_score[lbl] = sc
        ranked = sorted(class_score.items(), key=lambda kv: -kv[1])
        winner, wscore = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = wscore - runner
        # Confidence = the winner's ABSOLUTE blend score, not its share of all
        # class scores. Wrong classes keep a ~0.5 score floor (cos/parts/geom
        # never reach 0), so a share collapses as classes accumulate — with 5
        # classes even a perfect match capped near 0.30 and every prediction
        # went silent. The absolute score is class-count invariant.
        conf = float(min(max(wscore, 0.0), 1.0))
        support = self._class_support(winner, user_id=user_id)
        tier = self._tier_group(conf, margin, support)
        top_k = [(l, round(float(v), 4)) for l, v in ranked[:5]]
        return ((winner if tier != "silent" else None), round(float(conf), 4),
                tier, top_k, round(best_cw, 4))

    def _class_support(self, label: str | None, user_id: str | None = None) -> int:
        """How many stored examples back this class (cold-start gate). Cached and
        only rescanned when the memory size or active user changes, so a 'segment
        everything' call doesn't full-scan the store per mask."""
        if label is None:
            return 0
        key = (len(self.memory), user_id or "")
        if getattr(self, "_sup_key", None) != key:
            from collections import Counter
            self._sup_counts = Counter(
                e.label for e in self.memory.iter_entries()
                if (not user_id) or e.user_id == user_id)
            self._sup_key = key
        return self._sup_counts.get(label, 0)

    def _tier_group(self, conf: float, margin: float, support: int) -> Tier:
        """Confidence tier with open-set + class-margin protections layered on the
        cold-start support gate."""
        if (conf >= self.cfg.conf_auto_apply and margin >= self.cfg.margin_auto
                and support >= self.cfg.min_support_auto):
            return "auto"
        if conf >= self.cfg.conf_suggest and support >= self.cfg.min_support_suggest:
            return "suggest"
        return "silent"

    # ---------- introspection ----------
    def stats(self) -> dict:
        # Touch the property (not the private _memory) so a freshly started
        # server reports the TRUE persisted count rather than a misleading 0.
        # Memory is lazy-loaded and stats() is exactly when a caller wants it
        # loaded; health() never touches memory, so LS heartbeats stay light.
        s = self.memory.stats()
        return {
            "snowlabel_version": "0.1.0",
            "dino_variant": self.cfg.dino_variant,
            "sam_variant": self.cfg.sam_variant,
            "predict_mode": self.cfg.predict_mode,
            "memory_backend": self.cfg.memory_backend,
            "memory": s,
            "thresholds": {
                "auto_apply": self.cfg.conf_auto_apply,
                "suggest": self.cfg.conf_suggest,
            },
            # 三库缓存积累量(体现"越用越多的数据集"):图片/mask/向量数。
            "imgcache": (self.cache.stats() if self.cache is not None
                         else {"enabled": False}),
        }
