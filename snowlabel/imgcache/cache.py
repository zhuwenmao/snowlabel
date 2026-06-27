"""ImageCache:三库编排层。

对外只暴露 image_hash / lookup / store / put_image / image_done / stats。
内部把图片(MinIO)、mask 元数据(MySQL)、向量(Milvus)三处协调好。

命中读取设计:为规避 milvus-lite 写后读的秒级延迟(与记忆库 RAM 镜像同理),
lookup 完全走 MySQL —— v_whole / v_parts / geom 都以 base64(float16) 存在 MySQL,
解码时维度从字节长度自动推断,无需外部传 dim。Milvus 那份向量是持久档案 + 未来
ANN 检索源。任何一步异常都降级(返回 None / 吞掉),绝不阻断主标注流程。
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

log = logging.getLogger("snowlabel.imgcache")


# ---------- 编解码 ----------
def _enc(arr) -> str:
    if arr is None:
        return ""
    return base64.b64encode(
        np.asarray(arr, dtype=np.float16).tobytes()).decode("ascii")


def _dec(s: str) -> np.ndarray | None:
    if not s:
        return None
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


def _mask_to_png(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, "L").save(
        buf, format="PNG")
    return buf.getvalue()


def _png_to_mask(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("L")) > 127


def image_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class CachedMask:
    """一个命中缓存的 mask。同时充当 MaskFeatures(有 v_whole/v_parts/geom),
    可直接喂给 predictor._predict_from_features 做最新记忆库的实时重判。"""
    mask: np.ndarray
    bbox: tuple
    v_whole: np.ndarray
    v_parts: np.ndarray | None
    geom: np.ndarray | None
    pred_label: str
    tier: str
    confidence: float


class ImageCache:
    def __init__(self, cfg):
        from .meta_store import MetaStore
        from .object_store import ObjectStore
        self.cfg = cfg
        self.meta = MetaStore(cfg.mysql_url)
        self.obj = ObjectStore(cfg.minio_endpoint, cfg.minio_access_key,
                               cfg.minio_secret_key, cfg.minio_bucket,
                               secure=cfg.minio_secure)
        self._vec = None  # 延迟建(首次 store 时按 v_whole 维度建 collection)

    def _get_vec(self, dim: int):
        if self._vec is None:
            from .vector_store import MaskVectorStore
            self._vec = MaskVectorStore(self.cfg.imgcache_collection, dim,
                                        uri=self.cfg.milvus_uri,
                                        token=self.cfg.milvus_token)
        return self._vec

    # ---------- 写路径 ----------
    def put_image(self, image_hash: str, data: bytes, *, ext: str = ".jpg",
                  orig_name: str = "", width: int = 0, height: int = 0) -> str:
        """上传时把图片落进 MinIO 并登记(sam_done=False,排队待处理)。返回对象 key。"""
        key = self.obj.put_image(image_hash, data, ext=ext)
        # 仅在尚无该图记录时登记,不覆盖已处理完的 sam_done。
        if self.meta.get_image(image_hash) is None:
            self.meta.upsert_image(image_hash, width=width, height=height,
                                   minio_key=key, orig_name=orig_name,
                                   sam_done=False)
        return key

    def store(self, image_hash: str, items: list[dict], *,
              img_bytes: bytes | None = None, ext: str = ".jpg",
              orig_name: str = "", width: int = 0, height: int = 0) -> None:
        """把一张图的全部 mask(SAM+DINO 算好的)写入三库,标记 sam_done。
        items[i] = {mask, bbox, v_whole, v_parts, geom, pred_label, tier, confidence}。
        """
        try:
            key = ""
            if img_bytes is not None:
                key = self.obj.put_image(image_hash, img_bytes, ext=ext)
            rows = []
            for i, it in enumerate(items):
                x0, y0, x1, y1 = (it.get("bbox") or (0, 0, 0, 0))
                rows.append(dict(
                    idx=i, bbox_x0=int(x0), bbox_y0=int(y0),
                    bbox_x1=int(x1), bbox_y1=int(y1),
                    area=int(np.asarray(it["mask"]).sum()),
                    geom_enc=_enc(it.get("geom")),
                    whole_enc=_enc(it["v_whole"]),
                    parts_enc=_enc(it.get("v_parts")),
                    mask_png=_mask_to_png(it["mask"]),
                    pred_label=(it.get("pred_label") or "")[:128],
                    tier=(it.get("tier") or "silent")[:16],
                    confidence=float(it.get("confidence") or 0.0),
                ))
            mask_ids = self.meta.replace_masks(image_hash, rows)
            # 向量入 Milvus(档案)。失败不影响 MySQL 已写好的缓存。
            try:
                wholes = [np.asarray(it["v_whole"], np.float32) for it in items]
                if wholes:
                    self._get_vec(wholes[0].shape[0]).add(mask_ids, image_hash, wholes)
            except Exception as e:
                log.warning("Milvus 向量写入失败(不影响缓存命中): %s", e)
            self.meta.upsert_image(image_hash, width=width, height=height,
                                   minio_key=key, orig_name=orig_name,
                                   sam_done=True, n_masks=len(items))
            log.info("cached image %s: %d masks", image_hash[:12], len(items))
        except Exception as e:
            log.error("ImageCache.store 失败 %s: %s", image_hash[:12], e)

    # ---------- 读路径 ----------
    def image_done(self, image_hash: str) -> bool:
        try:
            return self.meta.image_done(image_hash)
        except Exception:
            return False

    def lookup(self, image_hash: str) -> list[CachedMask] | None:
        """命中返回该图所有 mask(含特征);未处理/未命中/异常都返回 None。"""
        try:
            if not self.meta.image_done(image_hash):
                return None
            rows = self.meta.get_masks(image_hash)
            if not rows:
                return None
            out: list[CachedMask] = []
            for r in rows:
                whole = _dec(r.whole_enc)
                if whole is None:
                    return None  # 没有向量无法重判,视作未命中走重算
                dim = whole.shape[0]
                parts_flat = _dec(r.parts_enc)
                parts = (parts_flat.reshape(-1, dim)
                         if parts_flat is not None and parts_flat.size else None)
                geom = _dec(r.geom_enc)
                out.append(CachedMask(
                    mask=_png_to_mask(r.mask_png),
                    bbox=(r.bbox_x0, r.bbox_y0, r.bbox_x1, r.bbox_y1),
                    v_whole=whole, v_parts=parts, geom=geom,
                    pred_label=r.pred_label, tier=r.tier,
                    confidence=r.confidence))
            return out
        except Exception as e:
            log.warning("ImageCache.lookup 失败 %s: %s", image_hash[:12], e)
            return None

    def stats(self) -> dict:
        try:
            d = self.meta.stats()
        except Exception:
            d = {}
        try:
            d["vectors"] = self._vec.count() if self._vec is not None else 0
        except Exception:
            d["vectors"] = 0
        return d


def make_image_cache(cfg):
    """按 cfg.imgcache_enabled 决定是否启用。任何连接失败都降级为 None(主流程照常)。"""
    if not getattr(cfg, "imgcache_enabled", False):
        return None
    try:
        c = ImageCache(cfg)
        log.info("ImageCache 启用 (MinIO+MySQL+Milvus)")
        return c
    except Exception as e:
        log.error("ImageCache 初始化失败,降级为不缓存: %s", e)
        return None
