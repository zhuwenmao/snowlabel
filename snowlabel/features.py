"""Mask feature groups — crop-encode + parts + geometry.

Replaces the old "encode whole image at 224, mask-pool one vector" approach,
which wasted resolution on small objects and collapsed all spatial structure
into a single averaged vector (fatal for fine-grained classes).

For each mask we produce a small GROUP of descriptors:
  - v_whole  [D]      : the mask's region CROPPED, letterboxed to a square (no
                        aspect distortion), resized to the backbone resolution
                        (SNOWLABEL_RES, e.g. 512), encoded, then soft-coverage
                        pooled. The whole crop is spent on this one object ->
                        real effective resolution.
  - v_parts  [P, D]   : the crop's patch grid split into a parts_grid x
                        parts_grid layout (SNOWLABEL_PARTS_GRID, e.g. 4 -> 16
                        cells), each cell soft-coverage pooled -> coarse spatial
                        structure survives (breaks the "mean collision").
  - geom     [G]      : shape descriptors (area fraction, aspect, extent,
                        solidity) the appearance vector throws away.

`features_batch` encodes every crop of one image in a single batched DINO
forward, so "segment everything" stays fast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

try:
    import cv2  # ships with ultralytics; used only for solidity
except Exception:  # pragma: no cover
    cv2 = None

from .config import get_config
from .models import DinoFeatureExtractor

P_GRID_DEFAULT = 4         # 4x4 = 16 part vectors (override via SNOWLABEL_PARTS_GRID)
GEOM_DIM = 4
CROP_PAD = 0.15            # bbox padding used by crop-encode (part of the space id)


def feature_space_id(dim: int) -> str:
    """Identity of the vector space every stored embedding lives in.

    Changing any of these knobs (backbone, resolution, parts grid, crop pad)
    silently invalidates every stored vector — cosine against old vectors
    still "works" but means nothing. Memory backends persist this string and
    refuse to load a non-empty store written under a different one.
    """
    from pathlib import Path
    cfg = get_config()
    name = Path(str(cfg.dino_variant)).name
    return (f"{name}|res{cfg.input_resolution}|parts{cfg.parts_grid}"
            f"|dim{dim}|pad{CROP_PAD}|v1")


@dataclass
class MaskFeatures:
    v_whole: np.ndarray    # [D]   f32, L2-normalized
    v_parts: np.ndarray    # [P,D] f32, each row L2-normalized (zero row = empty cell)
    geom: np.ndarray       # [G]   f32


class MaskFeatureExtractor:
    """Turns (image, mask) into a MaskFeatures group via crop-encoding."""

    def __init__(self, dino: DinoFeatureExtractor, pad: float = CROP_PAD,
                 p_grid: int | None = None) -> None:
        self.dino = dino
        self.pad = pad
        self.p_grid = int(p_grid if p_grid is not None else
                          getattr(get_config(), "parts_grid", P_GRID_DEFAULT))
        self.n_parts = self.p_grid * self.p_grid

    # ---------- geometry ----------
    @staticmethod
    def _bbox(mask: np.ndarray):
        ys, xs = np.where(mask)
        if xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _geom(self, mask: np.ndarray) -> np.ndarray:
        H, W = mask.shape
        a = float(mask.sum())
        bb = self._bbox(mask)
        if bb is None or a < 1:
            return np.zeros(GEOM_DIM, np.float32)
        x0, y0, x1, y1 = bb
        bw, bh = x1 - x0, y1 - y0
        area_frac = a / (H * W)
        aspect = bw / max(1, bh)
        extent = a / max(1, bw * bh)
        solidity = extent
        if cv2 is not None:
            try:
                cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    hull = cv2.convexHull(np.vstack(cnts))
                    ha = float(cv2.contourArea(hull))
                    if ha > 1:
                        solidity = a / ha
            except Exception:
                pass
        return np.array([np.sqrt(area_frac), float(np.log(aspect + 1e-3)),
                         extent, solidity], np.float32)

    # ---------- crop + letterbox ----------
    def _crop_letterbox(self, img: Image.Image, mask: np.ndarray):
        """Crop the padded bbox and paste onto a black square (preserve aspect).
        Returns (square_pil, square_mask) or (None, None) for an empty mask."""
        W, H = img.size
        bb = self._bbox(mask)
        if bb is None:
            return None, None
        x0, y0, x1, y1 = bb
        bw, bh = x1 - x0, y1 - y0
        pad = int(self.pad * max(bw, bh))
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
        crop = img.crop((x0, y0, x1, y1))
        cm = mask[y0:y1, x0:x1]
        cw, ch = crop.size
        side = max(cw, ch, 1)
        sq = Image.new("RGB", (side, side), (0, 0, 0))
        ox, oy = (side - cw) // 2, (side - ch) // 2
        sq.paste(crop, (ox, oy))
        msq = np.zeros((side, side), bool)
        msq[oy:oy + ch, ox:ox + cw] = cm
        return sq, msq

    # ---------- pooling ----------
    @staticmethod
    def _pool(tokens: torch.Tensor, w: torch.Tensor) -> np.ndarray:
        """Soft-coverage weighted mean of tokens[...,D] by weights w(same lead),
        L2-normalized. tokens flattened over leading dims."""
        D = tokens.shape[-1]
        t = tokens.reshape(-1, D)
        wf = w.reshape(-1)
        ws = float(wf.sum())
        feat = t.mean(0) if ws < 1e-6 else (t * wf.unsqueeze(1)).sum(0) / ws
        feat = feat / (feat.norm() + 1e-8)
        return feat.detach().cpu().numpy().astype(np.float32)

    def _mask_grid(self, msq: np.ndarray, g: int, device) -> torch.Tensor:
        m = Image.fromarray((msq.astype(np.uint8) * 255)).resize((g, g), Image.BILINEAR)
        return torch.from_numpy(np.asarray(m, np.float32) / 255.0).to(device)

    # ---------- public ----------
    def features_batch(self, img: Image.Image, masks: list[np.ndarray]) -> list[MaskFeatures]:
        s = self.dino._ensure_loaded()
        g, D = s.grid, s.embed_dim
        crops: list = []
        msqs: list = []
        geoms: list = []
        for mk in masks:
            sq, msq = self._crop_letterbox(img, mk)
            crops.append(sq); msqs.append(msq); geoms.append(self._geom(mk))

        valid = [i for i, c in enumerate(crops) if c is not None]
        tokens_by_i: dict[int, torch.Tensor] = {}
        B = 8
        for st in range(0, len(valid), B):
            idxs = valid[st:st + B]
            tk = self.dino.encode_batch([crops[i] for i in idxs])  # [b, g*g, D]
            for j, i in enumerate(idxs):
                tokens_by_i[i] = tk[j].reshape(g, g, D)

        bounds = [round(k * g / self.p_grid) for k in range(self.p_grid + 1)]
        out: list[MaskFeatures] = []
        for i in range(len(masks)):
            if i not in tokens_by_i:
                out.append(MaskFeatures(np.zeros(D, np.float32),
                                        np.zeros((self.n_parts, D), np.float32), geoms[i]))
                continue
            tg = tokens_by_i[i]                                   # [g,g,D]
            w = self._mask_grid(msqs[i], g, tg.device)           # [g,g]
            v_whole = self._pool(tg, w)
            parts = []
            for r in range(self.p_grid):
                for c in range(self.p_grid):
                    ts = tg[bounds[r]:bounds[r + 1], bounds[c]:bounds[c + 1], :]
                    ws = w[bounds[r]:bounds[r + 1], bounds[c]:bounds[c + 1]]
                    parts.append(self._pool(ts, ws) if float(ws.sum()) > 1e-3
                                 else np.zeros(D, np.float32))
            out.append(MaskFeatures(v_whole, np.stack(parts).astype(np.float32), geoms[i]))
        return out

    def features(self, img: Image.Image, mask: np.ndarray) -> MaskFeatures:
        return self.features_batch(img, [mask])[0]


# ---------- comparison helpers (used by the predictor's rerank) ----------
def parts_maxsim(q: np.ndarray, c: np.ndarray) -> float:
    """Asymmetric MaxSim over part vectors: mean over query parts of the best
    matching candidate part (ignores empty/zero rows). A cheap, pose-tolerant
    surrogate for optimal transport."""
    if q is None or c is None or q.size == 0 or c.size == 0:
        return 0.0
    qn = np.linalg.norm(q, axis=1)
    cn = np.linalg.norm(c, axis=1)
    qv = q[qn > 1e-6]
    cv_ = c[cn > 1e-6]
    if qv.shape[0] == 0 or cv_.shape[0] == 0:
        return 0.0
    sims = qv @ cv_.T                       # parts are pre-normalized
    return float(sims.max(axis=1).mean())


def geom_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.5
    d = float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))
    return float(np.exp(-d))
