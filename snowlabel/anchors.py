"""Anchor preview helpers.

User-taught memory entries are the real "anchors" that influence future
predictions. Their feature vectors live in the memory backend; this module
stores a small visual preview beside them in MinIO so the UI can inspect and
remove bad anchors.
"""
from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

log = logging.getLogger("snowlabel.anchors")

PREVIEW_SIZE = 160


def anchor_scope(user_id: str | None) -> str:
    return (user_id or "global").strip() or "global"


def anchor_preview_key(entry_id: int, user_id: str | None = "") -> str:
    return f"anchors/{anchor_scope(user_id)}/{int(entry_id)}.webp"


def anchor_preview_bytes(img: Image.Image, mask: np.ndarray) -> bytes | None:
    """Return a square WebP preview of the masked crop, or None for empty masks."""
    m = np.asarray(mask) > 0
    if m.ndim != 2 or not m.any():
        return None

    h, w = m.shape
    ys, xs = np.where(m)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad = max(4, int(max(bw, bh) * 0.18))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)

    crop = img.convert("RGB").crop((x0, y0, x1, y1)).convert("RGBA")
    mc = Image.fromarray((m[y0:y1, x0:x1].astype(np.uint8) * 255), "L")
    tint = Image.new("RGBA", crop.size, (56, 128, 243, 82))
    crop = Image.alpha_composite(crop, Image.composite(tint, Image.new("RGBA", crop.size), mc))

    crop.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (PREVIEW_SIZE, PREVIEW_SIZE), (18, 20, 26))
    ox = (PREVIEW_SIZE - crop.width) // 2
    oy = (PREVIEW_SIZE - crop.height) // 2
    canvas.paste(crop.convert("RGB"), (ox, oy))

    buf = io.BytesIO()
    canvas.save(buf, format="WEBP", quality=82, method=4)
    return buf.getvalue()


def store_anchor_preview(cache, entry_id: int, user_id: str | None,
                         img: Image.Image, mask: np.ndarray) -> str | None:
    if cache is None or not getattr(cache, "obj", None):
        return None
    data = anchor_preview_bytes(img, mask)
    if not data:
        return None
    key = anchor_preview_key(entry_id, user_id)
    try:
        cache.obj.put_bytes(key, data, content_type="image/webp")
        return key
    except Exception as e:
        log.warning("anchor preview store failed entry=%s: %s", entry_id, e)
        return None


def delete_anchor_preview(cache, entry_id: int, user_id: str | None = "") -> None:
    if cache is None or not getattr(cache, "obj", None):
        return
    try:
        cache.obj.delete(anchor_preview_key(entry_id, user_id))
    except Exception as e:
        log.warning("anchor preview delete failed entry=%s: %s", entry_id, e)


def anchor_preview_exists(cache, entry_id: int, user_id: str | None = "") -> bool:
    if cache is None or not getattr(cache, "obj", None):
        return False
    try:
        return bool(cache.obj.exists(anchor_preview_key(entry_id, user_id)))
    except Exception:
        return False
