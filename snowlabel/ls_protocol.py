"""Label Studio protocol helpers — RLE encode/decode, prompt extraction.

Label Studio uses its own RLE format for brush masks (different from
COCO RLE!). It's a flat int array produced by JS-side run-length encoding
of the mask alpha channel. We use the official helper functions shipped
with `label-studio-sdk` / `label_studio_converter`.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("snowlabel.ls_protocol")


# ----------------------------------------------------------------------
# Brush RLE
# ----------------------------------------------------------------------
def encode_mask_as_brush_rle(mask_hw: np.ndarray) -> list[int]:
    """Encode a boolean H×W mask as Label Studio brush RLE."""
    try:
        from label_studio_sdk.converter.brush import mask2rle  # newer
    except ImportError:
        try:
            from label_studio_converter.brush import mask2rle
        except ImportError as e:
            raise ImportError(
                "label-studio-sdk or label-studio-converter is required "
                "for brush RLE encoding"
            ) from e
    # LS expects a 2D single-channel 0/255 uint8 array
    arr = (mask_hw.astype(np.uint8) * 255)
    return mask2rle(arr)


def decode_brush_rle(rle: list[int], *, width: int, height: int) -> np.ndarray:
    """Decode Label Studio brush RLE back to a boolean H×W mask."""
    try:
        from label_studio_sdk.converter.brush import decode_rle
    except ImportError:
        try:
            from label_studio_converter.brush import decode_rle
        except ImportError as e:
            raise ImportError(
                "label-studio-sdk or label-studio-converter is required "
                "for brush RLE decoding"
            ) from e
    arr = decode_rle(rle)
    # LS brush mask has 4 channels; the alpha channel encodes the mask.
    if arr.size == width * height * 4:
        arr = arr.reshape(height, width, 4)
        mask = arr[..., 3] > 0
    elif arr.size == width * height:
        mask = arr.reshape(height, width) > 0
    else:
        raise ValueError(
            f"Decoded RLE has size {arr.size}; expected {width*height} "
            f"or {width*height*4} for {width}x{height}."
        )
    return mask


# ----------------------------------------------------------------------
# Bounding box
# ----------------------------------------------------------------------
def bbox_from_mask(mask_hw: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return xyxy bbox for a boolean mask, or None if empty."""
    ys, xs = np.where(mask_hw)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ----------------------------------------------------------------------
# Interactive prompt extraction
# ----------------------------------------------------------------------
def extract_interactive_prompt(context: dict | None) -> dict | None:
    """Parse the LS `context` blob for an interactive prompt.

    Returns one of:
        {"type": "point", "x": int, "y": int}
        {"type": "box", "bbox": (x1, y1, x2, y2)}
        None  -- no prompt; caller should run auto mode.
    """
    if not context:
        return None
    results = context.get("result") or []
    if not results:
        return None

    # LS sends the cumulative list of smart prompts on the canvas — every
    # previous keypoint plus the new click. Treat the LAST item as the most
    # recent user action ("click once → predict at this click"); earlier items
    # were already processed in their own /predict calls.
    item = results[-1]
    typ = item.get("type", "")
    val = item.get("value", {}) or {}
    img_w = item.get("original_width") or val.get("original_width")
    img_h = item.get("original_height") or val.get("original_height")

    if typ in ("keypointlabels", "keypoint"):
        # LS gives coords in percent (0-100). We need absolute pixels.
        if img_w is None or img_h is None:
            log.warning("Keypoint without image dims; cannot convert to pixels.")
            return None
        x = int(round(float(val.get("x", 0)) * img_w / 100.0))
        y = int(round(float(val.get("y", 0)) * img_h / 100.0))
        return {"type": "point", "x": x, "y": y}

    if typ in ("rectanglelabels", "rectangle"):
        if img_w is None or img_h is None:
            log.warning("Rectangle without image dims; cannot convert to pixels.")
            return None
        x = float(val.get("x", 0)) * img_w / 100.0
        y = float(val.get("y", 0)) * img_h / 100.0
        w = float(val.get("width", 0)) * img_w / 100.0
        h = float(val.get("height", 0)) * img_h / 100.0
        return {
            "type": "box",
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
        }

    log.debug("Unknown interactive prompt type: %s", typ)
    return None
