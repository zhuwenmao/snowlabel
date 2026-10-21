"""Profile the uncached Everything pipeline without starting the web server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn):
    _sync()
    started = time.perf_counter()
    value = fn()
    _sync()
    return value, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-masks", type=int, default=20)
    parser.add_argument("--dino", default="weights/facebookdinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--parts-grid", type=int, default=4)
    parser.add_argument("--cache-store", action="store_true")
    args = parser.parse_args()

    os.environ["SNOWLABEL_SAM_POINTS_PER_SIDE"] = str(args.points_per_side)
    os.environ["SNOWLABEL_DINO"] = args.dino
    os.environ["SNOWLABEL_RES"] = str(args.resolution)
    os.environ["SNOWLABEL_PARTS_GRID"] = str(args.parts_grid)

    from snowlabel.features import MaskFeatureExtractor
    from snowlabel.models import DinoFeatureExtractor, SamSegmenter

    image = Image.open(args.image).convert("RGB")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_started = time.perf_counter()
    sam = SamSegmenter()
    _, sam_load_s = _timed(sam._ensure_loaded)
    masks, sam_generate_s = _timed(
        lambda: sam.predict_all(image, max_masks=args.max_masks)
    )

    dino = DinoFeatureExtractor(
        variant=args.dino,
        input_res=args.resolution,
    )
    features = MaskFeatureExtractor(dino, p_grid=args.parts_grid)
    _, dino_load_s = _timed(dino._ensure_loaded)
    encoded, dino_features_s = _timed(
        lambda: features.features_batch(image, [item.mask for item in masks])
    )
    cache_store_s = None
    if args.cache_store:
        os.environ["SNOWLABEL_IMGCACHE"] = "1"
        os.environ.setdefault("SNOWLABEL_MEMORY_DIR", "data/memory_public")
        from snowlabel.config import get_config
        from snowlabel.imgcache.cache import ImageCache, image_sha256

        image_bytes = args.image.read_bytes()
        profile = (
            f"auto_impl=v2|sam=sam2.1_t.pt|max={args.max_masks}|"
            f"pps={args.points_per_side}|crop=0|conf=0.8|stab=0.9"
        )
        image_hash = hashlib.sha256(
            f"{image_sha256(image_bytes)}|{profile}".encode("utf-8")
        ).hexdigest()
        cache = ImageCache(get_config())
        items = [
            {
                "mask": mask.mask,
                "bbox": mask.bbox,
                "v_whole": feature.v_whole,
                "v_parts": feature.v_parts,
                "geom": feature.geom,
                "pred_label": "",
                "tier": "silent",
                "confidence": 0.0,
            }
            for mask, feature in zip(masks, encoded)
        ]
        _, cache_store_s = _timed(
            lambda: cache.store(
                image_hash,
                items,
                img_bytes=image_bytes,
                ext=args.image.suffix,
                orig_name=args.image.name,
                width=image.width,
                height=image.height,
            )
        )
    total_s = time.perf_counter() - total_started

    peak_mib = None
    allocated_mib = None
    if torch.cuda.is_available():
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        allocated_mib = torch.cuda.memory_allocated() / 1024**2

    result = {
        "image": str(args.image),
        "image_size": list(image.size),
        "points_per_side": args.points_per_side,
        "max_masks": args.max_masks,
        "mask_count": len(masks),
        "feature_count": len(encoded),
        "timings_s": {
            "sam_load": round(sam_load_s, 3),
            "sam_generate": round(sam_generate_s, 3),
            "dino_load": round(dino_load_s, 3),
            "dino_features": round(dino_features_s, 3),
            "cache_store": round(cache_store_s, 3) if cache_store_s is not None else None,
            "total": round(total_s, 3),
        },
        "gpu_memory_mib": {
            "peak_allocated": round(peak_mib, 1) if peak_mib is not None else None,
            "allocated_at_end": round(allocated_mib, 1) if allocated_mib is not None else None,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
