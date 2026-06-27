"""Smoke-test the web labeling tool wiring WITHOUT loading SAM/DINO.

Verifies the cheap, model-independent parts so a typo doesn't surface only
after a 15s model load:
  - dataset_dir resolves to a real folder that actually contains images
  - the mask <-> PNG base64 round-trip is lossless
  - the /api router builds and exposes the four expected routes

Run:  python scripts/smoke_webtool.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from snowlabel import web_api  # noqa: E402
from snowlabel.config import get_config  # noqa: E402
from snowlabel.predictor import SnowPredictor  # noqa: E402

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


def main() -> None:
    cfg = get_config()
    print("dataset_dir :", cfg.dataset_dir)
    imgs = ([f for f in os.listdir(cfg.dataset_dir) if f.lower().endswith(_EXTS)]
            if cfg.dataset_dir.is_dir() else [])
    print("n_images    :", len(imgs), "| sample:", imgs[:2])
    assert imgs, "dataset_dir has no images — set SNOWLABEL_DATASET_DIR"

    m = np.zeros((23, 31), bool)
    m[5:18, 6:27] = True
    b = web_api.mask_to_png_b64(m)
    m2 = web_api.png_b64_to_mask(b)
    print("mask rt     :", m.shape == m2.shape, "| equal:", bool((m == m2).all()))
    assert m.shape == m2.shape and (m == m2).all()

    router = web_api.build_router(SnowPredictor(cfg))
    paths = sorted({r.path for r in router.routes})
    print("api routes  :", paths)
    for need in ("/api/dataset", "/api/image/{image_id}",
                 "/api/segment", "/api/commit"):
        assert need in paths, f"missing route {need}"

    print("\nSMOKE OK — web tool wiring is sound (models not loaded).")


if __name__ == "__main__":
    main()
