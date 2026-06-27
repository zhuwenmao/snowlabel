"""End-to-end smoke test for Phase 1 — exercises the real pipeline:

    SAM 2.1 (mask) -> DINOv2 (feature) -> SnowballMemory (store + query)

This actually downloads ~480 MB of model weights on first run. Re-runs are
cached. Use a fresh memory dir so old labels don't pollute.

Run:
    python scripts/smoke_test_phase1.py
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from snowlabel.config import SnowConfig
from snowlabel.predictor import SnowPredictor

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("smoke")

# Use a dedicated test memory dir so we don't trash the user's prod data.
TEST_ROOT = Path(__file__).resolve().parent.parent
TEST_MEM_DIR = TEST_ROOT / "data" / "memory_smoke"
if TEST_MEM_DIR.exists():
    shutil.rmtree(TEST_MEM_DIR)


def main() -> None:
    # 1. Build a predictor and point memory at a clean smoke-test dir.
    import os
    os.environ.setdefault("SNOWLABEL_DINO", "dinov2_vits14")  # small for smoke test
    predictor = SnowPredictor()
    # Re-point memory to the smoke dir so this run is isolated. Use
    # get_memory() so SNOWLABEL_MEMORY_BACKEND (numpy / milvus) is honored.
    from snowlabel.memory import get_memory
    predictor._memory = get_memory(TEST_MEM_DIR, dim=predictor.dino.embed_dim)
    print(f"  Memory backend: {type(predictor._memory).__name__}")

    # 2. Find an image with a known centroid for a click test.
    coco_ann = TEST_ROOT / "data" / "coco" / "annotations" / "instances_val2017_snowlabel.json"
    img_dir = TEST_ROOT / "data" / "coco" / "val2017"
    if not coco_ann.exists():
        raise SystemExit(f"Missing {coco_ann}. Run scripts/prepare_data.py first.")
    coco = COCO(str(coco_ann))
    cat_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    # 3. Loop over up to 12 images, click on the centroid of the largest
    # ground-truth instance, see if SAM gives a similar mask and how the
    # snowball memory evolves.
    image_ids = sorted(coco.getImgIds())[:12]
    results = []
    for i, img_id in enumerate(image_ids):
        info = coco.loadImgs(img_id)[0]
        img_path = img_dir / info["file_name"]
        if not img_path.exists():
            continue
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                if a["area"] > 32 * 32 and not a.get("iscrowd", 0)]
        if not anns:
            continue
        # Pick the largest instance and click its bbox center.
        ann = max(anns, key=lambda a: a["area"])
        gt_label = cat_name[ann["category_id"]]
        bx, by, bw, bh = ann["bbox"]
        cx, cy = int(bx + bw / 2), int(by + bh / 2)

        img = Image.open(img_path).convert("RGB")
        t0 = time.time()
        pred = predictor.predict_from_point(img, cx, cy)
        dt = time.time() - t0

        if pred is None:
            log.warning("[%d] SAM returned no mask for %s", i, info["file_name"])
            continue

        # Did we predict the right label?
        correct = (pred.label == gt_label) if pred.label else None
        log.info(
            "[%2d] %s  click=(%d,%d)  gt=%s  pred=%s  conf=%.2f  tier=%s  mem=%d  dt=%.2fs",
            i, info["file_name"], cx, cy, gt_label, pred.label,
            pred.confidence, pred.tier, len(predictor.memory), dt,
        )
        results.append({
            "img_id": img_id, "gt": gt_label, "pred": pred.label,
            "conf": pred.confidence, "tier": pred.tier,
            "correct": correct, "mem_size_before": len(predictor.memory),
        })

        # Always teach the predictor with the ground truth (simulates the
        # user submitting the correction). Use the SAM-produced mask, not the
        # GT mask, so we mimic the real LS webhook path.
        predictor.remember(img, pred.mask, gt_label, image_uri=str(img_path),
                           verified=True)

    # 4. Headline summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    with_pred = [r for r in results if r["correct"] is not None]
    n_pred = len(with_pred)
    n_corr = sum(1 for r in with_pred if r["correct"])
    print(f"Predictions (with non-empty memory): {n_pred}")
    if n_pred:
        print(f"Correct                              : {n_corr} ({100*n_corr/n_pred:.0f}%)")
    print(f"Final memory size                    : {len(predictor.memory)}")
    print(f"Classes seen                         : {predictor.memory.classes}")
    print(f"Memory dir                           : {TEST_MEM_DIR}")
    print()
    print("Phase 1 pipeline END-TO-END: PASSED" if n_pred and n_corr >= n_pred / 2
          else "Phase 1 pipeline END-TO-END: needs investigation")


if __name__ == "__main__":
    main()
