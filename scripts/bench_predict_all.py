"""Time 'segment everything' (SAM auto-mask + batched crop-encode) per model.

The supervision benchmark only measures single clicks; this covers the other
interactive path so a SAM swap decision sees the full UX cost.
Usage: python scripts/bench_predict_all.py sam2.1_t.pt sam2.1_b.pt
"""
import os
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES = sorted((ROOT / "demo_images").glob("*.jpg"))[:3]


def bench(variant: str) -> None:
    os.environ["SNOWLABEL_SAM"] = variant
    # fresh predictor per variant (config is read per-instantiation)
    from snowlabel.predictor import SnowPredictor
    pred = SnowPredictor()
    out = []
    for i, p in enumerate(IMAGES):
        img = Image.open(p).convert("RGB")
        t0 = time.time()
        masks = pred.predict_all(img, max_masks=20)
        dt = time.time() - t0
        out.append((p.name, len(masks), dt))
    # first call includes model load; report it separately
    print(f"--- {variant} ---")
    for j, (name, n, dt) in enumerate(out):
        tag = " (incl. model load)" if j == 0 else ""
        print(f"  {name}: {n} masks in {dt:.1f}s{tag}")
    warm = [dt for _, _, dt in out[1:]]
    if warm:
        print(f"  warm avg: {sum(warm)/len(warm):.1f}s")


if __name__ == "__main__":
    for v in sys.argv[1:]:
        bench(v)
