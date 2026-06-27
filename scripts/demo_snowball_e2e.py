"""Headless end-to-end snowball demo — the real pipeline the Label Studio UI
drives, run as a reproducible script so the "gets better as you label" curve is
quantified without needing the browser.

For each image (interleaved across the 5 classes): click the GT instance center
-> SAM2.1 mask -> DINOv3-L feature -> snowball memory predict (tier/conf) ->
record correctness vs GT -> teach the memory the GT label (simulating the user
confirming). Then plot rolling accuracy + confidence-tier evolution.

This uses REAL SAM masks (not COCO GT masks like Phase 0), so it validates the
full production pipeline end to end.

Run with the production stack:
    $env:SNOWLABEL_DINO=...dinov3-vitl16...; $env:SNOWLABEL_SAM="sam2.1_t.pt"
    python scripts/demo_snowball_e2e.py
"""
import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from snowlabel.config import get_config  # noqa: E402
from snowlabel.memory import SnowballMemory  # noqa: E402
from snowlabel.predictor import SnowPredictor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_PER = 25
CLASSES = ["cat", "dog", "bird", "bottle", "car"]
WINDOW = 20

cfg = get_config()
print("dino:", str(cfg.dino_variant).split("\\")[-1], "| sam:", cfg.sam_variant,
      "| res:", cfg.input_resolution)
print("cold-start gate: auto>=", cfg.min_support_auto, " suggest>=", cfg.min_support_suggest)

coco = COCO(str(ROOT / "data" / "coco" / "annotations" / "instances_val2017_snowlabel.json"))
per_class = {}
for c in CLASSES:
    cid = coco.getCatIds(catNms=[c])[0]
    picked = []
    for iid in coco.getImgIds(catIds=[cid]):
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=[cid]))
                if not a.get("iscrowd", 0) and a["area"] >= 64 * 64]
        if anns:
            picked.append((iid, max(anns, key=lambda a: a["area"]), c))
        if len(picked) >= N_PER:
            break
    per_class[c] = picked
    print(f"  {c}: {len(picked)} images")

# Interleave across classes
seq = []
for i in range(N_PER):
    for c in CLASSES:
        if i < len(per_class[c]):
            seq.append(per_class[c][i])
print("total sequence:", len(seq))

pred = SnowPredictor()
mem_dir = ROOT / "data" / "memory_e2e"
if mem_dir.exists():
    shutil.rmtree(mem_dir)
pred._memory = SnowballMemory(mem_dir, dim=pred.dino.embed_dim)

rows = []
t0 = time.time()
for step, (iid, ann, gt) in enumerate(seq):
    info = coco.loadImgs(iid)[0]
    img = Image.open(ROOT / "data" / "coco" / "val2017" / info["file_name"]).convert("RGB")
    bx, by, bw, bh = ann["bbox"]
    cx, cy = int(bx + bw / 2), int(by + bh / 2)
    p = pred.predict_from_point(img, cx, cy)
    if p is None:
        continue
    correct = int(p.label == gt) if p.label else 0
    rows.append({"step": step, "gt": gt, "pred": p.label or "-",
                 "conf": round(p.confidence, 3), "tier": p.tier,
                 "correct": correct, "mem": len(pred.memory)})
    pred.remember(img, p.mask, gt, image_uri=info["file_name"], verified=True)
    if step % 15 == 0:
        print(f"  step {step:3d}/{len(seq)}  {gt}->{p.label} ({p.tier})  mem={len(pred.memory)}")
print(f"done {len(rows)} steps in {time.time()-t0:.0f}s")

out = ROOT / "results" / "demo_e2e"
out.mkdir(parents=True, exist_ok=True)
with open(out / "e2e_predictions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Rolling accuracy over predictions that produced a label
labeled = [r for r in rows if r["pred"] != "-"]
flags = [r["correct"] for r in labeled]
acc_x, acc_y = [], []
for i in range(len(labeled)):
    lo = max(0, i - WINDOW + 1)
    acc_x.append(i)
    acc_y.append(float(np.mean(flags[lo:i + 1])) * 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(acc_x, acc_y, color="#1976D2", lw=2)
axes[0].axhline(20, ls="--", color="gray", label="random (5-class)")
axes[0].set_ylim(0, 105)
axes[0].set_title("Rolling accuracy — DINOv3-L + SAM2.1 real masks (snowball)")
axes[0].set_xlabel("labeling step")
axes[0].set_ylabel(f"accuracy % (window={WINDOW})")
axes[0].legend()
axes[0].grid(alpha=0.3)

tiers = ["silent", "suggest", "auto"]
colors = {"silent": "#BDBDBD", "suggest": "#FFB300", "auto": "#388E3C"}
binsize = 20
bins = defaultdict(lambda: defaultdict(int))
for i, r in enumerate(rows):
    bins[i // binsize][r["tier"]] += 1
bx = sorted(bins.keys())
bottoms = [0] * len(bx)
for t in tiers:
    vals = [bins[b][t] for b in bx]
    axes[1].bar([b * binsize for b in bx], vals, bottom=bottoms,
                width=binsize * 0.85, label=t, color=colors[t])
    bottoms = [bottoms[i] + vals[i] for i in range(len(bx))]
axes[1].set_title("Confidence-tier evolution (cold-start -> auto)")
axes[1].set_xlabel("labeling step")
axes[1].set_ylabel(f"count per {binsize}-step bin")
axes[1].legend()
plt.tight_layout()
plt.savefig(out / "snowball_e2e.png", dpi=110)

overall = np.mean(flags) * 100 if flags else 0
late = np.mean(flags[len(flags) // 2:]) * 100 if flags else 0
auto_rows = [r for r in rows if r["tier"] == "auto"]
auto_acc = np.mean([r["correct"] for r in auto_rows]) * 100 if auto_rows else 0
print("\n=== SUMMARY ===")
print(f"overall accuracy : {overall:.1f}%")
print(f"late-half accuracy: {late:.1f}%  (snowball effect)")
print(f"auto-tier accuracy: {auto_acc:.1f}%  (n={len(auto_rows)} auto predictions)")
print(f"saved: {out}/snowball_e2e.png + e2e_predictions.csv")
