"""
Deep analysis of Phase 0 results — cuts past the headline accuracy
and gives honest baselines + per-class signal.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

CSV = Path("results/phase0_predictions.csv")
ANN = Path("data/coco/annotations/instances_val2017_snowlabel.json")

df = pd.read_csv(CSV)
with open(ANN, "r", encoding="utf-8") as f:
    coco = json.load(f)

cats = {c["id"]: c["name"] for c in coco["categories"]}
cat_ids = sorted(cats.keys())
cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
idx_to_name = {i: cats[cid] for cid, i in cat_id_to_idx.items()}
n_classes = len(cat_ids)
print(f"Classes (0..{n_classes-1}): {[idx_to_name[i] for i in range(n_classes)]}")
print()

scored = df[df["correct"].notna()].copy()
scored["correct"] = scored["correct"].astype(int)
print(f"Total predictions evaluated: {len(scored)}")
print()

# --- Honest baselines ---
print("=" * 60)
print("HONEST BASELINES")
print("=" * 60)

class_counts = scored["gt"].value_counts().sort_index()
print("\nGT distribution among the evaluated predictions:")
for idx in range(n_classes):
    n = int(class_counts.get(idx, 0))
    pct = 100 * n / len(scored)
    print(f"  {idx_to_name[idx]:<10} {n:>4}  ({pct:.1f}%)")

majority_baseline = class_counts.max() / len(scored)
random_baseline = 1.0 / n_classes
overall_acc = scored["correct"].mean()

print(f"\n  Uniform random       : {random_baseline:.3f}")
print(f"  Majority class       : {majority_baseline:.3f}  (always predict "
      f"'{idx_to_name[int(class_counts.idxmax())]}')")
print(f"  ----")
print(f"  Our model overall    : {overall_acc:.3f}")
print(f"  Lift over majority   : {overall_acc - majority_baseline:+.3f}  "
      f"({100*(overall_acc - majority_baseline):+.1f} pp)")

# --- Per-class accuracy ---
print()
print("=" * 60)
print("PER-CLASS ACCURACY  (THE REAL TEST)")
print("=" * 60)
print(f"{'class':<10} {'n':>4} {'acc':>6} {'precision':>10} {'recall':>7}")
for idx in range(n_classes):
    cls_rows = scored[scored["gt"] == idx]
    if len(cls_rows) == 0:
        continue
    recall = cls_rows["correct"].mean()
    pred_rows = scored[scored["pred"] == idx]
    precision = pred_rows["correct"].mean() if len(pred_rows) > 0 else float("nan")
    print(f"  {idx_to_name[idx]:<8} {len(cls_rows):>4} "
          f"{cls_rows['correct'].mean():>6.3f} "
          f"{precision:>10.3f} "
          f"{recall:>7.3f}")

# --- Confusion matrix ---
print()
print("=" * 60)
print("CONFUSION MATRIX  (rows = GT, cols = prediction)")
print("=" * 60)
cm = pd.crosstab(scored["gt"], scored["pred"],
                 rownames=["GT"], colnames=["Pred"], dropna=False)
cm.index = [idx_to_name[i] for i in cm.index]
cm.columns = [idx_to_name[i] for i in cm.columns]
print(cm.to_string())

# --- Snowball check: per-class trend over time ---
print()
print("=" * 60)
print("SNOWBALL CHECK — does accuracy improve as memory grows, per class?")
print("=" * 60)
# Split each class's predictions into early-half vs late-half by mem_size
print(f"{'class':<10} {'early-acc':>10} {'late-acc':>9} {'lift':>7}  "
      f"{'early-n':>8} {'late-n':>8}")
for idx in range(n_classes):
    cls_rows = scored[scored["gt"] == idx].copy()
    if len(cls_rows) < 4:
        print(f"  {idx_to_name[idx]:<8}  (n={len(cls_rows)}, too few to split)")
        continue
    cls_rows = cls_rows.sort_values("mem_size")
    half = len(cls_rows) // 2
    early = cls_rows.iloc[:half]
    late = cls_rows.iloc[half:]
    print(f"  {idx_to_name[idx]:<8} {early['correct'].mean():>10.3f} "
          f"{late['correct'].mean():>9.3f} "
          f"{late['correct'].mean() - early['correct'].mean():>+7.3f}  "
          f"{len(early):>8} {len(late):>8}")

# --- High-confidence behavior ---
print()
print("=" * 60)
print("HIGH-CONFIDENCE PRECISION")
print("=" * 60)
for thresh in [0.5, 0.7, 0.85, 0.95]:
    hi = scored[scored["conf"] >= thresh]
    cov = len(hi) / len(scored)
    acc = hi["correct"].mean() if len(hi) > 0 else float("nan")
    print(f"  conf >= {thresh:.2f}:  coverage {cov:>5.1%}  acc {acc:.3f}  (n={len(hi)})")

# --- Per-class confidence (does the model KNOW when it's right?) ---
print()
print("=" * 60)
print("CONFIDENCE WHEN CORRECT vs WRONG, per class")
print("=" * 60)
print(f"{'class':<10} {'mean-conf-correct':>18} {'mean-conf-wrong':>18}")
for idx in range(n_classes):
    cls_rows = scored[scored["gt"] == idx]
    if len(cls_rows) < 2:
        continue
    c_corr = cls_rows[cls_rows["correct"] == 1]["conf"].mean()
    c_wrong = cls_rows[cls_rows["correct"] == 0]["conf"].mean()
    print(f"  {idx_to_name[idx]:<8} {c_corr:>18.3f} {c_wrong:>18.3f}")

print()
print("Done.")
