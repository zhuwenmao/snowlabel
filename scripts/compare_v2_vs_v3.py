"""Apples-to-apples comparison: DINOv2-Small vs DINOv3-Large on the
same balanced COCO subset used in Phase 0.

Runs the Phase 0 algorithm twice — once with each backbone — and writes
side-by-side plots into results/v2_vs_v3/.

Why this exists:
    The Phase 0 notebook showed +69.7pp lift with DINOv2-S. The user has now
    plugged in DINOv3-L (1024-dim, register tokens, larger pretraining set).
    Is it actually better for our retrieval-based labeling? This script gives
    the honest answer.

Usage:
    python scripts/compare_v2_vs_v3.py \
        --v3-path "D:\\project\\cc_workspace\\LabelAnything\\weights\\facebookdinov3-vitl16-pretrain-lvd1689m"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm.auto import tqdm

# ---- Config (matches notebooks/phase0_validation.py) ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANN_PATH = PROJECT_ROOT / "data" / "coco" / "annotations" / "instances_val2017_snowlabel.json"
IMG_DIR = PROJECT_ROOT / "data" / "coco" / "val2017"
SEED = 42
K_NEIGHBORS = 5
MIN_MASK_PIXELS = 32 * 32
MAX_INSTANCES_PER_IMAGE = 8
MAX_INSTANCES_PER_CLASS = 100
# INPUT_RES / OUT_DIR are filled in by main() once we know --res.
INPUT_RES = 224  # default, overridden by CLI


def decode_coco_mask(ann: dict, height: int, width: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


def topk_vote(query: np.ndarray, memory: np.ndarray, labels: np.ndarray, k: int):
    sims = memory @ query
    k_eff = min(k, len(memory))
    if k_eff == 1:
        top_idx = np.array([int(np.argmax(sims))])
    else:
        top_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
    top_sims = sims[top_idx]
    top_lbls = labels[top_idx]
    weights = np.clip(top_sims, 0.0, None)
    scores: dict[int, float] = defaultdict(float)
    for w, l in zip(weights, top_lbls):
        scores[int(l)] += float(w)
    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    total = sum(scores.values())
    conf = scores[pred] / total if total > 0 else 0.0
    # Also compute the best raw cosine similarity (margin-free).
    best_sim = float(top_sims[0])
    return pred, conf, best_sim


def run_one_experiment(
    *,
    label: str,
    dino_variant: str,
    coco: COCO,
    out_csv: Path,
) -> pd.DataFrame:
    """Run the full snowball experiment with one DINO backbone."""
    print(f"\n{'='*70}\n  Running experiment: {label}  ({dino_variant})\n{'='*70}")

    # Pass variant EXPLICITLY — never rely on env vars between sequential
    # runs (config defaults are frozen per-instance, but explicit args
    # bypass that whole question).
    from snowlabel.models import DinoFeatureExtractor
    dino = DinoFeatureExtractor(variant=dino_variant, input_res=INPUT_RES)
    print(f"  Loading {dino_variant} ...")
    _ = dino.embed_dim  # trigger load
    print(f"  Loaded. embed_dim={dino.embed_dim}")

    # Cat metadata
    cat_ids = sorted(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_name = {i: cat_id_to_name[cid] for cid, i in cat_id_to_idx.items()}
    n_classes = len(cat_ids)

    # Build the same balanced record set as the Phase 0 notebook.
    image_ids = sorted(coco.getImgIds())
    records = []
    for img_id in image_ids:
        info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        keep = [
            a for a in anns
            if not a.get("iscrowd", 0) and a["area"] >= MIN_MASK_PIXELS
        ]
        if not keep:
            continue
        keep = keep[:MAX_INSTANCES_PER_IMAGE]
        records.append((img_id, info, keep))

    random.Random(SEED).shuffle(records)
    per_class_seen: Counter = Counter()
    balanced = []
    for img_id, info, anns in records:
        kept = []
        for a in anns:
            cls = cat_id_to_idx[a["category_id"]]
            if per_class_seen[cls] < MAX_INSTANCES_PER_CLASS:
                kept.append(a)
                per_class_seen[cls] += 1
        if kept:
            balanced.append((img_id, info, kept))
    records = balanced
    print(f"  Class distribution: {dict(per_class_seen)}")
    print(f"  Total instances: {sum(len(r[2]) for r in records)}")

    # Run the experiment.
    mem_vecs: list[np.ndarray] = []
    mem_lbls: list[int] = []
    rows: list[dict] = []
    for img_id, info, anns in tqdm(records, desc=label):
        img_path = IMG_DIR / info["file_name"]
        if not img_path.exists():
            continue
        img_pil = Image.open(img_path).convert("RGB")
        H, W = info["height"], info["width"]
        tokens = dino.encode_image(img_pil)
        for ann in anns:
            mask = decode_coco_mask(ann, H, W)
            feat = dino.pool_by_mask(tokens, mask)
            gt = cat_id_to_idx[ann["category_id"]]
            if len(mem_vecs) >= 1:
                M = np.stack(mem_vecs)
                L = np.asarray(mem_lbls)
                pred, conf, best_sim = topk_vote(feat, M, L, K_NEIGHBORS)
                correct = int(pred == gt)
            else:
                pred, conf, best_sim, correct = -1, 0.0, 0.0, None
            rows.append({
                "img_id": img_id, "gt": gt, "gt_name": idx_to_name[gt],
                "pred": pred,
                "pred_name": idx_to_name.get(pred, ""),
                "conf": conf, "best_sim": best_sim,
                "mem_size": len(mem_vecs), "correct": correct,
            })
            mem_vecs.append(feat)
            mem_lbls.append(gt)

    # Drop the cold-start first row (no memory) for accuracy reporting.
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    scored = df[df["correct"].notna()].copy()
    print(f"  Final memory: {len(mem_vecs)}  /  overall accuracy: {scored['correct'].mean():.3f}")
    # Free GPU before next model loads.
    del dino, mem_vecs, mem_lbls
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return df


def plot_comparison(df_v2: pd.DataFrame, df_v3: pd.DataFrame, out_path: Path) -> None:
    """Two snowball curves on one chart for direct comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, df, label, color in [
        (axes[0], df_v2, "DINOv2-S", "#1f77b4"),
        (axes[0], df_v3, "DINOv3-L", "#d62728"),
    ]:
        s = df[df["correct"].notna()].copy()
        s["correct"] = s["correct"].astype(int)
        window = max(20, len(s) // 30)
        s["roll"] = s["correct"].rolling(window=window, min_periods=1).mean()
        axes[0].plot(s["mem_size"], s["roll"], lw=2, color=color,
                     label=f"{label} rolling acc")

    n_classes_v2 = max(df_v2["gt"].max() + 1, 5)
    axes[0].axhline(1.0 / n_classes_v2, color="black", ls=":", lw=1.2,
                    label=f"random baseline (1/{n_classes_v2})")
    axes[0].set_xlabel("Memory size at prediction time")
    axes[0].set_ylabel("Accuracy on next instance")
    axes[0].set_title("Snowball curve — DINOv2-S vs DINOv3-L")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    # Side-by-side per-class accuracy bars
    classes = sorted(df_v2["gt_name"].unique())
    v2_acc = [df_v2[(df_v2["correct"].notna()) & (df_v2["gt_name"] == c)]
              ["correct"].mean() for c in classes]
    v3_acc = [df_v3[(df_v3["correct"].notna()) & (df_v3["gt_name"] == c)]
              ["correct"].mean() for c in classes]
    x = np.arange(len(classes))
    w = 0.38
    axes[1].bar(x - w / 2, v2_acc, w, label="DINOv2-S", color="#1f77b4")
    axes[1].bar(x + w / 2, v3_acc, w, label="DINOv3-L", color="#d62728")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=20)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per-class accuracy")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()
    for i, (a, b) in enumerate(zip(v2_acc, v3_acc)):
        axes[1].text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        axes[1].text(i + w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  Saved comparison plot -> {out_path}")


def main() -> None:
    global INPUT_RES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-variant", default="dinov2_vits14",
                    help="DINOv2 short name (torch.hub)")
    ap.add_argument("--v3-path", required=True,
                    help="Absolute path to DINOv3 HF snapshot directory")
    ap.add_argument("--res", type=int, default=224,
                    help="Input resolution; must be divisible by patch size of "
                         "each backbone (14 for v2, 16 for v3). Skips v2 if not "
                         "divisible by 14.")
    ap.add_argument("--skip-v2", action="store_true",
                    help="Run v3 only (faster; useful for resolution sweeps).")
    args = ap.parse_args()

    INPUT_RES = args.res
    out_dir = PROJECT_ROOT / "results" / (
        "v2_vs_v3" if args.res == 224 else f"v2_vs_v3_res{args.res}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ANN_PATH.exists():
        raise SystemExit(f"Missing {ANN_PATH}. Run prepare_data.py first.")
    if not Path(args.v3_path).is_dir():
        raise SystemExit(f"--v3-path is not a directory: {args.v3_path}")
    if args.res % 16 != 0:
        raise SystemExit(f"--res must be divisible by 16 for DINOv3, got {args.res}")

    skip_v2 = args.skip_v2 or (args.res % 14 != 0)
    if skip_v2 and not args.skip_v2:
        print(f"Note: --res {args.res} not divisible by 14, skipping DINOv2 run.")

    coco = COCO(str(ANN_PATH))

    v2_csv = out_dir / "predictions_v2.csv"
    v3_csv = out_dir / "predictions_v3.csv"

    if skip_v2:
        df_v2 = None
    else:
        df_v2 = run_one_experiment(
            label="DINOv2-S", dino_variant=args.v2_variant,
            coco=coco, out_csv=v2_csv,
        )
    df_v3 = run_one_experiment(
        label="DINOv3-L", dino_variant=args.v3_path,
        coco=coco, out_csv=v3_csv,
    )

    # Quick textual summary
    def _summary(df: pd.DataFrame) -> dict:
        s = df[df["correct"].notna()].copy()
        s["correct"] = s["correct"].astype(int)
        per_cls = (s.groupby("gt_name")["correct"].mean()).to_dict()
        return {
            "n": int(len(s)),
            "overall_acc": float(s["correct"].mean()),
            "late_acc": float(s.tail(max(50, len(s) // 10))["correct"].mean()),
            "per_class": {k: round(v, 3) for k, v in per_cls.items()},
        }

    out = {"res": args.res, "v3": _summary(df_v3)}
    if df_v2 is not None:
        out["v2"] = _summary(df_v2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(json.dumps(out, indent=2))

    if df_v2 is not None:
        plot_comparison(df_v2, df_v3, out_dir / "comparison.png")
    print(f"\nDone. See {out_dir}/")


if __name__ == "__main__":
    main()
