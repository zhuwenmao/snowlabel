"""Analyze prediction CSVs to recommend per-backbone confidence thresholds.

We answer the practical UX question:
    "At what confidence can SnowLabel auto-apply a label without making the
     user lose trust?"

Method:
    For each backbone (v2, v3):
      - Sort predictions by descending confidence.
      - For each confidence threshold, compute (coverage, precision).
      - Pick:
          AUTO_APPLY = smallest threshold where precision >= 0.97
          SUGGEST    = smallest threshold where precision >= 0.85
    Save to results/thresholds/ + write a Markdown summary.

Usage:
    python scripts/tune_thresholds.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "results" / "v2_vs_v3"
OUT_DIR = PROJECT_ROOT / "results" / "thresholds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Quality targets — chosen so "auto-apply" feels safe.
AUTO_PRECISION = 0.97
SUGGEST_PRECISION = 0.85


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["correct"].notna()].copy()
    df["correct"] = df["correct"].astype(int)
    return df


def precision_coverage_curve(df: pd.DataFrame) -> pd.DataFrame:
    """For every confidence threshold, compute (coverage, precision)."""
    sorted_df = df.sort_values("conf", ascending=False).reset_index(drop=True)
    cum_correct = sorted_df["correct"].cumsum().to_numpy()
    cum_count = np.arange(1, len(sorted_df) + 1, dtype=float)
    precision = cum_correct / cum_count
    coverage = cum_count / len(df)
    return pd.DataFrame({
        "threshold": sorted_df["conf"].to_numpy(),
        "precision": precision,
        "coverage": coverage,
        "n": cum_count.astype(int),
    })


def pick_threshold(pc: pd.DataFrame, min_prec: float) -> dict:
    """Pick the LOWEST threshold whose precision is >= min_prec.

    Use .loc with idxmax (which returns the original DataFrame index),
    not .iloc (which interprets that index as a positional integer).
    """
    feasible = pc[pc["precision"] >= min_prec]
    if not len(feasible):
        best = pc.loc[pc["precision"].idxmax()]
        return {
            "ok": False,
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "coverage": float(best["coverage"]),
            "note": f"target {min_prec} not reachable; best is {best['precision']:.3f}",
        }
    best = feasible.loc[feasible["coverage"].idxmax()]
    return {
        "ok": True,
        "threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "coverage": float(best["coverage"]),
    }


def analyze_backbone(name: str, df: pd.DataFrame) -> dict:
    pc = precision_coverage_curve(df)
    return {
        "backbone": name,
        "n_predictions": int(len(df)),
        "overall_accuracy": float(df["correct"].mean()),
        "conf_distribution": {
            "mean_correct": float(df[df["correct"] == 1]["conf"].mean()),
            "mean_wrong": float(df[df["correct"] == 0]["conf"].mean()),
            "std_correct": float(df[df["correct"] == 1]["conf"].std()),
            "std_wrong": float(df[df["correct"] == 0]["conf"].std()),
        },
        "auto_apply": pick_threshold(pc, AUTO_PRECISION),
        "suggest": pick_threshold(pc, SUGGEST_PRECISION),
        "_pc": pc,
    }


def plot_curves(analyses: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"v2": "#1f77b4", "v3": "#d62728"}

    for k, a in analyses.items():
        pc = a["_pc"]
        axes[0].plot(pc["coverage"], pc["precision"], color=colors[k],
                     lw=2, label=f"{k} ({a['backbone']})")
        for name, marker in [("auto_apply", "o"), ("suggest", "s")]:
            t = a[name]
            axes[0].scatter(
                [t["coverage"]], [t["precision"]],
                color=colors[k], marker=marker, s=80, edgecolor="black",
                zorder=5,
            )
    axes[0].axhline(AUTO_PRECISION, color="green", ls="--", lw=1, alpha=0.5,
                    label=f"auto-apply target (p={AUTO_PRECISION})")
    axes[0].axhline(SUGGEST_PRECISION, color="orange", ls="--", lw=1, alpha=0.5,
                    label=f"suggest target (p={SUGGEST_PRECISION})")
    axes[0].set_xlabel("Coverage (fraction of predictions kept)")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision vs coverage")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].set_ylim(0, 1.02)

    for k, a in analyses.items():
        pc = a["_pc"]
        axes[1].plot(pc["threshold"], pc["precision"], color=colors[k], lw=2,
                     label=f"{k}")
        for name, ls in [("auto_apply", "--"), ("suggest", ":")]:
            t = a[name]
            axes[1].axvline(t["threshold"], color=colors[k], ls=ls, lw=1, alpha=0.6)
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Precision at threshold")
    axes[1].set_title("Precision vs threshold")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"Saved {out_path}")


def render_markdown(analyses: dict, out_path: Path) -> None:
    L = []
    L.append("# SnowLabel — recommended confidence thresholds\n")
    L.append("Derived empirically from `results/v2_vs_v3/predictions_*.csv` "
             "(458 predictions per backbone, balanced 5-class COCO subset).\n")
    L.append(f"**Quality targets**: auto-apply ≥ {AUTO_PRECISION} precision, "
             f"suggest ≥ {SUGGEST_PRECISION} precision.\n")
    L.append("## Recommended defaults per backbone\n")
    L.append("| Backbone | mean conf (correct/wrong) | "
             "AUTO threshold (prec / coverage) | "
             "SUGGEST threshold (prec / coverage) |")
    L.append("|----------|--------|---------|---------|")
    for k, a in analyses.items():
        cd = a["conf_distribution"]
        au = a["auto_apply"]; su = a["suggest"]
        L.append(
            f"| **{a['backbone']}** "
            f"| {cd['mean_correct']:.3f} / {cd['mean_wrong']:.3f} "
            f"| **{au['threshold']:.2f}** "
            f"({au['precision']:.3f} / {au['coverage']:.1%}) "
            f"| **{su['threshold']:.2f}** "
            f"({su['precision']:.3f} / {su['coverage']:.1%}) |"
        )
    L.append("\n## How to apply\n")
    L.append("```powershell")
    v2 = analyses.get("v2")
    if v2:
        L.append("# DINOv2 (default open stack)")
        L.append(f"$env:SNOWLABEL_CONF_AUTO    = '{v2['auto_apply']['threshold']:.2f}'")
        L.append(f"$env:SNOWLABEL_CONF_SUGGEST = '{v2['suggest']['threshold']:.2f}'")
    v3 = analyses.get("v3")
    if v3:
        L.append("\n# DINOv3-L")
        L.append(f"$env:SNOWLABEL_CONF_AUTO    = '{v3['auto_apply']['threshold']:.2f}'")
        L.append(f"$env:SNOWLABEL_CONF_SUGGEST = '{v3['suggest']['threshold']:.2f}'")
    L.append("```\n")
    L.append("## Notes\n")
    L.append("- DINOv3-L has *flatter* cosine distribution than DINOv2-S "
             "(higher embedding dim → tighter sims). Thresholds do not transfer "
             "1:1; tune separately.")
    L.append("- 'Coverage' = fraction of predictions whose confidence ≥ threshold "
             "(the fraction the UI would auto-apply).")
    L.append("- Precision target 0.97 keeps user trust: roughly 3 of 100 "
             "auto-applied labels are wrong.")
    L.append("\n## Reproducing\n")
    L.append("```powershell")
    L.append("python scripts/compare_v2_vs_v3.py --v3-path '<dinov3-snapshot>'")
    L.append("python scripts/tune_thresholds.py")
    L.append("```\n")
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Saved {out_path}")


def main() -> None:
    v2_csv = IN_DIR / "predictions_v2.csv"
    v3_csv = IN_DIR / "predictions_v3.csv"
    if not v2_csv.exists() or not v3_csv.exists():
        raise SystemExit("Run compare_v2_vs_v3.py first; missing CSVs.")

    analyses = {
        "v2": analyze_backbone("DINOv2-S", load(v2_csv)),
        "v3": analyze_backbone("DINOv3-L", load(v3_csv)),
    }
    plot_curves(analyses, OUT_DIR / "precision_coverage.png")

    summary = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in analyses.items()
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    render_markdown(analyses, OUT_DIR / "THRESHOLDS.md")

    print("\n" + "=" * 60)
    print("Recommended thresholds")
    print("=" * 60)
    for k, a in analyses.items():
        au = a["auto_apply"]; su = a["suggest"]
        print(f"  {a['backbone']:>10}: AUTO={au['threshold']:.2f} "
              f"(prec {au['precision']:.3f} cov {au['coverage']:.1%}), "
              f"SUGGEST={su['threshold']:.2f} "
              f"(prec {su['precision']:.3f} cov {su['coverage']:.1%})")


if __name__ == "__main__":
    main()
