"""Metrics, gate verdicts, baseline regression comparison, and report output.

Run artifacts land in results/supervise/<ts>_<tag>/:
    predictions.csv  every step of both phases
    metrics.json     all metrics + full config snapshot (runs without their
                     config are not comparable)
    curves.png       rolling accuracy / tier evolution / auto quality
    REPORT.md        human-readable scorecard
history.csv (one line per run) tracks the trend; baseline/ holds the
promoted reference run that --compare diffs against.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .effort import effort_saving

TIER_COLORS = {"silent": "#BDBDBD", "suggest": "#FFB300", "auto": "#388E3C"}
HISTORY_FIELDS = ["timestamp", "tag", "n_snowball", "auto_precision_late",
                  "late_auto_coverage", "poison_rate", "openset_false_auto",
                  "effort_saving_late", "latency_p95_ms", "mask_miou", "verdict"]


# ------------------------------------------------------------- metrics
def _mean(vals) -> float | None:
    vals = list(vals)
    return round(float(np.mean(vals)), 4) if vals else None


def compute_metrics(rows_a: list[dict], rows_b: list[dict], gates_cfg: dict,
                    snapshot: dict, tag: str) -> dict:
    costs = gates_cfg["effort_costs"]
    late = rows_a[len(rows_a) // 2:]
    # Label verdicts only count where the mask is valid (IoU >= 0.5): on a bad
    # cut the user repairs the mask anyway, so the label was never trusted —
    # that cost lands in the effort model / mask_fail_rate instead.
    auto_late = [r for r in late if r["tier"] == "auto" and r["iou"] >= 0.5]
    labeled_late = [r for r in late if r["tier"] != "silent" and r["iou"] >= 0.5]
    lat = sorted(r["latency_ms"] for r in rows_a + rows_b)

    per_class: dict[str, list[int]] = defaultdict(list)
    for r in labeled_late:
        per_class[r["gt"]].append(r["correct"])

    es_late = effort_saving(late, costs)
    es_all = effort_saving(rows_a, costs)
    return {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": snapshot,
        "n_snowball": len(rows_a),
        "n_openset": len(rows_b),
        # --- gated metrics ---
        "auto_precision_late": _mean(r["correct"] for r in auto_late),
        "n_auto_late": len(auto_late),    # for noise-aware regression check
        "poison_rate": round(sum(1 for r in rows_a
                                 if r["tier"] == "auto" and not r["correct"]
                                 and r["iou"] >= 0.5)
                             / max(1, len(rows_a)), 4),
        "openset_false_auto": round(sum(1 for r in rows_b if r["tier"] == "auto")
                                    / max(1, len(rows_b)), 4),
        "openset_false_any": round(sum(1 for r in rows_b
                                       if r["tier"] in ("auto", "suggest"))
                                   / max(1, len(rows_b)), 4),
        "effort_saving_late": None if es_late is None else round(es_late, 4),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1) if lat else None,
        # --- descriptive (no gate, anti-score-gaming) ---
        "late_auto_coverage": _mean(1 if r["tier"] == "auto" else 0 for r in late),
        "late_suggest_coverage": _mean(1 if r["tier"] == "suggest" else 0 for r in late),
        "late_labeled_acc": _mean(r["correct"] for r in labeled_late),
        "effort_saving_overall": None if es_all is None else round(es_all, 4),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1) if lat else None,
        "mask_miou": _mean(r["iou"] for r in rows_a),
        "mask_fail_rate": _mean(1 if r["iou"] < 0.5 else 0 for r in rows_a),
        "fail_mask": sum(1 for r in rows_a if r["fail"] == "mask"),
        "fail_retrieval": sum(1 for r in rows_a if r["fail"] == "retrieval"),
        "fail_no_mask": sum(1 for r in rows_a if r["fail"] == "no_mask"),
        "per_class_late_acc": {c: _mean(v) for c, v in sorted(per_class.items())},
    }


# ------------------------------------------------------------- verdicts
def _check(op: str, value: float, threshold: float) -> bool:
    return value >= threshold if op == ">=" else value <= threshold


def evaluate_gates(metrics: dict, gates_cfg: dict) -> list[dict]:
    """Each gate -> {name, value, op, threshold, hard, status} where status is
    pass / FAIL / n/a (metric undefined, e.g. zero auto predictions)."""
    results = []
    for kind, table in (("gate", gates_cfg["gates"]), ("warn", gates_cfg["warnings"])):
        for name, g in table.items():
            v = metrics.get(name)
            if v is None:
                status = "n/a"
            elif _check(g["op"], v, g["value"]):
                status = "pass"
            else:
                status = "FAIL" if kind == "gate" else "WARN"
            results.append({"name": name, "value": v, "op": g["op"],
                            "threshold": g["value"],
                            "hard": bool(g.get("hard")) and kind == "gate",
                            "kind": kind, "status": status,
                            "why": g.get("why", "")})
    return results


def hard_failed(gate_results: list[dict]) -> bool:
    return any(r["status"] == "FAIL" and r["hard"] for r in gate_results)


def compare_baseline(metrics: dict, base: dict, rules: dict) -> list[str]:
    """Regression findings (empty list = no regression)."""
    regs: list[str] = []

    def drop_pp(name: str, limit: float):
        cur, prev = metrics.get(name), base.get(name)
        if cur is not None and prev is not None and (prev - cur) * 100 > limit:
            regs.append(f"{name}: {prev:.4f} -> {cur:.4f} "
                        f"(降 {(prev - cur) * 100:.1f}pp > 红线 {limit}pp)")

    # auto_precision: never flag a drop smaller than ONE example's worth — with
    # ~40 late autos one mislabel = ~2.5pp, so a 1pp red line is finer than the
    # measurement and would block any change unlucky enough to hit a single
    # ambiguous/GT-wrong image (e.g. COCO's dog-labeled-cat 000000169076).
    n = min(base.get("n_auto_late") or 99, metrics.get("n_auto_late") or 99)
    ap_limit = max(rules["auto_precision_late_max_drop_pp"], 100.0 / max(1, n))
    drop_pp("auto_precision_late", ap_limit)
    drop_pp("effort_saving_late", rules["effort_saving_late_max_drop_pp"])
    # Median latency only — p95 over ~175 samples swings with GPU state and
    # produced false regressions; the absolute p95 gate still applies.
    frac = rules["latency_p50_max_increase_frac"]
    cur, prev = metrics.get("latency_p50_ms"), base.get("latency_p50_ms")
    if cur is not None and prev and cur > prev * (1 + frac):
        regs.append(f"latency_p50_ms: {prev:.0f} -> {cur:.0f} (升 >{frac:.0%})")
    return regs


# ------------------------------------------------------------- artifacts
def write_predictions_csv(path: Path, rows: list[dict]) -> None:
    fields = ["phase", "step", "file", "gt", "pred", "conf", "margin", "cw",
              "tier", "correct", "iou", "fail", "cand", "cand_correct",
              "taught", "latency_ms", "mem"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_curves(path: Path, rows_a: list[dict], window: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    labeled = [r for r in rows_a if r["tier"] != "silent"]
    flags = [r["correct"] for r in labeled]
    ys = [float(np.mean(flags[max(0, i - window + 1):i + 1])) * 100
          for i in range(len(flags))]
    axes[0].plot(range(len(ys)), ys, color="#1976D2", lw=2)
    axes[0].axhline(20, ls="--", color="gray", label="random (5-class)")
    axes[0].set(title=f"Rolling labeled accuracy (window={window})",
                xlabel="labeled prediction #", ylabel="accuracy %", ylim=(0, 105))
    axes[0].legend(); axes[0].grid(alpha=0.3)

    binsize = max(5, window // 2)
    bins: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i, r in enumerate(rows_a):
        bins[i // binsize][r["tier"]] += 1
    bx = sorted(bins)
    bottoms = [0] * len(bx)
    for t in ("silent", "suggest", "auto"):
        vals = [bins[b][t] for b in bx]
        axes[1].bar([b * binsize for b in bx], vals, bottom=bottoms,
                    width=binsize * 0.85, label=t, color=TIER_COLORS[t])
        bottoms = [bottoms[i] + vals[i] for i in range(len(bx))]
    axes[1].set(title="Tier evolution (cold start -> auto)",
                xlabel="labeling step", ylabel=f"count / {binsize} steps")
    axes[1].legend()

    # Cumulative auto quality: is auto coverage growing WITHOUT precision decay?
    n_auto = n_auto_ok = 0
    cov, prec = [], []
    for i, r in enumerate(rows_a, start=1):
        if r["tier"] == "auto":
            n_auto += 1
            n_auto_ok += r["correct"]
        cov.append(n_auto / i * 100)
        prec.append(n_auto_ok / n_auto * 100 if n_auto else np.nan)
    axes[2].plot(cov, color="#388E3C", lw=2, label="auto coverage %")
    axes[2].plot(prec, color="#D32F2F", lw=2, label="auto precision %")
    axes[2].set(title="Cumulative auto coverage vs precision",
                xlabel="labeling step", ylim=(0, 105))
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def render_markdown(path: Path, metrics: dict, gate_results: list[dict],
                    regressions: list[str] | None, baseline_tag: str | None) -> None:
    m = metrics
    lines = [f"# SnowLabel 监督报告 — {m['tag']}",
             "", f"- 时间:{m['timestamp']}",
             f"- 雪球基准 {m['n_snowball']} 步 / 开集探针 {m['n_openset']} 步", ""]
    lines += ["## Scorecard", "",
              "| 指标 | 值 | 门槛 | 判定 |", "|---|---|---|---|"]
    for r in gate_results:
        v = "n/a" if r["value"] is None else f"{r['value']}"
        hard = "(硬)" if r["hard"] else ""
        lines.append(f"| {r['name']}{hard} | {v} | {r['op']} {r['threshold']} "
                     f"| {r['status']} |")
    lines += ["", "## 描述性指标(无门槛,防刷分)", "",
              f"- late_auto_coverage: {m['late_auto_coverage']}",
              f"- late_suggest_coverage: {m['late_suggest_coverage']}",
              f"- late_labeled_acc: {m['late_labeled_acc']}",
              f"- effort_saving_overall: {m['effort_saving_overall']}",
              f"- latency_p50_ms: {m['latency_p50_ms']}",
              f"- mask_miou: {m['mask_miou']} / mask_fail_rate: {m['mask_fail_rate']}",
              f"- 错误归因:mask 切错 {m['fail_mask']} / 检索认错 "
              f"{m['fail_retrieval']} / 无 mask {m['fail_no_mask']}",
              f"- 后半程各类准确率:{json.dumps(m['per_class_late_acc'], ensure_ascii=False)}"]
    if regressions is not None:
        lines += ["", f"## 对比 baseline({baseline_tag})", ""]
        lines += ([f"- ❌ {r}" for r in regressions] if regressions
                  else ["- ✅ 无回归"])
    lines += ["", "## 配置快照", "", "```json",
              json.dumps(m["config"], indent=2, ensure_ascii=False), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_history(history_csv: Path, metrics: dict, verdict: str) -> None:
    new = not history_csv.exists()
    with open(history_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({**{k: metrics.get(k) for k in HISTORY_FIELDS if k != "verdict"},
                    "verdict": verdict})


def scorecard_text(metrics: dict, gate_results: list[dict],
                   regressions: list[str] | None, verdict: str) -> str:
    out = [f"\n=== SCORECARD [{metrics['tag']}] ==="]
    for r in gate_results:
        v = "n/a" if r["value"] is None else r["value"]
        mark = {"pass": "PASS", "FAIL": "FAIL", "WARN": "WARN", "n/a": " n/a"}[r["status"]]
        hard = " (hard)" if r["hard"] else ""
        out.append(f"  [{mark}] {r['name']:<22} {v}  ({r['op']} {r['threshold']}){hard}")
    out.append(f"  ----  late_auto_coverage={metrics['late_auto_coverage']}  "
               f"mask_miou={metrics['mask_miou']}  "
               f"p50={metrics['latency_p50_ms']}ms")
    if regressions is not None:
        out.append("  baseline 对比: " + ("REGRESSION: " + "; ".join(regressions)
                                          if regressions else "无回归"))
    out.append(f"=== VERDICT: {verdict} ===")
    return "\n".join(out)


def promote(run_dir: Path, baseline_dir: Path) -> None:
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    shutil.copytree(run_dir, baseline_dir)
