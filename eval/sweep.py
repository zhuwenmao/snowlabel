"""Post-hoc threshold sweep over a saved supervision run.

Key insight: in the benchmark, teaching is GT-driven regardless of tier, so
the prediction stream (conf/margin/correct per step) is threshold-independent
— tiers can be re-simulated offline from predictions.csv at zero GPU cost.

For every (conf_auto, conf_suggest, margin_auto) combo we replay the tier
gate over phase-A rows (effort/precision) and phase-B rows (open-set false
accepts), then pick the combo that maximizes LATE effort saving subject to
the same hard gates the scorecard enforces. Support gating is assumed
satisfied (late half has >=12 examples per class; early steps barely move
the objective).
"""
from __future__ import annotations

import csv
from pathlib import Path

from .effort import effort_saving


def load_rows(run_dir: Path) -> tuple[list[dict], list[dict]]:
    rows_a, rows_b = [], []
    with open(run_dir / "predictions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "cand" not in r or not r["cand"]:
                raise SystemExit(
                    "predictions.csv lacks the 'cand' column — re-run the "
                    "benchmark first (older runs can't be swept).")
            row = {**r, "conf": float(r["conf"]), "margin": float(r["margin"]),
                   "iou": float(r["iou"]), "correct": int(r["cand_correct"])}
            (rows_a if r["phase"] == "snowball" else rows_b).append(row)
    return rows_a, rows_b


def _resim(row: dict, auto: float, suggest: float, margin: float) -> str:
    if row["cand"] in ("", "-"):          # silent for non-threshold reasons
        return "silent"                    # (no mask / open-set cw guard)
    if row["conf"] >= auto and row["margin"] >= margin:
        return "auto"
    if row["conf"] >= suggest:
        return "suggest"
    return "silent"


def evaluate_combo(rows_a: list[dict], rows_b: list[dict], gates_cfg: dict,
                   auto: float, suggest: float, margin: float) -> dict:
    costs = gates_cfg["effort_costs"]
    a = [{**r, "tier": _resim(r, auto, suggest, margin)} for r in rows_a]
    b = [{**r, "tier": _resim(r, auto, suggest, margin)} for r in rows_b]
    late = a[len(a) // 2:]
    auto_late = [r for r in late if r["tier"] == "auto" and r["iou"] >= 0.5]
    prec = (sum(r["correct"] for r in auto_late) / len(auto_late)
            if auto_late else None)
    return {
        "conf_auto": auto, "conf_suggest": suggest, "margin_auto": margin,
        "auto_precision_late": None if prec is None else round(prec, 4),
        "late_auto_coverage": round(sum(1 for r in late if r["tier"] == "auto")
                                    / max(1, len(late)), 4),
        "effort_saving_late": round(effort_saving(late, costs) or 0.0, 4),
        "poison_rate": round(sum(1 for r in a if r["tier"] == "auto"
                                 and not r["correct"] and r["iou"] >= 0.5)
                             / max(1, len(a)), 4),
        "openset_false_auto": round(sum(1 for r in b if r["tier"] == "auto")
                                    / max(1, len(b)), 4),
        "openset_false_any": round(sum(1 for r in b
                                       if r["tier"] in ("auto", "suggest"))
                                   / max(1, len(b)), 4),
    }


def feasible(m: dict, gates_cfg: dict) -> bool:
    g = gates_cfg["gates"]
    if m["auto_precision_late"] is not None and \
            m["auto_precision_late"] < g["auto_precision_late"]["value"]:
        return False
    return (m["poison_rate"] <= g["poison_rate"]["value"]
            and m["openset_false_auto"] <= g["openset_false_auto"]["value"]
            and m["openset_false_any"] <= g["openset_false_any"]["value"])


def sweep(run_dir: Path, gates_cfg: dict) -> dict:
    rows_a, rows_b = load_rows(run_dir)
    results = []
    autos = [round(0.55 + 0.01 * i, 2) for i in range(36)]      # 0.55..0.90
    suggests = [round(0.45 + 0.01 * i, 2) for i in range(31)]   # 0.45..0.75
    margins = [0.02, 0.05, 0.08, 0.12]
    for mg in margins:
        for at in autos:
            for sg in suggests:
                if sg > at:
                    continue
                results.append(evaluate_combo(rows_a, rows_b, gates_cfg,
                                              at, sg, mg))
    ok = [m for m in results if feasible(m, gates_cfg)]
    pool = ok or results
    key = lambda m: (-m["effort_saving_late"], -m["conf_auto"],  # noqa: E731
                     -m["conf_suggest"], -m["margin_auto"])
    # Maximize effort; tie-break to the SAFEST combo (highest thresholds) so
    # we don't sit on a cliff edge of the feasible region.
    pool.sort(key=key)
    # Strict pool: on the CALIBRATION run itself we demand a clean sheet
    # (zero open-set auto, perfect auto precision). Thresholds chosen at the
    # exact gate boundary overfit this seed — one noisy probe on the next run
    # flips them into a hard FAIL.
    strict = sorted((m for m in ok
                     if m["openset_false_auto"] == 0.0
                     and m["openset_false_any"] <= 0.10
                     and (m["auto_precision_late"] in (None, 1.0))), key=key)
    return {"n_combos": len(results), "n_feasible": len(ok),
            "best": pool[0], "best_strict": strict[0] if strict else None,
            "top10": pool[:10], "top10_strict": strict[:10]}
