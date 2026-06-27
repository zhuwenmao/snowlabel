"""Simulated-annotator effort model.

The baseline is "SAM-assisted manual": the user still clicks the object and
SAM still cuts the mask, but with NO memory they must pick the class by hand
every time (manual = click + pick = 2 actions). Savings therefore attribute
ONLY to the prediction memory, not to SAM — the honest comparison.

Asymmetry is deliberate: a wrong auto label costs MORE than manual (the user
has to notice it among green "trusted" labels and then fix it), so a system
cannot buy effort savings by auto-labeling recklessly.
"""
from __future__ import annotations


def row_cost(row: dict, costs: dict) -> float:
    # A bad SAM cut (IoU < 0.5) means the user repairs the mask (re-click /
    # negative clicks) and picks the class — the tier's label is moot.
    if row.get("iou", 1.0) < 0.5:
        return costs["mask_fix"]
    tier = row["tier"]
    if tier == "auto":
        return costs["auto_ok"] if row["correct"] else costs["auto_wrong"]
    if tier == "suggest":
        return costs["suggest_ok"] if row["correct"] else costs["suggest_wrong"]
    return costs["silent"]


def effort_saving(rows: list[dict], costs: dict) -> float | None:
    """1 - (simulated cost / all-manual cost); None when rows is empty."""
    if not rows:
        return None
    manual = costs["manual"] * len(rows)
    actual = sum(row_cost(r, costs) for r in rows)
    return 1.0 - actual / manual
