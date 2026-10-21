"""Rebuild publishable VisA evidence from a completed model run."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_visa import contact_sheet, select_cases  # noqa: E402


CASE_PATTERN = re.compile(r"^(snowball|openset)_(\d{4})_(.+)\.webp$")


def typed_row(row: dict[str, str]) -> dict:
    result = dict(row)
    for key in ("step", "correct", "cand_correct", "mem", "area"):
        result[key] = int(result[key])
    for key in (
        "conf",
        "margin",
        "cw",
        "iou",
        "latency_ms",
        "commit_latency_ms",
    ):
        result[key] = float(result[key]) if result[key] else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "v0.1")
    args = parser.parse_args()

    with (args.run_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = [typed_row(row) for row in csv.DictReader(handle)]
    by_key = {(row["phase"], row["step"]): row for row in rows}
    seen = [row for row in rows if row["phase"] == "snowball"]
    openset = [row for row in rows if row["phase"] == "openset"]
    late = seen[len(seen) // 2 :]
    valid_open = [row for row in openset if row["iou"] >= 0.5]

    metrics_path = args.run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "late_mask_success_rate": round(
                sum(row["iou"] >= 0.5 for row in late) / max(1, len(late)), 4
            ),
            "late_valid_auto_coverage": round(
                sum(row["tier"] == "auto" and row["iou"] >= 0.5 for row in late)
                / max(1, len(late)),
                4,
            ),
            "n_openset_valid_mask": len(valid_open),
            "openset_false_auto_valid_mask": round(
                sum(row["tier"] == "auto" for row in valid_open)
                / max(1, len(valid_open)),
                4,
            ),
            "openset_false_any_valid_mask": round(
                sum(row["tier"] in ("auto", "suggest") for row in valid_open)
                / max(1, len(valid_open)),
                4,
            ),
        }
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cases = []
    for visualization in sorted((args.run_dir / "failure_cases").glob("*.webp")):
        match = CASE_PATTERN.match(visualization.name)
        if not match:
            continue
        phase, step, kind = match.group(1), int(match.group(2)), match.group(3)
        row = by_key[(phase, step)]
        cases.append({"kind": kind, "row": row, "visualization": str(visualization)})
    chosen = select_cases(cases)
    contact_sheet(args.run_dir / "failure_cases.webp", chosen)
    public_cases = []
    for case in chosen:
        item = {"kind": case["kind"], **case["row"]}
        item["visualization"] = Path(case["visualization"]).name
        public_cases.append(item)
    (args.run_dir / "failure_cases.json").write_text(
        json.dumps(public_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    copies = {
        "metrics.json": "visa_metrics.json",
        "manifest.json": "visa_manifest.json",
        "predictions.csv": "visa_predictions.csv",
        "failure_cases.json": "visa_failure_cases.json",
        "failure_cases.webp": "visa_failure_cases.webp",
    }
    for source, target in copies.items():
        shutil.copy2(args.run_dir / source, args.output / target)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_failure_cases": len(chosen),
                "late_valid_auto_coverage": metrics["late_valid_auto_coverage"],
                "openset_false_auto_valid_mask": metrics[
                    "openset_false_auto_valid_mask"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
