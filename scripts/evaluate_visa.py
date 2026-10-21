"""Run the fixed SnowLabel progressive-annotation protocol on official VisA."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import textwrap
import time
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import report  # noqa: E402
from eval.effort import effort_saving  # noqa: E402
from eval.visa_protocol import (  # noqa: E402
    DATASET_BYTES,
    build_protocol,
    load_instances,
    load_target,
    predict_step,
)


PREDICTION_FIELDS = [
    "phase",
    "step",
    "object",
    "file",
    "gt",
    "pred",
    "conf",
    "margin",
    "cw",
    "tier",
    "correct",
    "iou",
    "fail",
    "cand",
    "cand_correct",
    "taught",
    "latency_ms",
    "commit_latency_ms",
    "mem",
    "area",
]


def percentile(rows: list[dict], field: str, q: int) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(float(np.percentile(values, q)), 1) if values else None


def progression(rows: list[dict], costs: dict) -> list[dict]:
    result = []
    edges = np.linspace(0, len(rows), 5, dtype=int)
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        bucket = rows[start:end]
        valid_auto = [
            row for row in bucket if row["tier"] == "auto" and row["iou"] >= 0.5
        ]
        saving = effort_saving(bucket, costs)
        result.append(
            {
                "quarter": index,
                "steps": len(bucket),
                "auto_coverage": round(
                    sum(row["tier"] == "auto" for row in bucket) / max(1, len(bucket)),
                    4,
                ),
                "auto_precision": (
                    round(sum(row["correct"] for row in valid_auto) / len(valid_auto), 4)
                    if valid_auto
                    else None
                ),
                "effort_saving": None if saving is None else round(saving, 4),
                "mask_success": round(
                    sum(row["iou"] >= 0.5 for row in bucket) / max(1, len(bucket)), 4
                ),
            }
        )
    return result


def render_case(
    output: Path,
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None,
    row: dict,
) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool) if pred_mask is not None else np.zeros_like(gt)
    if pred.shape != gt.shape:
        pred = np.zeros_like(gt)
    overlay = rgb.astype(np.float32)
    only_gt = gt & ~pred
    only_pred = pred & ~gt
    overlap = gt & pred
    overlay[only_gt] = overlay[only_gt] * 0.45 + np.array([40, 210, 90]) * 0.55
    overlay[only_pred] = overlay[only_pred] * 0.45 + np.array([230, 60, 70]) * 0.55
    overlay[overlap] = overlay[overlap] * 0.35 + np.array([245, 200, 50]) * 0.65
    visual = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    visual.thumbnail((480, 300), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (500, 390), "#111318")
    x = (500 - visual.width) // 2
    canvas.paste(visual, (x, 84))
    draw = ImageDraw.Draw(canvas)
    title = f"{row['fail'] or 'failure'} | {row['object']} | IoU {row['iou']:.3f}"
    detail = f"GT: {row['gt']}\nPred: {row['pred']} ({row['tier']}, {row['conf']:.3f})"
    draw.text((12, 10), title, fill="#ffffff")
    draw.multiline_text(
        (12, 30), "\n".join(textwrap.wrap(detail, width=76)), fill="#c8ccd6", spacing=3
    )
    draw.text((12, 370), "green=GT  red=prediction  yellow=overlap", fill="#9ea4b2")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "WEBP", quality=88, method=6)


def select_cases(cases: list[dict], limit: int = 10) -> list[dict]:
    priority = {
        "wrong_auto": 0,
        "openset_auto": 1,
        "no_mask": 2,
        "mask": 3,
        "wrong_suggest": 4,
        "openset_suggest": 5,
    }
    cases = sorted(
        cases,
        key=lambda case: (
            priority.get(case["kind"], 9),
            case["row"]["iou"],
            case["row"]["object"],
            case["row"]["file"],
        ),
    )
    chosen: list[dict] = []
    seen_kinds: set[str] = set()
    for case in cases:
        if case["kind"] not in seen_kinds:
            chosen.append(case)
            seen_kinds.add(case["kind"])
        if len(chosen) >= limit:
            return chosen
    seen_groups: set[tuple[str, str]] = set()
    for case in cases:
        key = (case["kind"], case["row"]["object"])
        if case not in chosen and key not in seen_groups:
            chosen.append(case)
            seen_groups.add(key)
        if len(chosen) >= limit:
            return chosen
    for case in cases:
        if case not in chosen:
            chosen.append(case)
        if len(chosen) >= limit:
            break
    return chosen


def contact_sheet(output: Path, chosen: list[dict]) -> None:
    if not chosen:
        return
    width, height = 500, 390
    sheet = Image.new("RGB", (width * 2, height * ((len(chosen) + 1) // 2)), "#08090c")
    for index, case in enumerate(chosen):
        tile = Image.open(case["visualization"]).convert("RGB")
        sheet.paste(tile, ((index % 2) * width, (index // 2) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "WEBP", quality=90, method=6)


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def configure_runtime(run_dir: Path) -> None:
    for key in (
        "SNOWLABEL_CONF_AUTO",
        "SNOWLABEL_CONF_SUGGEST",
        "SNOWLABEL_MARGIN_AUTO",
        "SNOWLABEL_OPEN_MIN_SIM",
        "SNOWLABEL_MIN_SUPPORT_AUTO",
        "SNOWLABEL_MIN_SUPPORT_SUGGEST",
        "SNOWLABEL_K",
        "SNOWLABEL_RERANK_N",
    ):
        os.environ.pop(key, None)
    os.environ.update(
        {
            "SNOWLABEL_ROOT": str(run_dir / "runtime"),
            "SNOWLABEL_MEMORY_BACKEND": "numpy",
            "SNOWLABEL_IMGCACHE": "0",
            "SNOWLABEL_MULTIUSER": "0",
            "SNOWLABEL_DINO": str(
                ROOT / "weights" / "facebookdinov3-vitl16-pretrain-lvd1689m"
            ),
            "SNOWLABEL_SAM": str(ROOT / "sam2.1_t.pt"),
            "SNOWLABEL_RES": "512",
            "SNOWLABEL_PARTS_GRID": "4",
        }
    )


def failure_kind(row: dict, phase: str) -> str | None:
    if phase == "openset" and row["tier"] in ("auto", "suggest"):
        return f"openset_{row['tier']}"
    if row["fail"] == "no_mask":
        return "no_mask"
    if row["iou"] < 0.5:
        return "mask"
    if row["tier"] in ("auto", "suggest") and not row["correct"]:
        return f"wrong_{row['tier']}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "datasets" / "visa")
    parser.add_argument("--tag", default="visa-v0.1")
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--n-per", type=int, default=6)
    parser.add_argument("--openset-per-object", type=int, default=4)
    parser.add_argument("--limit-seen", type=int)
    parser.add_argument("--limit-openset", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--publish-evidence", action="store_true")
    args = parser.parse_args()

    archive = args.data_root / "VisA_20220922.tar"
    if not archive.is_file() or archive.stat().st_size != DATASET_BYTES:
        raise FileNotFoundError(
            f"official VisA archive missing or incomplete: {archive}; "
            "run scripts/download_visa.py"
        )
    instances = load_instances(args.data_root, args.min_area)
    seen, openset, manifest = build_protocol(
        instances, args.n_per, args.openset_per_object
    )
    manifest["dataset"]["accessed"] = date.today().isoformat()
    manifest["protocol"]["min_component_area"] = args.min_area
    print(
        f"VisA: {len(instances)} instances, {manifest['inventory']['classes']} classes; "
        f"seen={len(seen)} ({len(manifest['seen']['classes'])} classes), "
        f"openset={len(openset)}"
    )
    if args.prepare_only:
        print(json.dumps(manifest["openset"]["holdout_by_object"], indent=2))
        return 0

    if args.limit_seen is not None:
        seen = seen[: args.limit_seen]
    if args.limit_openset is not None:
        openset = openset[: args.limit_openset]
    is_full = len(seen) == manifest["seen"]["samples"] and len(openset) == manifest["openset"]["samples"]
    if args.publish_evidence and not is_full:
        raise ValueError("refusing to publish evidence from a limited run")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "visa" / f"{stamp}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    configure_runtime(run_dir)

    from snowlabel.config import get_config
    from snowlabel.features import feature_space_id
    from snowlabel.predictor import SnowPredictor

    config = get_config()
    predictor = SnowPredictor(config)
    if len(predictor.memory):
        raise RuntimeError("VisA benchmark must start with empty memory")

    if seen:
        warm_image, warm_mask = load_target(args.data_root, seen[0])
        from eval.protocol import click_point

        x, y = click_point(warm_mask, {"bbox": seen[0].bbox})
        predictor.predict_from_point(warm_image, x, y)

    rows_seen: list[dict] = []
    rows_open: list[dict] = []
    cases: list[dict] = []
    cases_dir = run_dir / "failure_cases"

    for step, item in enumerate(seen):
        prediction, image, gt_mask, row = predict_step(
            predictor, args.data_root, item, "snowball", step
        )
        commit_started = time.perf_counter()
        if prediction is not None and row["iou"] >= 0.5:
            taught_mask = prediction.mask
            row["taught"] = "sam"
        else:
            taught_mask = gt_mask
            row["taught"] = "gt_fix"
        predictor.remember(
            image, taught_mask, item.label, image_uri=item.image, verified=True
        )
        row["commit_latency_ms"] = round(
            (time.perf_counter() - commit_started) * 1000, 1
        )
        row["mem"] = len(predictor.memory)
        rows_seen.append(row)
        kind = failure_kind(row, "snowball")
        if kind:
            visual = cases_dir / f"snowball_{step:04d}_{kind}.webp"
            render_case(
                visual,
                image,
                gt_mask,
                prediction.mask if prediction is not None else None,
                row,
            )
            cases.append({"kind": kind, "row": row.copy(), "visualization": str(visual)})
        if step % 20 == 0:
            print(
                f"[seen] {step:3d}/{len(seen)} {row['gt']} -> {row['pred']} "
                f"({row['tier']}, iou={row['iou']}) mem={row['mem']}"
            )

    for step, item in enumerate(openset):
        prediction, image, gt_mask, row = predict_step(
            predictor, args.data_root, item, "openset", step
        )
        row["correct"] = 0
        row["commit_latency_ms"] = None
        row["fail"] = "" if row["tier"] == "silent" else "openset_accept"
        rows_open.append(row)
        kind = failure_kind(row, "openset")
        if kind:
            visual = cases_dir / f"openset_{step:04d}_{kind}.webp"
            render_case(
                visual,
                image,
                gt_mask,
                prediction.mask if prediction is not None else None,
                row,
            )
            cases.append({"kind": kind, "row": row.copy(), "visualization": str(visual)})
        if step % 12 == 0:
            print(
                f"[open] {step:3d}/{len(openset)} {row['gt']} -> {row['pred']} "
                f"({row['tier']})"
            )

    gates = json.loads((ROOT / "eval" / "gates.json").read_text(encoding="utf-8"))
    snapshot = {
        "benchmark": "VisA progressive annotation v0.1",
        "dino": Path(config.dino_variant).name,
        "sam": Path(config.sam_variant).name,
        "resolution": config.input_resolution,
        "parts_grid": config.parts_grid,
        "conf_auto": config.conf_auto_apply,
        "conf_suggest": config.conf_suggest,
        "margin_auto": config.margin_auto,
        "open_set_min_sim": config.open_set_min_sim,
        "min_support_auto": config.min_support_auto,
        "min_support_suggest": config.min_support_suggest,
        "feature_space_id": feature_space_id(predictor.memory.dim),
    }
    metrics = report.compute_metrics(rows_seen, rows_open, gates, snapshot, args.tag)
    late = rows_seen[len(rows_seen) // 2 :]
    valid_open = [row for row in rows_open if row["iou"] >= 0.5]
    metrics.update(
        {
            "dataset": manifest["dataset"],
            "protocol": manifest["protocol"],
            "n_seen_classes": len(manifest["seen"]["classes"]),
            "n_holdout_classes": len(manifest["openset"]["holdout_by_object"]),
            "progression": progression(rows_seen, gates["effort_costs"]),
            "commit_latency_p50_ms": percentile(rows_seen, "commit_latency_ms", 50),
            "commit_latency_p95_ms": percentile(rows_seen, "commit_latency_ms", 95),
            "mask_success_rate": round(
                sum(row["iou"] >= 0.5 for row in rows_seen) / max(1, len(rows_seen)), 4
            ),
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
            "auto_errors": sum(
                row["tier"] == "auto" and not row["correct"] and row["iou"] >= 0.5
                for row in rows_seen
            ),
            "full_protocol": is_full,
        }
    )

    all_rows = rows_seen + rows_open
    write_predictions(run_dir / "predictions.csv", all_rows)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    chosen = select_cases(cases)
    contact_sheet(run_dir / "failure_cases.webp", chosen)
    public_cases = []
    for case in chosen:
        item = {"kind": case["kind"], **case["row"]}
        item["visualization"] = Path(case["visualization"]).name
        public_cases.append(item)
    (run_dir / "failure_cases.json").write_text(
        json.dumps(public_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report.plot_curves(run_dir / "curves.png", rows_seen, window=max(10, args.n_per * 2))

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"run_dir={run_dir}")

    if args.publish_evidence:
        evidence = ROOT / "evidence" / "v0.1"
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "metrics.json", evidence / "visa_metrics.json")
        shutil.copy2(run_dir / "manifest.json", evidence / "visa_manifest.json")
        shutil.copy2(run_dir / "predictions.csv", evidence / "visa_predictions.csv")
        shutil.copy2(run_dir / "failure_cases.json", evidence / "visa_failure_cases.json")
        if (run_dir / "failure_cases.webp").is_file():
            shutil.copy2(run_dir / "failure_cases.webp", evidence / "visa_failure_cases.webp")
        print(f"published evidence to {evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
