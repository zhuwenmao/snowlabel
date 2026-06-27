"""SnowLabel supervision entry point — run after ANY pipeline change.

    python scripts/supervise.py --tag "512x16parts"          # full benchmark
    python scripts/supervise.py --quick --tag smoke          # fast iteration
    python scripts/supervise.py --tag X --compare            # diff vs baseline
    python scripts/supervise.py --promote results/supervise/<run_dir>

Exit codes: 0 = pass, 1 = hard gate failed, 2 = regression vs baseline.
Thresholds, effort costs and protocol parameters live in eval/gates.json.
NOTE: needs the production env (SNOWLABEL_DINO/RES/...); stop the 9090 server
first — two model stacks don't fit the GPU together.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import protocol, report  # noqa: E402


def config_snapshot() -> dict:
    from snowlabel import predictor as P
    from snowlabel.config import get_config
    cfg = get_config()
    return {
        "dino_variant": str(cfg.dino_variant),
        "sam_variant": cfg.sam_variant,
        "input_resolution": cfg.input_resolution,
        "parts_grid": cfg.parts_grid,
        "rerank_n": cfg.rerank_n,
        "rerank_weights": [P.RERANK_W_WHOLE, P.RERANK_W_PARTS, P.RERANK_W_GEOM],
        "open_set_min_sim": cfg.open_set_min_sim,
        "conf_auto_apply": cfg.conf_auto_apply,
        "conf_suggest": cfg.conf_suggest,
        "margin_auto": cfg.margin_auto,
        "min_support_auto": cfg.min_support_auto,
        "min_support_suggest": cfg.min_support_suggest,
        "memory": "numpy isolated (data/memory_supervise)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--quick", action="store_true", help="5x10 + 20 openset")
    ap.add_argument("--compare", action="store_true",
                    help="diff vs baseline; regression -> exit code 2")
    ap.add_argument("--promote", metavar="RUN_DIR",
                    help="set RUN_DIR as the new baseline and exit")
    ap.add_argument("--sweep", metavar="RUN_DIR",
                    help="offline threshold sweep over a saved run and exit")
    ap.add_argument("--n-per", type=int, help="override instances per class")
    ap.add_argument("--openset-n", type=int, help="override open-set probes")
    ap.add_argument("--seed", type=int,
                    help="override open-set sampling seed (robustness checks)")
    ap.add_argument("--classes", metavar="CSV",
                    help="custom class suite (e.g. 'car,truck,bus,cup,bowl'); "
                         "uses full val2017 annotations, images-on-disk only. "
                         "Do not --compare custom suites against the standard "
                         "baseline.")
    args = ap.parse_args()

    out_root = ROOT / "results" / "supervise"
    baseline_dir = out_root / "baseline"

    if args.sweep:
        from eval import sweep as sweep_mod
        gates_cfg = json.loads((ROOT / "eval" / "gates.json").read_text(encoding="utf-8"))
        res = sweep_mod.sweep(Path(args.sweep), gates_cfg)
        print(f"swept {res['n_combos']} combos, {res['n_feasible']} feasible")
        for title, rows in (("top (feasible)", res["top10"]),
                            ("top (STRICT: 校准跑零误放)", res["top10_strict"])):
            print(f"--- {title} ---")
            print(f"{'auto':>6} {'sugg':>6} {'marg':>6} | {'effort':>7} "
                  f"{'autoCov':>8} {'autoPrec':>9} {'osFalseAuto':>11}")
            for m in rows:
                print(f"{m['conf_auto']:>6} {m['conf_suggest']:>6} "
                      f"{m['margin_auto']:>6} | {m['effort_saving_late']:>7} "
                      f"{m['late_auto_coverage']:>8} "
                      f"{str(m['auto_precision_late']):>9} "
                      f"{m['openset_false_auto']:>11}")
        out = Path(args.sweep) / "sweep.json"
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"saved: {out}")
        return 0

    if args.promote:
        run_dir = Path(args.promote)
        if not (run_dir / "metrics.json").is_file():
            print(f"no metrics.json in {run_dir}")
            return 1
        report.promote(run_dir, baseline_dir)
        print(f"baseline <- {run_dir}")
        return 0

    gates_cfg = json.loads((ROOT / "eval" / "gates.json").read_text(encoding="utf-8"))
    bench = gates_cfg["benchmark"]
    n_per = args.n_per or (bench["quick"]["n_per"] if args.quick else bench["n_per"])
    openset_n = args.openset_n or (bench["quick"]["openset_n"] if args.quick
                                   else bench["openset_n"])
    if args.seed is not None:
        bench["seed"] = args.seed
    custom_suite = bool(args.classes)
    if custom_suite:
        bench["classes"] = [c.strip() for c in args.classes.split(",") if c.strip()]

    snapshot = config_snapshot()
    if custom_suite:
        snapshot["suite_classes"] = bench["classes"]
    print("config:", json.dumps(snapshot, ensure_ascii=False))
    print(f"protocol: {len(bench['classes'])} classes x {n_per} + "
          f"{openset_n} open-set probes (seed={bench['seed']})")

    # --- predictor with an ISOLATED numpy memory (never the production store)
    from snowlabel.memory import SnowballMemory
    from snowlabel.predictor import SnowPredictor
    pred = SnowPredictor()
    mem_dir = ROOT / "data" / "memory_supervise"
    if mem_dir.exists():
        shutil.rmtree(mem_dir)
    pred._memory = SnowballMemory(mem_dir, dim=pred.dino.embed_dim)

    img_dir = ROOT / "data" / "coco" / "val2017"
    ann_dir = ROOT / "data" / "coco" / "annotations"
    coco_full = protocol.load_coco(ann_dir / "instances_val2017.json")
    # Standard suite reads the filtered json (keeps sequences identical to
    # the promoted baseline); custom suites draw from the full annotations
    # restricted to images on disk.
    if custom_suite:
        coco = coco_full
        seen_img_dir = img_dir
    else:
        coco = protocol.load_coco(ann_dir / "instances_val2017_snowlabel.json")
        seen_img_dir = None

    t0 = time.time()
    seq_a = protocol.build_seen_sequence(coco, bench["classes"], n_per,
                                         bench["min_area"], img_dir=seen_img_dir)
    print(f"phase A: {len(seq_a)} steps")
    rows_a = protocol.run_snowball(pred, coco, seq_a, img_dir)

    seq_b = protocol.build_openset_sequence(
        coco_full, bench["classes"], img_dir, openset_n, bench["min_area"],
        bench["seed"], bench["openset_per_class_cap"],
        exclude_iids={it["iid"] for it in seq_a})
    print(f"phase B: {len(seq_b)} probes "
          f"({len(set(i['cls'] for i in seq_b))} unseen classes)")
    rows_b = protocol.run_openset(pred, coco_full, seq_b, img_dir)
    print(f"benchmark done in {time.time() - t0:.0f}s")

    # --- phase C: metrics, gates, baseline diff, artifacts
    metrics = report.compute_metrics(rows_a, rows_b, gates_cfg, snapshot, args.tag)
    gate_results = report.evaluate_gates(metrics, gates_cfg)

    regressions = None
    baseline_tag = None
    if args.compare:
        base_file = baseline_dir / "metrics.json"
        if base_file.is_file():
            base = json.loads(base_file.read_text(encoding="utf-8"))
            baseline_tag = base.get("tag")
            regressions = report.compare_baseline(metrics, base,
                                                  gates_cfg["regression"])
        else:
            print("(--compare 指定了但还没有 baseline,跳过对比)")

    failed_hard = report.hard_failed(gate_results)
    regressed = bool(regressions)
    verdict = ("REGRESSION" if regressed else
               "FAIL" if failed_hard else "PASS")

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{ts}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report.write_predictions_csv(run_dir / "predictions.csv", rows_a + rows_b)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report.plot_curves(run_dir / "curves.png", rows_a, bench["rolling_window"])
    report.render_markdown(run_dir / "REPORT.md", metrics, gate_results,
                           regressions, baseline_tag)
    report.append_history(out_root / "history.csv", metrics, verdict)

    print(report.scorecard_text(metrics, gate_results, regressions, verdict))
    print(f"artifacts: {run_dir}")
    return 2 if regressed else (1 if failed_hard else 0)


if __name__ == "__main__":
    raise SystemExit(main())
