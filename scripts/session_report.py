"""Summarize REAL labeling sessions from data/telemetry/events.jsonl.

The benchmark (scripts/supervise.py) is a simulation; this is the final judge:
what actually happened when a human labeled. Events come from the web tool's
commit flow — "accept" (prediction untouched, not stored) and "commit"
(corrected or hand-added, stored), each carrying pred_label/tier/source.

    python scripts/session_report.py            # all days
    python scripts/session_report.py --day 2026-06-12
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "data" / "telemetry" / "events.jsonl"


def load_events() -> list[dict]:
    if not EVENTS.is_file():
        return []
    out = []
    with open(EVENTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def summarize(events: list[dict]) -> None:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_day[(e.get("ts") or "?")[:10]].append(e)

    print(f"{'日期':<12}{'确认对':>8}{'纠正':>6}{'手动加':>8}{'auto接受':>10}"
          f"{'suggest接受':>12}{'纠正率':>8}")
    for day in sorted(by_day):
        ev = by_day[day]
        accepts = [e for e in ev if e["type"] == "accept"]
        commits = [e for e in ev if e["type"] == "commit"]
        corrected = [e for e in commits if e.get("corrected")]
        added = [e for e in commits if not e.get("pred_label")]
        n_pred = len(accepts) + len(corrected)   # predictions shown & judged
        corr_rate = f"{len(corrected) / n_pred:.0%}" if n_pred else "-"
        print(f"{day:<12}{len(accepts):>8}{len(corrected):>6}{len(added):>8}"
              f"{sum(1 for e in accepts if e.get('tier') == 'auto'):>10}"
              f"{sum(1 for e in accepts if e.get('tier') == 'suggest'):>12}"
              f"{corr_rate:>8}")

    corrected_all = [e for e in events if e["type"] == "commit" and e.get("corrected")]
    if corrected_all:
        print("\n最常被纠正(模型预测 -> 用户改成):")
        pairs = Counter((e.get("pred_label"), e.get("label")) for e in corrected_all)
        for (pl, l), n in pairs.most_common(10):
            print(f"  {pl} -> {l}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="only this day (YYYY-MM-DD)")
    args = ap.parse_args()
    events = load_events()
    if args.day:
        events = [e for e in events if (e.get("ts") or "").startswith(args.day)]
    if not events:
        print(f"无事件({EVENTS}); 先用网页工具标几张图。")
        return
    summarize(events)


if __name__ == "__main__":
    main()
