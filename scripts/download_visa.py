"""Download the official VisA archive with resumable parallel HTTP ranges."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def download_range(url: str, path: Path, start: int, end: int) -> int:
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return expected
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(1, 6):
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 120),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status_code}")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"range {start}-{end} has {temporary.stat().st_size}, expected {expected}"
                )
            os.replace(temporary, path)
            return expected
        except (OSError, requests.RequestException, RuntimeError):
            if attempt == 5:
                raise
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "datasets" / "visa" / "VisA_20220922.tar",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()

    response = requests.head(args.url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers["Content-Length"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.output.stat().st_size if args.output.exists() else 0
    if prefix > total:
        raise RuntimeError(f"existing file is larger than source: {prefix} > {total}")
    if prefix == total:
        print(f"already complete: {args.output} ({total} bytes)")
        return 0

    chunk_size = args.chunk_mib * 1024 * 1024
    ranges = []
    start = prefix
    while start < total:
        end = min(total - 1, start + chunk_size - 1)
        ranges.append((start, end))
        start = end + 1
    parts_dir = args.output.with_name(args.output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"source={total} bytes existing_prefix={prefix} "
        f"remaining_ranges={len(ranges)} workers={args.workers}"
    )

    completed = prefix
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_range,
                args.url,
                parts_dir / f"{start:012d}-{end:012d}.part",
                start,
                end,
            ): (start, end)
            for start, end in ranges
        }
        for future in as_completed(futures):
            completed += future.result()
            print(f"downloaded {completed / total:.1%}")

    assembled = args.output.with_suffix(args.output.suffix + ".assembling")
    with assembled.open("wb") as target:
        if prefix:
            with args.output.open("rb") as source:
                remaining = prefix
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError("existing prefix became shorter during assembly")
                    target.write(block)
                    remaining -= len(block)
        for start, end in ranges:
            part = parts_dir / f"{start:012d}-{end:012d}.part"
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)
    if assembled.stat().st_size != total:
        raise RuntimeError(f"assembled archive has wrong size: {assembled.stat().st_size}")
    os.replace(assembled, args.output)
    for part in parts_dir.iterdir():
        part.unlink()
    parts_dir.rmdir()
    print(f"complete: {args.output} ({total} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
