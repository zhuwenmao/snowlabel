"""Measure uncached and cached Everything latency through the real web API."""

from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, PngImagePlugin


ROOT = Path(__file__).resolve().parent.parent


def post_timed(url: str, **kwargs) -> tuple[requests.Response, float]:
    started = time.perf_counter()
    response = requests.post(url, **kwargs)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9090")
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "data" / "datasets" / "visa" / "pcb4" / "Data" / "Images" / "Anomaly" / "099.JPG",
    )
    parser.add_argument("--max-masks", type=int, default=80)
    parser.add_argument("--user-id", default="benchmark_everything_v01")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "v0.1" / "everything_latency.json",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    nonce = datetime.now(timezone.utc).isoformat()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("snowlabel_benchmark_nonce", nonce)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", pnginfo=metadata, optimize=True)
    payload = encoded.getvalue()
    headers = {"X-Snow-User": args.user_id}

    upload, upload_seconds = post_timed(
        args.base_url.rstrip("/") + "/api/upload",
        headers=headers,
        files={"file": ("visa_everything.png", payload, "image/png")},
        timeout=60,
    )
    image_id = upload.json()["id"]
    request_body = {"image_id": image_id, "max_masks": args.max_masks}
    first, first_seconds = post_timed(
        args.base_url.rstrip("/") + "/api/segment_all",
        headers=headers,
        json=request_body,
        timeout=900,
    )
    second, second_seconds = post_timed(
        args.base_url.rstrip("/") + "/api/segment_all",
        headers=headers,
        json=request_body,
        timeout=120,
    )
    first_data = first.json()
    second_data = second.json()
    if first_data.get("from_cache") is not False:
        raise RuntimeError("first call was not an uncached Everything request")
    if second_data.get("from_cache") is not True:
        raise RuntimeError("second call did not hit the persistent image cache")
    if first_data.get("count") != second_data.get("count"):
        raise RuntimeError("cached and uncached calls returned different mask counts")

    result = {
        "status": "passed",
        "generated_at": nonce,
        "endpoint": "/api/segment_all",
        "source_image": args.image.relative_to(ROOT).as_posix(),
        "image_size": list(image.size),
        "max_masks": args.max_masks,
        "mask_count": first_data["count"],
        "limit_reached": first_data.get("limit_reached"),
        "timings_seconds": {
            "upload": round(upload_seconds, 3),
            "uncached": round(first_seconds, 3),
            "cached": round(second_seconds, 3),
        },
        "cache_speedup": round(first_seconds / max(second_seconds, 1e-9), 2),
        "assertions": {
            "first_from_cache": first_data["from_cache"],
            "second_from_cache": second_data["from_cache"],
            "same_mask_count": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    try:
        requests.post(
            args.base_url.rstrip("/") + "/api/reset",
            headers=headers,
            timeout=30,
        ).raise_for_status()
    except requests.RequestException as exc:
        print(f"warning: benchmark upload cleanup failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
