"""Start SnowLabel with the current interpreter and probe model-free endpoints."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def fetch(url: str) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.read(), response.headers.get_content_type()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "v0.1" / "environment_smoke.json",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env.update(
        {
            "SNOWLABEL_PORT": str(args.port),
            "SNOWLABEL_HOST": "127.0.0.1",
            "SNOWLABEL_MEMORY_BACKEND": "numpy",
            "SNOWLABEL_IMGCACHE": "0",
            "SNOWLABEL_MULTIUSER": "0",
            "SNOWLABEL_DATASET_DIR": str(ROOT / "demo_images"),
            "SNOWLABEL_DINO": str(
                ROOT / "weights" / "facebookdinov3-vitl16-pretrain-lvd1689m"
            ),
            "SNOWLABEL_SAM": str(ROOT / "sam2.1_t.pt"),
            "SNOWLABEL_RES": "512",
            "SNOWLABEL_PARTS_GRID": "4",
        }
    )
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [sys.executable, "-m", "snowlabel.server"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )

    started = time.perf_counter()
    base = f"http://127.0.0.1:{args.port}"
    error = "server did not become ready"
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                error = f"server exited with code {process.returncode}"
                break
            try:
                status, home, content_type = fetch(base + "/")
                if status == 200:
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.5)
        else:
            status, home, content_type = 0, b"", ""

        if status != 200:
            raise RuntimeError(error)
        stats_status, stats_body, _ = fetch(base + "/stats")
        dataset_status, dataset_body, _ = fetch(base + "/api/dataset")
        stats = json.loads(stats_body)
        dataset = json.loads(dataset_body)
        image_count = len(dataset.get("images", dataset if isinstance(dataset, list) else []))
        if stats_status != 200 or dataset_status != 200 or b"SnowLabel" not in home:
            raise RuntimeError("one or more endpoint assertions failed")

        result = {
            "status": "passed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "startup_seconds": round(time.perf_counter() - started, 3),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: metadata.version(name)
                for name in (
                    "snowlabel",
                    "torch",
                    "torchvision",
                    "ultralytics",
                    "transformers",
                    "fastapi",
                    "numpy",
                    "pymilvus",
                    "milvus-lite",
                    "label-studio-sdk",
                )
            },
            "checks": {
                "home": {"status": status, "content_type": content_type},
                "stats": {
                    "status": stats_status,
                    "memory_total": stats.get("memory", {}).get("total"),
                },
                "dataset": {"status": dataset_status, "image_count": image_count},
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=10)
        if output.strip():
            print(output[-4000:], file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
