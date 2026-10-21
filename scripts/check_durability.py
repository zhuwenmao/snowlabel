"""Verify live memory persistence across a SnowLabel server restart.

Use a dedicated multi-user tenant so the check never touches real user data:

    python scripts/check_durability.py commit
    # restart the server
    python scripts/check_durability.py verify
    python scripts/check_durability.py cleanup
"""

from __future__ import annotations

import argparse
import io

import requests
from PIL import Image, ImageDraw


LABEL = "__snowlabel_durability_check__"


def request_json(
    method: str,
    base_url: str,
    path: str,
    headers: dict[str, str],
    **kwargs,
) -> dict:
    response = requests.request(
        method,
        base_url.rstrip("/") + path,
        headers=headers,
        timeout=kwargs.pop("timeout", 120),
        **kwargs,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("detail"):
        raise RuntimeError(f"{path}: {data['detail']}")
    return data


def distribution(base_url: str, headers: dict[str, str]) -> dict:
    return request_json("GET", base_url, "/memory/distribution", headers)


def test_image() -> bytes:
    image = Image.new("RGB", (320, 240), "#343943")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 45, 250, 195), radius=24, fill="#d8e2ef")
    draw.ellipse((125, 85, 195, 155), fill="#5c6f82")
    encoded = io.BytesIO()
    image.save(encoded, "PNG")
    return encoded.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("commit", "verify", "cleanup"))
    parser.add_argument("--base-url", default="http://127.0.0.1:9090")
    parser.add_argument("--user-id", default="snowlabel_durability_check")
    args = parser.parse_args()
    headers = {"X-Snow-User": args.user_id}

    if args.mode == "commit":
        request_json("POST", args.base_url, "/api/reset", headers)
        uploaded = request_json(
            "POST",
            args.base_url,
            "/api/upload",
            headers,
            files={"file": ("durability.png", test_image(), "image/png")},
        )
        segmented = request_json(
            "POST",
            args.base_url,
            "/api/segment",
            headers,
            json={
                "image_id": uploaded["id"],
                "type": "point",
                "points": [[160, 120]],
                "labels": [1],
            },
            timeout=300,
        )
        if not segmented.get("ok") or not segmented.get("mask_png"):
            raise RuntimeError("point segmentation returned no mask")
        committed = request_json(
            "POST",
            args.base_url,
            "/api/commit",
            headers,
            json={
                "image_id": uploaded["id"],
                "mask_png": segmented["mask_png"],
                "label": LABEL,
            },
            timeout=300,
        )
        print(
            f"committed entry={committed['entry_id']} "
            f"memory={committed['memory_size']}"
        )
        print("distribution:", distribution(args.base_url, headers))
        return 0

    if args.mode == "verify":
        data = distribution(args.base_url, headers)
        ok = data.get("classes", {}).get(LABEL, 0) >= 1
        print("distribution after restart:", data)
        print("DURABILITY OK" if ok else "DURABILITY FAILED")
        return 0 if ok else 1

    reset = request_json("POST", args.base_url, "/api/reset", headers)
    print("cleanup:", reset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
