"""Verify that Python, package imports, editable path, and model weights agree."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEY_PACKAGES = (
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-weights", action="store_true")
    args = parser.parse_args()

    import snowlabel
    import torch

    package_path = Path(snowlabel.__file__).resolve()
    current_checkout = package_path.is_relative_to(ROOT)
    dino_dir = ROOT / "weights" / "facebookdinov3-vitl16-pretrain-lvd1689m"
    dino_ok = (dino_dir / "config.json").is_file() and any(
        dino_dir.glob("*.safetensors")
    )
    sam_path = ROOT / "sam2.1_t.pt"
    sam_ok = sam_path.is_file()
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "snowlabel_path": str(package_path),
        "current_checkout": current_checkout,
        "packages": {name: metadata.version(name) for name in KEY_PACKAGES},
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "weights": {
            "dino": str(dino_dir),
            "dino_ok": dino_ok,
            "sam": str(sam_path),
            "sam_ok": sam_ok,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not current_checkout:
        print("ERROR: snowlabel imports from a different checkout", file=sys.stderr)
        return 2
    if args.require_weights and not (dino_ok and sam_ok):
        print("ERROR: required production weights are incomplete", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
