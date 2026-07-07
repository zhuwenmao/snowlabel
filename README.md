<h1 align="center">SnowLabel</h1>

<p align="center">
Progressive mask auto-labeling with SAM, DINO feature memory, and FastAPI.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110%2B-009688">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

## Description

SnowLabel is an interactive auto-labeling tool for instance masks. It uses SAM
to generate masks, DINO features to encode taught objects, and vector memory to
pre-label similar objects in later images. The model is not retrained during
labeling; every useful correction becomes searchable memory.

The project is designed for visual annotation workflows where class definitions
are discovered progressively, such as industrial inspection, fine-grained object
labeling, and dataset bootstrapping.

## Features

- Canvas-based labeling UI with point, box, and whole-image segmentation modes.
- Progressive memory that reuses corrected masks to pre-fill future labels.
- Two-stage retrieval with whole-object features, part features, and geometry.
- FastAPI service exposing the web UI, labeling APIs, memory inspector, and
  legacy Label Studio ML-compatible endpoints.
- Pluggable memory backend with local numpy storage and optional Milvus
  persistence.
- Public demo startup script with MariaDB, MinIO, and Cloudflare tunnel support.

## Prerequisites & Installation

Requirements:

- Python 3.10 or newer.
- NVIDIA GPU recommended for practical SAM and DINO inference.
- Local DINOv3 weights are recommended for the production configuration.

Clone and install:

```powershell
git clone https://github.com/zhuwenmao/snowlabel.git
cd snowlabel
pip install -r requirements.txt
pip install -e .[dev]
pip install ultralytics transformers pymilvus
```

For GPU execution, install the PyTorch and torchvision wheels that match your
CUDA version before installing the remaining dependencies.

## Quick Start / Usage

Start the local FastAPI service:

```powershell
$env:SNOWLABEL_DINO="weights\facebookdinov3-vitl16-pretrain-lvd1689m"
$env:SNOWLABEL_MEMORY_BACKEND="milvus"
$env:SNOWLABEL_RES="512"
$env:SNOWLABEL_PARTS_GRID="4"
python -m snowlabel.server
```

Open the application:

```text
http://localhost:9090/
http://localhost:9090/try
http://localhost:9090/inspector
```

Start the public demo:

```powershell
powershell -ExecutionPolicy Bypass -File start_public_demo.ps1
```

The fixed public URL is:

```text
https://snowlabel.zhuwenmao.com
```

## Repository Structure

```text
|-- snowlabel/                 # Core Python package and FastAPI service
|-- snowlabel/static/          # Home page, Canvas labeler, and inspector UI
|-- scripts/                   # Evaluation, smoke-test, and maintenance scripts
|-- eval/                      # Supervision protocol and quality gates
|-- demo_images/               # Small demo image set
|-- docs/                      # Design notes and project decisions
|-- start_public_demo.ps1      # Public demo launcher
|-- requirements.txt           # Runtime dependency list
|-- pyproject.toml             # Package metadata and development extras
`-- README.md                  # Project overview
```

## FAQ

**Is Label Studio required?**

No. The current primary UI is SnowLabel's own FastAPI and Canvas labeler. Label
Studio-compatible endpoints are kept for legacy integration.

**What happens if DINOv3 weights are unavailable?**

SnowLabel can fall back to the default DINOv2 configuration. Do not use the
DINOv3-specific `SNOWLABEL_RES=512` setting unless the selected backbone supports
that resolution.

**When should I use `start_public_demo.ps1`?**

Use it when you need the full public demo stack, including the local service,
MariaDB, MinIO, and Cloudflare tunnel.

## Contributing

Pull requests are welcome. Keep changes scoped, avoid committing local data or
model weights, and run the checks that match the modified area.

Common validation commands:

```powershell
python scripts/test_protocol.py
python scripts/check_feature_space_guard.py
python scripts/check_api_edges.py
python scripts/check_durability.py
python scripts/smoke_webtool.py
```

For changes to feature extraction, retrieval, thresholds, memory backends, model
selection, resolution, or part-grid settings, run:

```powershell
python scripts/supervise.py --quick --tag <name> --compare
python scripts/supervise.py --tag <name> --compare
```

## License

SnowLabel is released under the MIT License. See `LICENSE` for details.

## Acknowledgments / Citations

SnowLabel builds on SAM and DINO from Meta AI, FastAPI, Milvus, and the Label
Studio ML backend protocol. Please follow the licenses and usage terms of all
upstream models and libraries when distributing derivative work.
