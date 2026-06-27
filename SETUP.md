# SnowLabel Setup Guide (Phase 0)

This guide walks you through running the Phase 0 validation experiment from a fresh Windows machine. Linux / macOS notes are at the bottom.

---

## Prerequisites check

We detected that **Python is not yet installed on this machine** (only the Microsoft Store stub at `C:\Users\...\WindowsApps\python.exe`, which is a redirect to the Store).

You have two options:

### Option A — Install Miniconda (recommended)

Conda makes managing Python + ML deps painless on Windows, and gives clean GPU/CPU torch installs.

1. Download Miniconda installer for Windows:
   https://docs.conda.io/en/latest/miniconda.html
   (pick the "Miniconda3 Windows 64-bit" installer)

2. Install with default options. Tick **"Add Miniconda3 to my PATH environment variable"** (yes, the installer warns against it; for personal use it's fine and simpler).

3. **Close and reopen PowerShell** so the PATH refreshes.

4. Verify:
   ```powershell
   conda --version
   python --version    # should now print Python 3.x
   ```

5. Create an isolated env for SnowLabel:
   ```powershell
   conda create -n snowlabel python=3.11 -y
   conda activate snowlabel
   ```

### Option B — Install Python 3.11 from python.org

1. Download from https://www.python.org/downloads/windows/ (any 3.10 / 3.11 / 3.12).

2. Run installer. **Tick "Add python.exe to PATH"** at the bottom of the first screen.

3. Verify in a new PowerShell window:
   ```powershell
   python --version
   ```

4. Create a venv:
   ```powershell
   cd <repo-root>
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   > If PowerShell complains about execution policy, run once:
   > `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Install dependencies

From the project root (`<repo-root>`) with your env activated:

### CPU-only (works everywhere, slow but fine for Phase 0)

```powershell
pip install -r requirements.txt
```

### With NVIDIA GPU (much faster: ~1 min vs ~30 min)

Pick the matching CUDA build first, then the rest:

```powershell
# Replace cu121 with cu118 / cu124 to match your driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU is visible:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

> **pycocotools on Windows**: if `pip install` fails with a compile error, you need
> Microsoft Visual C++ Build Tools (free). Easier alternative:
> `pip install pycocotools-windows` then add that name to a local override
> in requirements.txt.

---

## Download the dataset subset

This pulls down annotations (~241 MB temporary, deleted after extraction) and ~300 images (~50 MB) — total disk footprint when finished ~130 MB.

```powershell
python scripts/prepare_data.py --classes cat dog car person bottle --max-images 300
```

You should see:

```
[1/3] Downloading COCO annotations zip (~241 MB) -> data\coco\annotations_trainval2017.zip
[1/3] Extracting instances_val2017.json -> data\coco\annotations\instances_val2017.json
[1/3] Removing zip to save space (...)
[2/3] Loading instances_val2017.json (this is ~80 MB, takes a few seconds)
[2/3] Target classes: ['cat', 'dog', 'car', 'person', 'bottle']
[2/3] Images containing target classes: 2693 (will keep at most 300)
[2/3] Kept 300 images, ... instance annotations, 5 categories
[3/3] Downloading 300 images -> data\coco\val2017
images: 100%|████████████████| 300/300
...
Done. Subset ready at: <repo-root>\data\coco
```

If a download stalls, you can safely Ctrl+C and re-run — already-downloaded files are skipped.

---

## Run the Phase 0 notebook

The notebook lives at [`notebooks/phase0_validation.py`](notebooks/phase0_validation.py)
in the [jupytext "percent" format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format).
You have two ways to run it:

### A. Convert to .ipynb and use Jupyter

```powershell
jupytext --to ipynb notebooks/phase0_validation.py
jupyter notebook notebooks/phase0_validation.ipynb
```

### B. Open the .py file directly in VS Code

VS Code (with the Python extension) renders `# %%` cells as interactive notebook cells. Just open `notebooks/phase0_validation.py` and hit **"Run All"** above the first cell.

---

## What you should see

The notebook produces three plots in `results/`:

1. **`snowball_curve.png`** — the headline chart. Rolling accuracy on the
   next instance vs the number of labels already in memory.
   *We expect this curve to start near random and trend upward.*

2. **`snowball_buckets.png`** — same data, but bucketed for less noise.

3. **`snowball_confidence.png`** — accuracy stratified by prediction
   confidence. *High-confidence predictions should be nearly always correct.*

And a verdict printed at the end:

```
============================================================
Phase 0 verdict — automated checks:
  Random baseline           : 0.200
  Late-run accuracy         : 0.730
  2x baseline target        : 0.400  -> PASS ✅
  High-conf (>= 0.6) acc    : 0.910  -> PASS ✅
============================================================
```

If both PASS — Phase 0 is validated and we move on to Phase 1 (Label Studio backend with SAM 3 + DINOv3 + Milvus). If FAIL — the notebook's last section lists knobs to turn before giving up.

---

## Estimated time

| Hardware | Data prep | Notebook run |
|----------|-----------|--------------|
| CPU only | ~5–10 min | ~25–40 min |
| RTX 3060+ | ~5–10 min | ~1–2 min |

The CPU bottleneck is DINOv2 forward passes. Data download time is network-bound.

---

## Linux / macOS

Same flow, but bash:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py --classes cat dog car person bottle --max-images 300
jupytext --to ipynb notebooks/phase0_validation.py
jupyter notebook notebooks/phase0_validation.ipynb
```

---

## Troubleshooting

| Symptom | Likely fix |
|---------|-----------|
| `python --version` opens the Microsoft Store | You only have the Store stub. Install Python via Option A or B above. |
| `pip install pycocotools` compile error | Install [MS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or use `pip install pycocotools-windows`. |
| `torch.hub.load` SSL / connectivity error | DINOv2 fetches from GitHub; ensure you can reach github.com. Try setting `HTTPS_PROXY` if behind a corporate firewall. |
| OOM on GPU | Reduce `INPUT_RES` to 196 in the notebook, or use CPU. |
| Notebook runs but accuracy is at baseline | Confirm `MIN_MASK_PIXELS` filter is not removing all instances — print `class_inst_count` cell output. |

---

## Project layout reference

```
LabelAnything/
├── README.md                          ← project intro
├── RESEARCH_REPORT.md                 ← full design + competitor analysis
├── SETUP.md                           ← this file
├── requirements.txt
├── .gitignore
├── scripts/
│   └── prepare_data.py                ← downloads COCO subset
├── notebooks/
│   └── phase0_validation.py           ← THE experiment (jupytext .py)
├── data/coco/                         ← created by prepare_data.py
│   ├── annotations/
│   │   └── instances_val2017_snowlabel.json
│   └── val2017/
│       └── *.jpg
└── results/                           ← created by the notebook
    ├── snowball_curve.png
    ├── snowball_buckets.png
    ├── snowball_confidence.png
    └── phase0_predictions.csv
```
