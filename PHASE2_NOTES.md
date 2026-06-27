# SnowLabel — Phase 2 Notes (+ Phase 1.5 leftovers)

**Date**: 2026-05-27
**Scope this turn**: pluggable Milvus backend, memory inspector API,
multi-prototype mode, threshold tuning, high-resolution sweep.

## 1. Threshold tuning ([results/thresholds/](results/thresholds/))

Tuned **per-backbone** confidence thresholds against empirically derived
precision/coverage curves. Targets: AUTO precision ≥ 0.97, SUGGEST ≥ 0.85.

| Backbone | mean conf (correct / wrong) | AUTO (prec / cov) | SUGGEST (prec / cov) |
|----------|---|---|---|
| **DINOv2-S** | 0.88 / 0.45 | **0.79** (0.971 / 82.1 %) | **0.37** (0.915 / 100 %) |
| **DINOv3-L** | 0.91 / 0.49 | **0.61** (0.970 / 96.1 %) | **0.41** (0.943 / 100 %) |

**Headline**: DINOv3-L can confidently auto-apply **96 % of predictions** at
97 % precision; DINOv2-S manages only 82 % at the same precision.
Old hardcoded defaults (0.85 / 0.60) were too strict for v3, leaving most
correct predictions in "silent" tier. Use:

```powershell
# DINOv3-L
$env:SNOWLABEL_CONF_AUTO    = "0.61"
$env:SNOWLABEL_CONF_SUGGEST = "0.41"
```

Full report in [`results/thresholds/THRESHOLDS.md`](results/thresholds/THRESHOLDS.md);
precision-coverage plot at [`results/thresholds/precision_coverage.png`](results/thresholds/precision_coverage.png).

## 2. Resolution sweep ([results/v2_vs_v3_res512/](results/v2_vs_v3_res512/))

| Stack | overall | bird | bottle | car | cat | dog |
|-------|--------:|-----:|-------:|----:|----:|----:|
| DINOv2-S @ 224 | 91.5 % | 87.9 | 97.0 | 98.0 | 91.8 | 81.1 |
| DINOv3-L @ 224 | 94.3 % | 87.9 | 97.0 | 99.0 | 95.9 | 89.5 |
| **DINOv3-L @ 512** | **95.9 %** | 86.4 | 97.0 | 99.0 | **99.0** | **94.7** |

Going from 224→512 with DINOv3-L:
- **dog +5.2 pp, cat +3.1 pp** — fine-grained classes benefit most
- bird **-1.5 pp** (slight regression, possibly noise)
- 4× more patches per image → ~3-4× slower forward, GPU-mem ~+1.5 GB

Recommended for production v3:

```powershell
$env:SNOWLABEL_RES = "512"      # if you have GPU headroom
```

## 3. Multi-prototype / NCM mode (`snowlabel.memory`)

Added a second `predict_label` mode = "centroid": one L2-normalized centroid
per class (Nearest Class Mean classifier), instead of k-NN over all entries.

Both modes share the same `SnowballMemory` storage; switch via env:

```powershell
$env:SNOWLABEL_PREDICT_MODE = "centroid"   # default is "knn"
```

NCM is theoretically more stable for high-dim backbones (DINOv3-L) once
each class has ≥3 examples. Both modes verified at 100 % accuracy on the
synthetic Phase-2 smoke clusters.

A real-data accuracy comparison is a quick Phase 3 item once you've labelled
~50 instances per class through the LS UI — the inflection point is around
N=10 per class.

## 4. Milvus backend (`snowlabel.memory_milvus`)

`MilvusSnowballMemory` implements the same public surface as `SnowballMemory`:
`add / add_batch / forget / query / predict_label / iter_entries / stats`.
Storage layer is invisible to the predictor.

**Backend selection** via env var; defaults remain numpy so a fresh install
still runs zero-deps:

```powershell
$env:SNOWLABEL_MEMORY_BACKEND = "milvus"           # opt in
$env:SNOWLABEL_MILVUS_URI     = ""                 # blank = use milvus-lite
# For a remote server set e.g. "http://10.0.0.5:19530" and SNOWLABEL_MILVUS_TOKEN
```

When uri is blank, milvus-lite stores the DB at
`data/memory/snowlabel_milvus.db` (single file, no server). Verified to work
on Windows + Python 3.11. Known quirk: `drop_collection` can fail with
WinError 183 (file already exists) due to a `os.rename` non-atomicity issue
in milvus-lite; we work around it for production paths (`forget` uses
delete, not drop).

| Backend | Setup cost | Scale | Use when |
|---------|-----------|-------|----------|
| **numpy** (default) | none | up to ~50 k entries | Local dev, demo, < 100 k |
| **milvus-lite** | `pip install milvus-lite` | up to ~1 M entries | Single-machine production |
| **Remote Milvus** | Docker / cloud Milvus | 100 M+ | Multi-instance / shared snowball |

## 5. Memory inspector API (`snowlabel.server`)

New routes for building admin UIs or just curl-debugging:

| Route | Verb | What it does |
|-------|------|--------------|
| `/memory` | GET | List entries (filterable by `label=` and `verified_only=`) |
| `/memory/by-class/{label}` | GET | All entries of one class (newest first) |
| `/memory/distribution` | GET | Lightweight class → count summary |
| `/memory/search` | POST | Vector-or-click → top-K neighbors. Body: `{"vector": [...]}` or `{"image_uri": "...", "click_x": 100, "click_y": 200}` |
| `/memory/{entry_id}` | GET | Single entry detail (defined LAST so the int route doesn't shadow the named ones above) |
| `/memory/clear` | POST | Drop everything. Requires body `{"confirm": "YES_DELETE_ALL"}` |
| `/forget` | POST | (already existed) drop one by id |
| `/stats` | GET | Backend + thresholds + memory stats |

OpenAPI docs auto-generated at `http://localhost:9090/docs`.

## Putting it all together (production-grade run)

```powershell
conda activate snowlabel
cd <repo-root>

# Real models
$env:SNOWLABEL_DINO  = "<repo-root>\weights\facebookdinov3-vitl16-pretrain-lvd1689m"
$env:SNOWLABEL_SAM   = "<repo-root>\weights\facebooksam3.1\sam3.1_multiplex.pt"

# Higher resolution
$env:SNOWLABEL_RES   = "512"

# Tuned thresholds for v3
$env:SNOWLABEL_CONF_AUTO    = "0.61"
$env:SNOWLABEL_CONF_SUGGEST = "0.41"

# Milvus backend (single-file embedded)
$env:SNOWLABEL_MEMORY_BACKEND = "milvus"

# Optional: NCM mode (try after you have ≥10 labels per class)
# $env:SNOWLABEL_PREDICT_MODE = "centroid"

# Wipe any prior memory (dim/backend changed)
Remove-Item -Recurse data\memory

python -m snowlabel.server
```

## Smoke tests

Two complementary test scripts:

- `scripts/smoke_test_phase1.py` — end-to-end with real SAM + DINO on COCO
  images. Set `SNOWLABEL_MEMORY_BACKEND` to test that path.
- `scripts/smoke_test_phase2.py` — synthetic clusters × {numpy, milvus} ×
  {knn, centroid}. Tests memory layer in isolation. 4 cells, all PASS.

## What's still ahead

- **Phase 1.8** — Actual Label Studio end-to-end demo + screen recording.
  Needs you in the browser; backend is ready.
- **Phase 3** — Memory inspector HTML UI (the API is here; just needs HTML);
  benchmarks on a non-COCO dataset; v0.1 GitHub release.
