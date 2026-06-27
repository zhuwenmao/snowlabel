# SnowLabel Phase 1 — Label Studio Integration Guide

This guide gets you from "code is written" to "end-to-end demo working in your
browser". Approximate time: 15 minutes.

## What Phase 1 delivers

A working **Label Studio plug-in** where:

1. You open an image in Label Studio.
2. You click a point on an object.
3. The SnowLabel backend (running as a separate process) gets the click,
   runs SAM 2.1 to segment the object, extracts a DINOv2 feature for that
   mask, queries the snowball memory by cosine k-NN, and returns:
   - The mask (rendered as a brush stroke in LS)
   - The predicted class label (pre-filled in the dropdown)
   - The confidence (visible as the prediction score)
4. You accept or correct the label and save.
5. On save, the LS webhook sends the confirmed `(mask, label)` to the backend,
   which stores it in the snowball memory. The next prediction is now informed
   by your correction.

## Status of components

| Component | Phase 1 choice | Why | Phase 2 upgrade |
|-----------|---------------|-----|-----------------|
| Segmentation | **SAM 2.1 Tiny** (via `ultralytics`) | Apache 2.0, no gating, pip-installable on Windows with zero compile | SAM 3 — one-line swap once you have HF access |
| Features | **DINOv2 ViT-B/14** (via `torch.hub`) | Free, no gating, validated in Phase 0 | DINOv3 ViT-B/16 — one-line swap once you accept Meta's gate |
| Vector store | **numpy + JSON** (`SnowballMemory`) | Zero deps, Windows-native, supports up to ~50k entries fast | Milvus / Qdrant — same `SnowballMemory` interface |
| Server | **FastAPI** | Auto-OpenAPI docs at `/docs`, native async, easy to test | Same |

## Architecture

```
                          +-----------------------------+
                          |       Browser (you)         |
                          |     http://localhost:8080   |
                          +--------------+--------------+
                                         |
                              POST /predict / webhooks
                                         v
        +--------------------------------+--------------------------------+
        |                      Label Studio (LS)                          |
        |     listens on :8080, stores tasks + annotations in SQLite      |
        +--------------------------------+--------------------------------+
                                         |
                  interactive predict    |    annotation webhook
                  (LS protocol JSON)     v
        +--------------------------------+--------------------------------+
        |              SnowLabel ML Backend  (this repo)                  |
        |              listens on :9090  -- FastAPI + uvicorn             |
        |                                                                 |
        |   POST /predict   -- interactive: point/box -> mask + label     |
        |   POST /predict   -- auto: image -> all masks + labels          |
        |   POST /webhook   -- learn from saved annotations               |
        |   GET  /stats     -- memory size, classes, thresholds           |
        |   GET  /memory    -- list stored snowball entries               |
        |   POST /forget    -- drop an entry (memory inspector)           |
        |                                                                 |
        |  Internally:                                                    |
        |    SamSegmenter         (ultralytics SAM 2.1 Tiny)              |
        |    DinoFeatureExtractor (torch.hub DINOv2 ViT-B/14)             |
        |    SnowballMemory       (data/memory/embeddings.npy + jsonl)    |
        +-----------------------------------------------------------------+
```

## Quickstart

### 1. Start the SnowLabel backend

```powershell
# In one PowerShell window
conda activate snowlabel   # the env we built in Phase 0
cd <repo-root>
python -m snowlabel.server
```

You should see:

```
INFO     Starting SnowLabel server on 0.0.0.0:9090
INFO     Memory dir: <repo-root>\data\memory
INFO     Application startup complete.
INFO     Uvicorn running on http://0.0.0.0:9090
```

First request will trigger model downloads:
- SAM 2.1 Tiny  ~150 MB (auto-downloaded by ultralytics)
- DINOv2 ViT-B/14  ~330 MB (auto-downloaded by torch.hub)

Visit http://localhost:9090/docs in your browser — you should see the
auto-generated OpenAPI page with all endpoints.

### 2. Start Label Studio

```powershell
# In a SECOND PowerShell window
conda activate snowlabel
cd <repo-root>
label-studio start --port 8080
```

Open http://localhost:8080 in your browser, create an account on first run.

### 3. Create a SnowLabel project

1. Click **Create Project**.
2. Name it "SnowLabel demo".
3. **Data Import**: upload some images. The COCO subset under
   `data/coco/val2017/` works great — pick 30–50 images that cover your classes.
4. **Labeling Setup** → **Custom template** tab → paste the contents of
   `label_studio_config.xml`. Click **Save**.

### 4. Connect the ML backend

1. Project **Settings** → **Model**.
2. Click **Connect Model**.
3. URL: `http://localhost:9090`
4. Name: `SnowLabel`
5. Tick **Use for interactive preannotations** and **Use for training (webhook)**.
6. Click **Validate and Save**.

### 5. Configure the webhook (so we learn from saves)

1. Project **Settings** → **Webhooks**.
2. Click **Add Webhook**.
3. URL: `http://localhost:9090/webhook`
4. Events to send: `Annotation Created`, `Annotation Updated`.
5. Save.

### 6. Demo the snowball

1. Click **Label All Tasks**.
2. On the first image: click somewhere on an object. SAM segments it. The
   class dropdown shows nothing (memory is empty). Pick the right class,
   click **Submit**.
3. On the second image: click. SAM segments. If the object resembles what you
   just labeled, the class dropdown is pre-filled with that class.
4. After 5–10 images, most predictions arrive pre-labeled correctly.
5. Visit http://localhost:9090/stats to see the memory growing.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|-------------------|
| Backend connect fails ("Could not validate") | Backend probably not running, or `localhost` vs `0.0.0.0` mismatch. Try `http://127.0.0.1:9090`. |
| First predict takes ~30 s | Models are downloading on first call. Watch the backend logs. |
| "No labels predicted" | Memory may still be empty, or below `conf_suggest` threshold. Label 5–10 instances manually to seed the snowball. |
| Image URLs return 401 from LS | Your backend needs `LABEL_STUDIO_API_KEY` env var so it can fetch images stored inside LS. Get the token from LS → Account & Settings → Access Token, then `set LABEL_STUDIO_API_KEY=<token>` before starting the backend. |
| Webhook isn't firing | Re-check the project's webhook list. The destination must be reachable from the LS process (use `host.docker.internal:9090` if LS runs in Docker). |

## Upgrading to SAM 3 + DINOv3 (after Meta access approval)

Both SAM 3 and DINOv3 sit behind Meta's HuggingFace gate (you click "Agree"
once, per model). The SnowLabel codebase is already aware of both — only
weights are missing on a fresh machine.

```powershell
# 1. Run the guided setup (idempotent; safe to re-run anytime)
python scripts/setup_sam3_dinov3.py

# It will tell you exactly what to do:
#   - install huggingface_hub if missing
#   - prompt you to `huggingface-cli login` if not authed
#   - point you at the SAM 3 page to click "Agree" + download sam3.pt
#   - point you at the DINOv3 page to click "Agree" (downloads automatically)

# 2. Once done, swap the two env vars and start the server
$env:SNOWLABEL_SAM  = "sam3.pt"
$env:SNOWLABEL_DINO = "dinov3_vitb16"

# 3. The existing snowball memory is keyed by DINO embedding dim. Switching
#    DINOv2 (384/768/1024) <-> DINOv3 (384/768/1024) changes the dim, so
#    you'll get a "stored embeddings shape ... does not match" error.
#    Reset the memory:
Remove-Item -Recurse data/memory   # PowerShell
# rm -rf data/memory               # bash

python -m snowlabel.server
```

To verify what's set up:

```powershell
python scripts/setup_sam3_dinov3.py --check
#   HuggingFace login : YES (your-username)
#   SAM 3 weights     : YES (.../sam3.pt)
#   DINOv3 cached     : YES
#   ALL SET.
```

**Why bother upgrading?**

| Aspect | SAM 2.1 Tiny + DINOv2-B | SAM 3 + DINOv3-B |
|--------|------------------------|------------------|
| Mask quality | strong | **~2× better** per Meta paper |
| Concept understanding | none (point/box only) | **text prompts + image exemplars** |
| Feature discriminativeness | proven 91.5 % in Phase 0 | ~10 % higher on hard classes |
| License | Apache 2.0, open | Meta custom (commercial OK, attribution req.) |
| Weight download | fully automatic | manual `sam3.pt` + HF login |

## Configuration via env vars

All in `snowlabel/config.py`:

```powershell
# Use the bigger DINO backbone (slower but more discriminative)
$env:SNOWLABEL_DINO = "dinov2_vitl14"

# Use a different SAM checkpoint
$env:SNOWLABEL_SAM = "sam2.1_b.pt"

# Tune the confidence tiers
$env:SNOWLABEL_CONF_AUTO = "0.80"      # auto-apply threshold (default 0.85)
$env:SNOWLABEL_CONF_SUGGEST = "0.50"   # suggest threshold (default 0.60)

# Different port
$env:SNOWLABEL_PORT = "9000"
```

Then start the backend as usual.

## Where the memory lives

```
data/memory/
  embeddings.npy   # (N, 384) float32 — the actual feature vectors
  metadata.jsonl   # one JSON line per entry (label, image, verified, ...)
  index.json       # versioned header — class list, total count
```

Delete the folder to reset the snowball.

## Phase 1.5 / Phase 2 hooks

The codebase is structured so that:
- Swapping `SnowballMemory` for a Milvus-backed implementation requires only
  matching its public methods: `add`, `query`, `predict_label`, `forget`.
- Swapping `SamSegmenter` (e.g. for SAM 3) only requires matching
  `predict_from_point`, `predict_from_box`, `predict_all`.
- Swapping `DinoFeatureExtractor` (e.g. for DINOv3) only requires matching
  `feature_for_mask` and `embed_dim`.

No caller cares about the implementation — they all go through these methods.
