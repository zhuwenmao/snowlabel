# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SnowLabel — Phase 0 Validation
#
# **Question we are answering:**
#
# > As a user labels more instances, does our DINO-feature + k-NN retrieval pipeline
# > predict the **next** instance's label with measurably higher accuracy?
#
# If the answer is **yes**, the entire SnowLabel project rests on solid ground.
# If **no**, we need to rethink the pooling / retrieval strategy before
# building any product UI.
#
# **Setup choices for this phase (justified):**
# - **Backbone**: DINOv2 ViT-S/14 (public via `torch.hub`). DINOv3 is strictly
#   stronger; if v2 works, v3 will work better. Phase 1 swaps it in.
# - **Masks**: COCO ground-truth masks. This isolates the *retrieval* hypothesis
#   from SAM segmentation quality (which is well-established). Phase 1 plugs in SAM 3.
# - **Vector store**: plain numpy cosine top-K. Phase 1 swaps in Milvus.
# - **Dataset**: a small COCO val 2017 subset prepared by `scripts/prepare_data.py`.

# %% [markdown]
# ## 1. Setup

# %%
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torchvision import transforms
from tqdm.auto import tqdm

# --- Configuration ---
DATA_ROOT = Path("data/coco")
ANN_PATH = DATA_ROOT / "annotations" / "instances_val2017_snowlabel.json"
IMG_DIR = DATA_ROOT / "val2017"

SEED = 42
K_NEIGHBORS = 5
MIN_MASK_PIXELS = 32 * 32           # skip tiny instances (DINO patches need area)
INPUT_RES = 224                     # DINOv2 ViT-S/14 default
MAX_INSTANCES_PER_IMAGE = 8         # safety bound for runtime
MAX_INSTANCES_PER_CLASS = 100       # class-balance cap; 0/None disables

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cpu":
    print("  (CPU mode — expect ~20-40 mins for 300 images. GPU is ~1 min.)")

# %% [markdown]
# ## 2. Load DINOv2 backbone
#
# We use `torch.hub` to fetch DINOv2 ViT-S/14 — small (21 M params), fast,
# and the patch-token features are what we'll pool by mask.

# %%
print("Loading DINOv2 ViT-S/14 from torch.hub ...")
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino = dino.to(device).eval()
PATCH = dino.patch_size       # 14
EMBED_DIM = dino.embed_dim    # 384

print(f"  patch_size={PATCH}, embed_dim={EMBED_DIM}")
assert INPUT_RES % PATCH == 0, (
    f"INPUT_RES={INPUT_RES} must be divisible by patch_size={PATCH}"
)
GRID = INPUT_RES // PATCH     # 224 / 14 = 16  ->  16x16 patch grid

# Standard ImageNet normalization (DINOv2 was trained with this)
preprocess = transforms.Compose([
    transforms.Resize((INPUT_RES, INPUT_RES), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

# %% [markdown]
# ## 3. Load COCO subset
#
# Expects `scripts/prepare_data.py` to have produced
# `data/coco/annotations/instances_val2017_snowlabel.json` and `data/coco/val2017/`.

# %%
if not ANN_PATH.exists():
    raise FileNotFoundError(
        f"{ANN_PATH} not found. Run first:\n"
        f"    python scripts/prepare_data.py --classes cat dog car person bottle "
        f"--max-images 300"
    )

coco = COCO(str(ANN_PATH))

cat_ids = sorted(coco.getCatIds())
cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
# Build a contiguous 0-indexed label space (COCO ids are sparse: 1, 3, 6, ...)
cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
idx_to_name = {i: cat_id_to_name[cid] for cid, i in cat_id_to_idx.items()}
N_CLASSES = len(cat_ids)

print(f"Classes ({N_CLASSES}): {[idx_to_name[i] for i in range(N_CLASSES)]}")

image_ids = sorted(coco.getImgIds())
print(f"Images: {len(image_ids)}")

# Count instances per class (after skipping tiny / crowd)
class_inst_count: Counter = Counter()
for img_id in image_ids:
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        if ann.get("iscrowd", 0):
            continue
        if ann["area"] < MIN_MASK_PIXELS:
            continue
        class_inst_count[cat_id_to_name[ann["category_id"]]] += 1
print("Instance counts per class (after filtering):")
for n, c in class_inst_count.most_common():
    print(f"  {n:<10} {c}")

# %% [markdown]
# ## 4. Mask-region feature extraction
#
# The core operation: take an image and a binary mask, produce a single
# L2-normalized vector that represents *just that masked region*.
#
# Approach:
# 1. Forward DINOv2 on the resized image; collect patch tokens (16×16 grid of 384-d).
# 2. Resize the mask to 16×16 (bilinear → soft weights).
# 3. Weighted-average pool the patch tokens by the mask weights.
# 4. L2-normalize.
#
# Why this works: DINOv2 patch features are known to be *dense and locally
# coherent* — they describe what each patch contains. A weighted mean over
# the patches inside an object gives a strong descriptor of that object.

# %%
@torch.inference_mode()
def encode_image_patches(img_pil: Image.Image) -> torch.Tensor:
    """Forward DINOv2 once per image; return patch tokens (GRID*GRID, EMBED_DIM)."""
    img_t = preprocess(img_pil).unsqueeze(0).to(device)
    feats = dino.forward_features(img_t)
    return feats["x_norm_patchtokens"][0]  # (N, D) on device


def pool_by_mask(
    patch_tokens: torch.Tensor,
    mask_hw: np.ndarray,
) -> np.ndarray:
    """Mask-weighted average pool over patch tokens, then L2-normalize.

    Args:
        patch_tokens: (GRID*GRID, EMBED_DIM) tensor from `encode_image_patches`.
        mask_hw: H×W binary numpy array at the original image size.
    """
    # Resize the mask down to the patch grid.
    m = Image.fromarray((mask_hw.astype(np.uint8) > 0).astype(np.uint8) * 255)
    m = m.resize((GRID, GRID), Image.BILINEAR)
    w = np.asarray(m, dtype=np.float32) / 255.0           # (GRID, GRID)
    w_flat = torch.from_numpy(w.flatten()).to(patch_tokens.device)  # (N,)

    if w_flat.sum() < 1e-3:
        feat = patch_tokens.mean(dim=0)
    else:
        feat = (patch_tokens * w_flat.unsqueeze(1)).sum(dim=0) / w_flat.sum()

    feat = feat / (feat.norm() + 1e-8)
    return feat.cpu().numpy().astype(np.float32)


def decode_coco_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """Decode a COCO annotation's segmentation to a binary HxW numpy mask."""
    seg = ann["segmentation"]
    if isinstance(seg, list):  # polygon
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


# %% [markdown]
# ## 5. The snowball experiment
#
# Walk over the images in a randomized order. For each instance in each image:
#
# 1. Extract its mask-pooled DINOv2 feature.
# 2. If memory has ≥1 entry, predict the label via cosine top-K vote.
# 3. Record `(prediction, ground-truth, confidence, memory size)`.
# 4. Add `(feature, ground-truth-label)` to memory.
#
# This simulates an idealized user who always corrects to the ground truth,
# which is exactly the regime we want to evaluate: *how fast does the model
# learn from a perfect labeler?*

# %%
def topk_vote(
    query: np.ndarray,
    memory: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Return (predicted_label, confidence, top_sims, top_labels)."""
    sims = memory @ query
    k_eff = min(k, len(memory))
    if k_eff == 1:
        top_idx = np.array([int(np.argmax(sims))])
    else:
        top_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
    top_sims = sims[top_idx]
    top_lbls = labels[top_idx]
    # Similarity-weighted vote. Clip negatives to 0 so we don't get weird scores.
    weights = np.clip(top_sims, 0.0, None)
    scores: dict[int, float] = defaultdict(float)
    for w, l in zip(weights, top_lbls):
        scores[int(l)] += float(w)
    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    total = sum(scores.values())
    conf = scores[pred] / total if total > 0 else 0.0
    return pred, conf, top_sims, top_lbls


# Build the (image, list of instances) list, with filtering, then shuffle.
records = []
for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    keep = [
        a for a in anns
        if not a.get("iscrowd", 0) and a["area"] >= MIN_MASK_PIXELS
    ]
    if not keep:
        continue
    keep = keep[:MAX_INSTANCES_PER_IMAGE]
    records.append((img_id, img_info, keep))

random.Random(SEED).shuffle(records)

# Apply per-class instance cap to balance the experiment.
if MAX_INSTANCES_PER_CLASS:
    per_class_seen: Counter = Counter()
    balanced_records = []
    for img_id, info, anns in records:
        kept_anns = []
        for a in anns:
            cls = cat_id_to_idx[a["category_id"]]
            if per_class_seen[cls] < MAX_INSTANCES_PER_CLASS:
                kept_anns.append(a)
                per_class_seen[cls] += 1
        if kept_anns:
            balanced_records.append((img_id, info, kept_anns))
    print(f"After per-class cap ({MAX_INSTANCES_PER_CLASS}/class):")
    for idx in range(N_CLASSES):
        print(f"  {idx_to_name[idx]:<10} {per_class_seen[idx]}")
    records = balanced_records

print(f"Images contributing instances: {len(records)}")
print(f"Total instances to process: {sum(len(r[2]) for r in records)}")

# %%
# Run the experiment
mem_vecs: list[np.ndarray] = []
mem_lbls: list[int] = []
results: list[dict] = []

for img_id, img_info, anns in tqdm(records, desc="snowball"):
    img_path = IMG_DIR / img_info["file_name"]
    if not img_path.exists():
        continue
    img_pil = Image.open(img_path).convert("RGB")
    H, W = img_info["height"], img_info["width"]

    # Forward DINOv2 once per image — pool per instance is cheap from here.
    patch_tokens = encode_image_patches(img_pil)

    for ann in anns:
        mask = decode_coco_mask(ann, H, W)
        feat = pool_by_mask(patch_tokens, mask)
        gt = cat_id_to_idx[ann["category_id"]]

        if len(mem_vecs) >= 1:
            M = np.stack(mem_vecs)
            L = np.asarray(mem_lbls)
            pred, conf, top_sims, top_lbls = topk_vote(feat, M, L, K_NEIGHBORS)
            correct = int(pred == gt)
        else:
            pred, conf = -1, 0.0
            correct = None

        results.append({
            "img_id": img_id,
            "gt": gt,
            "pred": pred,
            "conf": conf,
            "mem_size": len(mem_vecs),
            "correct": correct,
        })
        mem_vecs.append(feat)
        mem_lbls.append(gt)

print(f"Final memory size: {len(mem_vecs)} instances, {N_CLASSES} classes")

# %% [markdown]
# ## 6. Results: does the snowball roll?

# %%
import pandas as pd  # noqa: E402  (imported here so the data cell stays self-contained)

df = pd.DataFrame(results)
scored = df[df["correct"].notna()].copy()
scored["correct"] = scored["correct"].astype(int)
random_baseline = 1.0 / N_CLASSES
print(f"Random-guess baseline: {random_baseline:.3f}  (1 / {N_CLASSES} classes)")
print(f"Overall accuracy:      {scored['correct'].mean():.3f}")
print(f"  (over {len(scored)} predictions; the first instance has no memory and is excluded)")

# Rolling accuracy
WINDOW = max(20, len(scored) // 30)
scored["rolling_acc"] = (
    scored["correct"].rolling(window=WINDOW, min_periods=1).mean()
)

# Bucketed by memory size
BUCKET = max(20, len(scored) // 20)
scored["bucket"] = (scored["mem_size"] // BUCKET) * BUCKET
bucketed = scored.groupby("bucket")["correct"].agg(["mean", "count"]).reset_index()

# Confidence-thresholded accuracy
THRESH = 0.6
hi_conf = scored[scored["conf"] >= THRESH]
print(
    f"\nWith confidence threshold {THRESH}:  "
    f"coverage {len(hi_conf) / len(scored):.1%}, "
    f"accuracy {hi_conf['correct'].mean():.3f}  "
    f"(over {len(hi_conf)} predictions)"
)

# %% [markdown]
# ### Plot 1: rolling accuracy vs cumulative instances seen
# This is **THE** chart. If it goes up, the snowball rolls.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(scored["mem_size"], scored["rolling_acc"], lw=2, color="#1f77b4",
        label=f"Rolling accuracy (window={WINDOW})")
ax.axhline(random_baseline, color="#d62728", ls="--", lw=1.5,
           label=f"Random baseline ({random_baseline:.2f})")
ax.set_xlabel("Number of labeled instances in memory")
ax.set_ylabel("Accuracy on next instance")
ax.set_ylim(0, 1)
ax.set_title("The Snowball Curve — does accuracy lift as memory grows?")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
Path("results").mkdir(exist_ok=True)
plt.savefig("results/snowball_curve.png", dpi=140)
plt.show()

# %% [markdown]
# ### Plot 2: bucketed accuracy (less noisy)

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(bucketed["bucket"], bucketed["mean"], width=BUCKET * 0.9,
       align="edge", color="#2ca02c", alpha=0.7, edgecolor="black")
ax.axhline(random_baseline, color="#d62728", ls="--", lw=1.5,
           label=f"Random baseline ({random_baseline:.2f})")
for _, row in bucketed.iterrows():
    ax.text(row["bucket"] + BUCKET / 2, row["mean"] + 0.01,
            f"n={int(row['count'])}", ha="center", fontsize=8)
ax.set_xlabel("Memory size at prediction time (bucketed)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Accuracy by memory-size bucket")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("results/snowball_buckets.png", dpi=140)
plt.show()

# %% [markdown]
# ### Plot 3: confidence distribution and accuracy-by-confidence

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(scored["conf"], bins=20, color="#9467bd", edgecolor="black")
axes[0].set_xlabel("Top-K vote confidence")
axes[0].set_ylabel("# predictions")
axes[0].set_title("Confidence distribution")
axes[0].grid(True, alpha=0.3)

# Accuracy stratified by confidence bucket
scored["conf_bucket"] = pd.cut(scored["conf"], bins=[0, 0.3, 0.5, 0.7, 0.85, 1.01])
by_conf = scored.groupby("conf_bucket", observed=True)["correct"].agg(
    ["mean", "count"]
).reset_index()
labels = [str(b) for b in by_conf["conf_bucket"]]
axes[1].bar(range(len(by_conf)), by_conf["mean"], color="#ff7f0e",
            edgecolor="black", alpha=0.8)
for i, row in enumerate(by_conf.itertuples()):
    axes[1].text(i, row.mean + 0.01, f"n={int(row.count)}",
                 ha="center", fontsize=8)
axes[1].axhline(random_baseline, color="#d62728", ls="--", lw=1.5)
axes[1].set_xticks(range(len(by_conf)))
axes[1].set_xticklabels(labels, rotation=20, ha="right")
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy by confidence bucket")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results/snowball_confidence.png", dpi=140)
plt.show()

# %% [markdown]
# ## 7. Decision rule
#
# Pass / fail criteria for Phase 0:
#
# 1. **Snowball curve trends UP** — rolling accuracy late in the run should be
#    significantly higher than early. We're looking for at least a **2× lift over
#    random baseline** by the time memory has 100+ entries.
# 2. **High-confidence predictions are very accurate** — at confidence ≥ 0.6
#    we want > 0.85 accuracy. This is what makes the UX "feel smart": the model
#    can be selectively quiet when uncertain.
# 3. **No catastrophic collapse** — accuracy doesn't *fall* as memory grows
#    (which would indicate feature drift / class confusion).

# %%
# Save the CSV FIRST (before any print that might hit Windows codec issues).
Path("results").mkdir(exist_ok=True)
csv_path = Path("results/phase0_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"Saved per-instance predictions: {csv_path}")

final_window = scored.tail(max(50, len(scored) // 10))["correct"].mean()
print("=" * 60)
print("Phase 0 verdict - automated checks:")
print(f"  Random baseline           : {random_baseline:.3f}")
print(f"  Late-run accuracy         : {final_window:.3f}")
verdict_1 = "PASS" if final_window >= 2 * random_baseline else "FAIL"
print(f"  2x baseline target        : {2 * random_baseline:.3f}  -> {verdict_1}")
if len(hi_conf) > 20:
    hi_acc = hi_conf["correct"].mean()
    verdict_2 = "PASS" if hi_acc >= 0.85 else "NEEDS-IMPROVEMENT"
    print(f"  High-conf (>= {THRESH}) acc   : {hi_acc:.3f}  -> {verdict_2}")
else:
    print(f"  Too few high-conf samples to evaluate ({len(hi_conf)} < 20)")
print("=" * 60)
print()
print("If both PASS - proceed to Phase 1 (Label Studio backend integration).")
print("If FAIL      - see notebook end for next steps to improve the algorithm.")

# %% [markdown]
# ## 8. If results disappoint: knobs to turn before giving up
#
# Things to try (roughly in order of likely impact):
#
# 1. **Bigger backbone**: switch to `dinov2_vitb14` (768-d, more discriminative)
#    or directly to **DINOv3** (gated download but worth the effort).
# 2. **Higher input resolution**: bump `INPUT_RES` to 518 → 37×37 patch grid;
#    catches finer-grained class differences.
# 3. **Concatenate CLS + mask-pool**: include the global image CLS token
#    alongside the mask-pooled patch tokens — gives the model context.
# 4. **Sharper mask weighting**: instead of bilinear resize, use binary mask
#    + only include patches with > 50 % overlap.
# 5. **Multi-prototype per class**: store *several* exemplars per class and
#    average them, then query against prototypes (closer to NCM classifier).
# 6. **Re-weight by recency / verification**: trust the user's most recent
#    labels more, deprioritize older noisy ones.

# %% [markdown]
# ## 9. Notes on saved results
#
# - `results/phase0_predictions.csv` — every prediction with conf, GT, memory size
# - `results/snowball_curve.png` — the headline rolling-accuracy curve
# - `results/snowball_buckets.png` — bucketed view
# - `results/snowball_confidence.png` — confidence calibration

# %%
print(f"All artifacts in: results/")
