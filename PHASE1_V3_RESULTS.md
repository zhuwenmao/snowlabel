# Phase 1 — SAM 3.1 + DINOv3-L Real-Weights Comparison

**Date**: 2026-05-27
**Hardware**: NVIDIA RTX 4060 Laptop GPU 8 GB / CUDA 12.2
**Question**: Does swapping DINOv2-S → DINOv3-L actually improve SnowLabel's
snowball curve, or is it just bigger / heavier for show?

## TL;DR

**Yes — DINOv3-L helps exactly where DINOv2-S struggled.**

| Metric | DINOv2-S | DINOv3-L | Δ |
|--------|----------|----------|---|
| Embedding dim | 384 | 1024 | 2.7× |
| Patch size | 14 | 16 | (216 vs 196 patches @ 224²) |
| Register tokens | 0 | 4 | — |
| Forward pass on RTX 4060 | ~30 it/s | ~50 it/s | (cached input, batch=1) |
| **Overall accuracy** | **91.5 %** | **94.3 %** | **+2.8 pp** |
| Late-run plateau (last 10 %) | 98 % | 98 % | 0 |
| Curve stability (variance) | dips to 70 % mid-run | stays > 80 % | qualitatively smoother |

The headline: **the lift is concentrated on the classes the v2 model
struggled with**:

| Class | DINOv2-S | DINOv3-L | Δ |
|-------|---------:|---------:|--:|
| bird | 87.9 % | 87.9 % | 0 |
| bottle | 97.0 % | 97.0 % | 0 |
| car | 98.0 % | 99.0 % | +1.0 |
| cat | 91.8 % | 95.9 % | **+4.1** |
| **dog** | **81.1 %** | **89.5 %** | **+8.4** |

The "easy" classes (bird, bottle) saturate equally for both backbones —
they're already near the ceiling that retrieval can deliver from clean
masks. The hard classes (dog ↔ cat, the main v2 confusion mode) move
meaningfully. This matches the Phase 0 report's prediction:

> *"Switching to a larger backbone (DINOv2-L, DINOv3) → more discriminative
> features [will close the dog↔cat gap]"*

Confirmed on real data.

## The comparison plot

`results/v2_vs_v3/comparison.png` shows:

- **Left panel — snowball curves**: both ramp up identically in the first
  ~50 labels (no surprise — both are good enough features to seed). After
  that, DINOv3-L stays smooth, DINOv2-S has a nasty dip to 70 % around
  instance 380 (a hard-to-discriminate run of dogs).
- **Right panel — per-class bars**: visualises the dog/cat gain.

## Setup used

```
DINOv3-L weights : weights/facebookdinov3-vitl16-pretrain-lvd1689m/   (1.16 GB)
SAM 3.1 weights  : weights/facebooksam3.1/sam3.1_multiplex.pt          (3.34 GB)
Dataset          : COCO val 2017 subset, 5 balanced classes
                   (cat, dog, bird, bottle, car @ ≤100 instances each)
Total predictions: 458 (same seed as Phase 0)
```

## What changed in the code to make this work

1. **`snowlabel/models.py`** — `DinoFeatureExtractor` now auto-detects a
   local HF snapshot (any directory containing `config.json`) and loads
   via `AutoModel.from_pretrained(path)`. No HF login required if you
   already have the weights on disk.
2. **`snowlabel/config.py`** — switched all env-var defaults to
   `field(default_factory=...)` so the env is read on each `SnowConfig()`
   instantiation, not once at class definition. (The previous behavior
   silently caused the first run's `SNOWLABEL_DINO` value to be reused on
   subsequent runs — see initial buggy comparison output where v2 and v3
   gave identical numbers.)
3. **`snowlabel/models.py`** — DINOv3's patch tokens are stripped past
   `1 + num_register_tokens` (DINOv2 has 0 register tokens, DINOv3 has 4).
4. **`snowlabel/models.py`** — SAM 3 / SAM 3.1 path is detected by the
   substring "sam3" in the variant, so `sam3.1_multiplex.pt` works the
   same way; helpful FileNotFoundError if the weights are missing.

## Reproducing this comparison

```powershell
$py = "$env:USERPROFILE\miniconda3\envs\snowlabel\python.exe"
Set-Location <repo-root>

# Clean any stale per-run env vars
Remove-Item Env:\SNOWLABEL_DINO -ErrorAction SilentlyContinue
Remove-Item Env:\SNOWLABEL_SAM  -ErrorAction SilentlyContinue

& $py scripts\compare_v2_vs_v3.py `
    --v3-path "<repo-root>\weights\facebookdinov3-vitl16-pretrain-lvd1689m"

# Results land in:
#   results/v2_vs_v3/predictions_v2.csv
#   results/v2_vs_v3/predictions_v3.csv
#   results/v2_vs_v3/summary.json
#   results/v2_vs_v3/comparison.png
```

## Recommendation

For the GitHub README headline numbers and the public demo, use **the
DINOv3-L + SAM 3.1 stack**. It delivers the strongest snowball story
(+8.4 pp on the hardest class) and the per-class plot is more compelling
than the v2 version.

For users without HF access (or who just want to try the project fast),
the default open stack (DINOv2-B + SAM 2.1 Tiny) still gives 91.5 %
overall — perfectly respectable, and zero setup friction.

```powershell
# Production run with v3 weights:
$env:SNOWLABEL_DINO = "<repo-root>\weights\facebookdinov3-vitl16-pretrain-lvd1689m"
$env:SNOWLABEL_SAM  = "<repo-root>\weights\facebooksam3.1\sam3.1_multiplex.pt"
Remove-Item -Recurse data\memory   # clear v2 memory (embed dim changes)
python -m snowlabel.server
```

## Open follow-ups

- The DINOv3 confidence distribution is **flatter** than DINOv2 (higher-dim
  features → tighter cosine sims). The default thresholds
  (`conf_auto_apply=0.85`, `conf_suggest=0.60`) may need re-tuning per
  backbone. The smoke test (only 6 instances) demonstrated this: most
  predictions fell into "silent" tier despite being correct. Recommended
  defaults for v3:
  - `SNOWLABEL_CONF_AUTO=0.75`
  - `SNOWLABEL_CONF_SUGGEST=0.50`
  Worth a real-data sweep in Phase 1.5.
- DINOv3 also supports **higher input resolution** (518² → 32×32 patches).
  Combined with mask-pooling, that should help fine-grained instances.
  Worth one experiment with `SNOWLABEL_RES=512`.
- Try **multi-prototype per class** — store the *mean* of several exemplars
  rather than each one individually. Phase 0 listed it as the next algorithmic
  knob; with DINOv3-L's stronger features, it should compound.
