# Phase 0 Results — SnowLabel Algorithm Validation

**Date**: 2026-05-26
**Hardware**: Windows 11 + NVIDIA RTX 4060 Laptop GPU (8 GB), CUDA 12.2
**Stack**: Python 3.11, PyTorch 2.5.1+cu121, DINOv2 ViT-S/14 (via torch.hub)
**Run time**: 17 seconds for 458 instances on GPU

## Verdict

**PHASE 0 PASSES. Proceed to Phase 1.**

The core hypothesis — *"DINO mask-pooled features + k-NN retrieval over user labels produce a measurable lift in next-instance accuracy as memory grows"* — is **confirmed** on a class-balanced 5-class COCO subset.

## Headline numbers

| Metric | Value |
|--------|-------|
| Classes | car, bird, cat, dog, bottle (~100 instances each) |
| Total predictions | 458 |
| **Overall accuracy** | **91.5%** |
| Uniform random baseline | 20.0% |
| Majority-class baseline | 21.8% |
| **Lift over majority baseline** | **+69.7 percentage points** |
| Late-run rolling accuracy | 98.0% |
| High-confidence (≥0.85) accuracy | 97.6% (covers 73.6% of predictions) |

## The snowball curve

`results/balanced_5class/snowball_curve.png`

Textbook learning curve:
- Instance 0–5: **cold start, ~0–30%** (no memory)
- Instance 5–50: **rapid climb to ~90%**
- Instance 50+: **plateau at 90–98%**

This is exactly the product story: the *first* image is hard, but it gets fast quickly. By ~30 user-corrected labels, the model is reliable for most instances.

## Per-class breakdown

| Class | n | Accuracy | Snowball lift (early → late) |
|-------|---|----------|------------------------------|
| **dog** | 95 | 81.1% | **+17.2 pp** (72.3% → 89.6%) ← biggest snowball |
| **cat** | 97 | 91.8% | +8.4 pp (87.5% → 95.9%) |
| **bottle** | 100 | 97.0% | +6.0 pp (94.0% → 100%) |
| **car** | 100 | 98.0% | +4.0 pp (96.0% → 100%) |
| **bird** | 66 | 87.9% | +0 pp (already saturated) |

The hardest class (dog) shows the strongest learning — exactly the regime where the snowball pipeline matters most.

## Confidence calibration

`results/balanced_5class/snowball_confidence.png`

| Confidence threshold | Coverage | Accuracy |
|---------------------|----------|----------|
| ≥ 0.50 | 98.7% | 92.3% |
| ≥ 0.70 | 85.6% | 95.7% |
| **≥ 0.85** | **73.6%** | **97.6%** |
| ≥ 0.95 | 72.1% | 97.6% |

**This is the product-UX gold standard**: the model *knows when it's confident*. 73.6% of predictions arrive at ≥97.6% accuracy — those can be auto-applied with no review. The remaining 26.4% should pause for user confirmation.

Mean confidence when correct vs wrong (per class):
| Class | Conf-correct | Conf-wrong |
|-------|--------------|------------|
| car | 0.988 | 0.783 |
| bottle | 0.994 | 0.637 |
| bird | 0.918 | 0.778 |
| cat | 0.912 | 0.717 |
| dog | 0.828 | 0.644 |

Clear separation in every class — the confidence score has signal.

## Confusion patterns

```
        car  bird  cat  dog  bottle
car      98     0    0    0       2
bird      0    58    2    4       2
cat       3     1   89    4       0
dog       4     0   13   77       1
bottle    2     0    1    0      97
```

Main errors are **dog → cat (13 cases)** — visually expected for small furry pets at varying poses. This is the kind of error that gets fixed by:
1. Switching to a larger backbone (DINOv2-L, DINOv3) → more discriminative features
2. Higher input resolution (518² gives finer patches)
3. Multi-prototype per class (store several exemplars, average them)

All three are Phase 1 / Phase 2 levers.

## What this means for the project

### ✓ Things that strongly worked
1. **DINOv2 mask-pooled features carry class identity well** — even the small ViT-S/14 model gets 91.5% on a balanced 5-class problem with zero training, only retrieval.
2. **Cosine k-NN voting is a viable predictor** — no neural classifier or fine-tuning needed.
3. **The snowball is real** — accuracy climbs from cold-start to plateau in <50 labels.
4. **Confidence is well-calibrated** — usable for selective auto-apply UX.

### ⚠ Caveats / honest observations
1. **DINOv2 saturates quickly**. The plateau at ~95% suggests we're at the limit of what ViT-S/14 can discriminate. DINOv3 / bigger backbones will lift this ceiling.
2. **Bird performance is flat**: 87.9% but no improvement over the run. Birds vary a lot in shape/pose; small ViT struggles.
3. **One snowball cycle ≠ proof for arbitrary domains**. COCO classes are visually clean. Medical / satellite / niche domains will need re-validation.
4. **We used GT masks**. Real SAM 3 outputs will have noisy boundaries; mask-pool quality may dip slightly. Need to test in Phase 1.

### Recommended next steps

**Proceed to Phase 1 with the following adjustments:**

1. **Use DINOv3** instead of DINOv2 in the production backend. Validated approach is identical; DINOv3 is just stronger.
2. **Bump input resolution to 518²** (37×37 patch grid) for finer instance discrimination — especially helpful for dog/cat splits.
3. **Implement Milvus** properly. The numpy cosine-search in Phase 0 worked because n < 1000; production needs Milvus HNSW for n > 10k.
4. **Build the confidence UI now**, while the algorithm is fresh. The 3-tier (auto-apply / suggest / silent) maps directly to:
   - conf ≥ 0.85: green check, auto-applied
   - 0.6 ≤ conf < 0.85: yellow "Suggest: X", click to accept
   - conf < 0.6: no prediction, manual labeling
5. **Test on a hard domain** before claiming generality. Pick something like FGVC-Aircraft or a fine-grained dataset to stress-test.

## Reproducibility

```powershell
# Activate env (already set up at %USERPROFILE%\miniconda3\envs\snowlabel)
& "$env:USERPROFILE\miniconda3\envs\snowlabel\python.exe" --version
# Python 3.11.15

# Re-prepare balanced data
python scripts/prepare_data.py --classes cat dog bird bottle car --max-images 600

# Re-run notebook (set MPLBACKEND for headless)
$env:MPLBACKEND = "Agg"
python notebooks/phase0_validation.py

# Re-analyze
python scripts/analyze_results.py
```

Seed: `SEED=42` in the notebook. Per-class cap: `MAX_INSTANCES_PER_CLASS=100`.

## Files

- `notebooks/phase0_validation.py` — the experiment
- `scripts/prepare_data.py` — data fetch
- `scripts/analyze_results.py` — deep analysis
- `results/balanced_5class/snowball_curve.png` — **the headline chart**
- `results/balanced_5class/snowball_buckets.png` — bucketed view
- `results/balanced_5class/snowball_confidence.png` — calibration
- `results/balanced_5class/phase0_predictions.csv` — raw data
