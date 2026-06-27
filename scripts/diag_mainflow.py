"""Main-flow cold-start diagnostic.

Loads whatever models the env vars point at, runs one real COCO image through
each stage, and reports per-stage status + VRAM. Designed to surface exactly
which links of the SAM->DINO->Memory chain are actually wired up with the
production (DINOv3 / SAM3.1 / Milvus) configuration.
"""
import glob
import time
import traceback

import torch
from PIL import Image

from snowlabel.config import get_config


def vram() -> str:
    if torch.cuda.is_available():
        return (f"{torch.cuda.memory_allocated()/1e9:.2f}GB alloc / "
                f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
    return "cpu"


cfg = get_config()
print("== CONFIG ==")
print("dino   :", cfg.dino_variant)
print("sam    :", cfg.sam_variant)
print("res    :", cfg.input_resolution)
print("backend:", cfg.memory_backend)
print("thr    :", cfg.conf_auto_apply, "/", cfg.conf_suggest)
print("cuda   :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

imgs = sorted(glob.glob(str(cfg.project_root / "data" / "coco" / "val2017" / "*.jpg")))
print("coco   :", len(imgs), "images")
img = Image.open(imgs[0]).convert("RGB")
print("image  :", imgs[0].split("\\")[-1], img.size)

# ---------------- STEP 1: DINO ----------------
print("\n== STEP 1: DINO feature extractor ==")
dino = None
try:
    from snowlabel.models import DinoFeatureExtractor
    dino = DinoFeatureExtractor()
    t = time.time()
    tokens = dino.encode_image(img)
    print(f"  OK  dim={dino.embed_dim}  tokens={tuple(tokens.shape)}  "
          f"{time.time()-t:.1f}s  vram={vram()}")
except Exception as e:
    print("  FAILED:", type(e).__name__, e)
    traceback.print_exc()

# ---------------- STEP 2: SAM ----------------
print("\n== STEP 2: SAM segmenter ==")
sam_ok = False
try:
    from snowlabel.models import SamSegmenter
    sam = SamSegmenter()
    t = time.time()
    sms = sam.predict_all(img, max_masks=5)
    print(f"  OK  {len(sms)} masks  {time.time()-t:.1f}s  vram={vram()}")
    sam_ok = True
except Exception as e:
    print("  FAILED:", type(e).__name__, repr(e)[:400])

# ---------------- STEP 3: Memory backend ----------------
print("\n== STEP 3: Memory backend ==")
try:
    from snowlabel.memory import get_memory
    if dino is not None:
        mem = get_memory(cfg.memory_dir, dim=dino.embed_dim)
        print(f"  OK  type={type(mem).__name__}  size={len(mem)}  "
              f"backend={cfg.memory_backend}")
    else:
        print("  SKIP (DINO failed, no dim)")
except Exception as e:
    print("  FAILED:", type(e).__name__, repr(e)[:400])

print("\n== SUMMARY ==")
print("DINO :", "OK" if dino is not None else "FAIL")
print("SAM  :", "OK" if sam_ok else "FAIL")
