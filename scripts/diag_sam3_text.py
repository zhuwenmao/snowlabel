"""Verify SAM3 text-prompt concept segmentation works with the multiplex ckpt.

Loads Sam3Model (image detector; tracker.* keys ignored) + Sam3Processor from
the local weights dir, runs a few text prompts on a COCO image that contains a
cat, and reports detected instances / scores / VRAM / latency.

Success = text='cat' finds >=1 instance with decent score on a cat image, and
off-target prompts find fewer/none -> proves SAM3 is wired correctly and the
concept semantics are right.
"""
import time
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from transformers import Sam3Model, Sam3Processor

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "weights" / "sam3_hf"


def vram() -> str:
    return (f"{torch.cuda.memory_allocated()/1e9:.2f}GB alloc / "
            f"{torch.cuda.memory_reserved()/1e9:.2f}GB resv")


print("== load SAM3 (Sam3Model, tracker ignored) ==")
t = time.time()
model = Sam3Model.from_pretrained(str(W)).to("cuda").eval()
proc = Sam3Processor.from_pretrained(str(W))
print(f"  loaded {time.time()-t:.1f}s  vram={vram()}")

# Find a COCO image that actually contains a cat.
coco = COCO(str(ROOT / "data" / "coco" / "annotations" / "instances_val2017_snowlabel.json"))
cat_id = coco.getCatIds(catNms=["cat"])[0]
img_ids = coco.getImgIds(catIds=[cat_id])
info = coco.loadImgs(img_ids[0])[0]
img_path = ROOT / "data" / "coco" / "val2017" / info["file_name"]
image = Image.open(img_path).convert("RGB")
n_gt_cat = len(coco.getAnnIds(imgIds=[img_ids[0]], catIds=[cat_id]))
print(f"  image={info['file_name']} size={image.size} gt_cats={n_gt_cat}")

print("\n== text prompts ==")
for text in ["cat", "dog", "car", "person"]:
    inputs = proc(images=image, text=text, return_tensors="pt").to("cuda")
    t = time.time()
    with torch.inference_mode():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=0.3, mask_threshold=0.5,
        target_sizes=[(image.height, image.width)],
    )[0]
    scores = [round(float(s), 2) for s in res["scores"][:8]]
    mshape = tuple(res["masks"].shape) if len(res["masks"]) else "()"
    print(f"  text='{text:7s}'  instances={len(res['scores'])}  "
          f"scores={scores}  masks={mshape}  {time.time()-t:.1f}s  vram={vram()}")
