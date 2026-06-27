"""Batch concept pre-annotation verification.

For each class the snowball memory would know, run SAM3 text-prompt across a
handful of COCO images and compare recalled instances vs ground truth. Proves
the 'learn class name from clicks -> SAM3 recalls every instance' loop works
across classes/images, not just one cherry-picked image.
"""
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from snowlabel.sam3_concept import Sam3ConceptSegmenter

ROOT = Path(__file__).resolve().parents[1]
coco = COCO(str(ROOT / "data" / "coco" / "annotations" / "instances_val2017_snowlabel.json"))
class_names = ["cat", "dog", "bird", "bottle", "car"]
N_PER_CLASS = 4

seg = Sam3ConceptSegmenter(score_threshold=0.3)
print("SAM3 weights available:", seg.available)
print(f"\n{'class':8s} {'imgs':>4s} {'GT':>4s} {'detected':>8s} {'avg_score':>9s} {'time':>6s}")
print("-" * 48)

for cname in class_names:
    cids = coco.getCatIds(catNms=[cname])
    if not cids:
        print(f"{cname:8s}  (not in dataset)")
        continue
    cid = cids[0]
    img_ids = coco.getImgIds(catIds=[cid])[:N_PER_CLASS]
    tot_gt = tot_det = 0
    scores: list[float] = []
    t = time.time()
    for iid in img_ids:
        info = coco.loadImgs(iid)[0]
        img = Image.open(ROOT / "data" / "coco" / "val2017" / info["file_name"]).convert("RGB")
        n_gt = len([a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=[cid]))
                    if not a.get("iscrowd", 0)])
        insts = seg.segment_concept(img, cname)
        tot_gt += n_gt
        tot_det += len(insts)
        scores += [i.score for i in insts]
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"{cname:8s} {len(img_ids):4d} {tot_gt:4d} {tot_det:8d} {avg:9.2f} {time.time()-t:5.1f}s")

print("\nDone. SAM3 concept batch-recall verified across classes.")
