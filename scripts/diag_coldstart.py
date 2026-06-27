"""Cold-start tier behavior check (current feature-group pipeline).

Feeds same-class (cat) GT masks one by one into a fresh memory and prints how
the predicted tier evolves with per-class support. With the cold-start gate,
early predictions (support < min_support_auto) should NOT reach 'auto' even
when the rerank blend score is high on a tiny memory — they ramp silent ->
suggest -> auto as evidence accumulates (textbook snowball).
"""
import shutil
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO

from snowlabel.config import get_config
from snowlabel.memory import SnowballMemory
from snowlabel.predictor import SnowPredictor

ROOT = Path(__file__).resolve().parents[1]
cfg = get_config()
print(f"gate: min_support_auto={cfg.min_support_auto} min_support_suggest={cfg.min_support_suggest}")
print(f"thresholds: auto={cfg.conf_auto_apply} suggest={cfg.conf_suggest}")
print(f"dino={cfg.dino_variant.split(chr(92))[-1]} mode={cfg.predict_mode}\n")

coco = COCO(str(ROOT / "data" / "coco" / "annotations" / "instances_val2017_snowlabel.json"))
cid = coco.getCatIds(catNms=["cat"])[0]
img_ids = coco.getImgIds(catIds=[cid])[:8]

pred = SnowPredictor()
mem_dir = ROOT / "data" / "memory_coldstart"
if mem_dir.exists():
    shutil.rmtree(mem_dir)
mem = SnowballMemory(mem_dir, dim=pred.dino.embed_dim)
pred._memory = mem  # isolate to clean numpy memory

print(f"{'step':>4} {'support_before':>14} {'conf':>6} {'tier':>9}  pred")
print("-" * 48)
for step, iid in enumerate(img_ids):
    info = coco.loadImgs(iid)[0]
    img = Image.open(ROOT / "data" / "coco" / "val2017" / info["file_name"]).convert("RGB")
    anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=[cid]))
            if not a.get("iscrowd", 0)]
    if not anns:
        continue
    ann = max(anns, key=lambda a: a["area"])
    mask = coco.annToMask(ann).astype(bool)

    mf = pred.feat.features(img, mask)
    label, conf, tier, _, _cw = pred._predict_from_features(mf)
    support = pred._class_support(label)
    print(f"{step:4d} {support:14d} {conf:6.2f} {tier:>9}  {label}")

    pred.remember(img, mask, "cat", image_uri=str(iid), verified=True)

print("\nExpected: step0 silent (empty) -> suggest while support<3 -> auto once support>=3")
