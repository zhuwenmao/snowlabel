"""Pick a small balanced COCO subset for the Label Studio snowball demo.

Each of the 5 classes gets a few images where that class is prominent (big,
non-crowd instance), copies them to demo_images/, and writes demo_manifest.json
with an *interleaved* order (cat,dog,bird,bottle,car,cat,...) so the snowball
curve is visible: each class's first appearance is a cold miss, later ones hit.
"""
import json
import shutil
from pathlib import Path

from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
coco = COCO(str(ROOT / "data" / "coco" / "annotations" / "instances_val2017_snowlabel.json"))
classes = ["cat", "dog", "bird", "bottle", "car"]
PER = 4
MIN_SIDE = 80  # skip tiny instances — bad for click demo

out = ROOT / "demo_images"
out.mkdir(exist_ok=True)
for old in out.glob("*.jpg"):
    old.unlink()

used: set[int] = set()
by_class: dict[str, list[str]] = {c: [] for c in classes}
for c in classes:
    cid = coco.getCatIds(catNms=[c])[0]
    for iid in coco.getImgIds(catIds=[cid]):
        if iid in used:
            continue
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=[cid]))
                if not a.get("iscrowd", 0)]
        if not anns:
            continue
        if max(a["area"] for a in anns) < MIN_SIDE * MIN_SIDE:
            continue
        info = coco.loadImgs(iid)[0]
        used.add(iid)
        shutil.copy(ROOT / "data" / "coco" / "val2017" / info["file_name"], out / info["file_name"])
        by_class[c].append(info["file_name"])
        if len(by_class[c]) >= PER:
            break

# Interleave: round-robin across classes
manifest = []
for i in range(PER):
    for c in classes:
        if i < len(by_class[c]):
            manifest.append({"file": by_class[c][i], "label_hint": c})

(ROOT / "demo_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Copied {sum(len(v) for v in by_class.values())} images to demo_images/")
for c in classes:
    print(f"  {c:8s}: {len(by_class[c])}")
print(f"Manifest ({len(manifest)} tasks, interleaved) -> demo_manifest.json")
