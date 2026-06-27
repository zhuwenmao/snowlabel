"""Build a FINE-GRAINED demo set into demo_images/ from full COCO val2017.

Unlike prepare_demo_data.py (5 wide classes from the filtered json), this picks
a few visually-similar classes (default: car/truck/bus) so the labeling UI
exercises the feature group's shape/part discrimination, not just coarse
semantics. Each image features the target class as its largest instance; images
are not reused across classes; output is interleaved so every class snowballs
in parallel.

    python scripts/prepare_demo_fine.py                      # car,truck,bus  x12
    python scripts/prepare_demo_fine.py --classes cat,dog,horse --per 12
"""
import argparse
import json
import shutil
from pathlib import Path

from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "coco" / "val2017"
ANN = ROOT / "data" / "coco" / "annotations" / "instances_val2017.json"
OUT = ROOT / "demo_images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="car,truck,bus")
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--min-side", type=int, default=80, help="skip tiny instances")
    args = ap.parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    min_area = args.min_side * args.min_side

    coco = COCO(str(ANN))
    cat_id = {c: coco.getCatIds(catNms=[c])[0] for c in classes}
    suite_ids = set(cat_id.values())   # the other confusable classes in this suite
    used: set[int] = set()             # image ids already claimed by an earlier class
    picked: dict[str, list[dict]] = {c: [] for c in classes}

    for c in classes:
        cid = cat_id[c]
        for iid in coco.getImgIds(catIds=[cid]):
            if iid in used:
                continue
            info = coco.loadImgs(iid)[0]
            if not (IMG_DIR / info["file_name"]).is_file():
                continue
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid))
                    if not a.get("iscrowd", 0)]
            tgt = [a for a in anns if a["category_id"] == cid and a["area"] >= min_area]
            if not tgt:
                continue
            t_area = max(a["area"] for a in tgt)
            # target must be the single largest object overall AND clearly
            # dominate any OTHER suite class in the frame, so clicking it is
            # unambiguous (no comparably-large truck next to the car, etc.).
            if max(anns, key=lambda a: a["area"])["category_id"] != cid:
                continue
            other_veh = [a["area"] for a in anns
                         if a["category_id"] in suite_ids and a["category_id"] != cid]
            if other_veh and max(other_veh) > 0.5 * t_area:
                continue
            used.add(iid)
            picked[c].append({"iid": iid, "file": info["file_name"]})
            if len(picked[c]) >= args.per:
                break
        print(f"  {c}: {len(picked[c])} images")

    # interleave c1,c2,c3,c1,c2,c3,...
    order: list[dict] = []
    for i in range(args.per):
        for c in classes:
            if i < len(picked[c]):
                order.append({**picked[c][i], "cls": c})

    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()
    for item in order:
        shutil.copy(IMG_DIR / item["file"], OUT / item["file"])
    (OUT / "demo_manifest.json").write_text(
        json.dumps({"classes": classes, "per": args.per,
                    "order": order}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nwrote {len(order)} images to {OUT} (classes={classes}, interleaved)")
    print("labels to teach:", ", ".join(classes))


if __name__ == "__main__":
    main()
