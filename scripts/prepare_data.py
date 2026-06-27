"""
Download a small, filtered subset of COCO val 2017 for SnowLabel Phase 0.

Why this exists:
- The full COCO val 2017 image archive is ~1 GB (5000 images, 80 classes).
- For Phase 0 we only need a few hundred images across a handful of classes.
- This script downloads only the annotations JSON + the specific images that
  contain the target classes, by URL.

Usage (PowerShell on Windows or bash on Linux/macOS):
    python scripts/prepare_data.py --classes cat dog car person bottle \
        --max-images 300 --out-dir data/coco

The script is idempotent: re-running it skips already-downloaded files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

COCO_ANN_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_VAL_IMG_URL_TPL = "http://images.cocodataset.org/val2017/{:012d}.jpg"


def download_file(url: str, dest: Path, desc: str | None = None) -> None:
    """Stream-download with a progress bar. Skip if dest already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc or dest.name,
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)


def ensure_annotations(out_dir: Path) -> Path:
    """Ensure instances_val2017.json exists locally, extracting from the zip."""
    ann_json = out_dir / "annotations" / "instances_val2017.json"
    if ann_json.exists():
        return ann_json

    zip_path = out_dir / "annotations_trainval2017.zip"
    print(f"[1/3] Downloading COCO annotations zip (~241 MB) -> {zip_path}")
    download_file(COCO_ANN_ZIP_URL, zip_path, desc="annotations.zip")

    print(f"[1/3] Extracting instances_val2017.json -> {ann_json}")
    with zipfile.ZipFile(zip_path) as zf:
        member = "annotations/instances_val2017.json"
        if member not in zf.namelist():
            raise RuntimeError(
                f"Expected '{member}' inside the COCO annotations zip"
            )
        ann_json.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(ann_json, "wb") as dst:
            shutil.copyfileobj(src, dst)

    print(f"[1/3] Removing zip to save space ({zip_path.stat().st_size / 1e6:.0f} MB)")
    zip_path.unlink()
    return ann_json


def filter_and_sample(
    ann_json: Path,
    class_names: list[str],
    max_images: int,
) -> tuple[dict, list[int]]:
    """Load annotations, filter to target classes, sample up to max_images.

    Returns:
        filtered_coco: a valid COCO-format dict containing only the kept
            images/annotations/categories.
        image_ids: list of image IDs to download.
    """
    print(f"[2/3] Loading {ann_json.name} (this is ~80 MB, takes a few seconds)")
    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    name_to_cat = {c["name"]: c for c in coco["categories"]}
    missing = [n for n in class_names if n not in name_to_cat]
    if missing:
        available = sorted(name_to_cat.keys())
        raise ValueError(
            f"Unknown COCO class name(s): {missing}\n"
            f"Available: {available}"
        )
    target_cat_ids = {name_to_cat[n]["id"] for n in class_names}
    print(f"[2/3] Target classes: {class_names}")
    print(f"[2/3] Target category IDs: {sorted(target_cat_ids)}")

    image_to_anns: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in target_cat_ids:
            image_to_anns[ann["image_id"]].append(ann)

    candidate_image_ids = sorted(image_to_anns.keys())
    print(
        f"[2/3] Images containing target classes: {len(candidate_image_ids)} "
        f"(will keep at most {max_images})"
    )

    image_ids = candidate_image_ids[:max_images]
    keep_set = set(image_ids)
    images_kept = [img for img in coco["images"] if img["id"] in keep_set]
    anns_kept = [
        ann for img_id in image_ids for ann in image_to_anns[img_id]
    ]
    cats_kept = [c for c in coco["categories"] if c["id"] in target_cat_ids]

    filtered = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": cats_kept,
        "images": images_kept,
        "annotations": anns_kept,
    }
    n_anns = len(anns_kept)
    print(
        f"[2/3] Kept {len(images_kept)} images, {n_anns} instance annotations, "
        f"{len(cats_kept)} categories"
    )
    return filtered, image_ids


def download_images(image_ids: list[int], out_dir: Path) -> int:
    """Download each image by URL. Skip already-existing files."""
    images_dir = out_dir / "val2017"
    images_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    skipped = 0
    print(f"[3/3] Downloading {len(image_ids)} images -> {images_dir}")
    for img_id in tqdm(image_ids, desc="images"):
        url = COCO_VAL_IMG_URL_TPL.format(img_id)
        dest = images_dir / f"{img_id:012d}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        try:
            download_file(url, dest)
            fetched += 1
        except Exception as e:
            print(f"  warn: failed to fetch {url}: {e}", file=sys.stderr)
    print(f"[3/3] Newly fetched: {fetched}, already cached: {skipped}")
    return fetched + skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["cat", "dog", "car", "person", "bottle"],
        help="COCO class names to include",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=300,
        help="Maximum number of images to keep",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/coco"),
        help="Output root directory",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    print(f"Output dir: {out_dir}")

    ann_json = ensure_annotations(out_dir)
    filtered, image_ids = filter_and_sample(
        ann_json, args.classes, args.max_images
    )

    filtered_path = out_dir / "annotations" / "instances_val2017_snowlabel.json"
    print(f"Writing filtered annotations -> {filtered_path}")
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f)

    n_total = download_images(image_ids, out_dir)
    print()
    print("=" * 60)
    print(f"Done. Subset ready at: {out_dir}")
    print(f"  images : {n_total}")
    print(f"  anns   : {filtered_path}")
    print(f"  classes: {args.classes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
