"""Deterministic VisA protocol for progressive industrial annotation."""

from __future__ import annotations

import csv
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .protocol import click_point, mask_iou


DATASET_URL = (
    "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/"
    "VisA_20220922.tar"
)
DATASET_BYTES = 1_929_840_640
DATASET_LICENSE = "CC BY 4.0"

# Official mapping from amazon-science/spot-diff utils/id2class.py.
ID2CLASS: dict[str, dict[int, str]] = {
    "candle": {
        1: "Chunk Of Wax Missing",
        2: "Damaged Corner of Packaging",
        3: "Weird Candle Wick",
        4: "Different Colour Spot",
        5: "Extra Wax in Candle",
        6: "Foreign Particles on Candle",
        7: "Wax Melted Out of the Candle",
        8: "Other",
    },
    "capsules": {
        1: "Bubble",
        2: "Discolor",
        3: "Scratch",
        4: "Leak",
        5: "Misshape",
    },
    "fryum": {
        1: "Burnt",
        2: "Corner or Edge Breakage",
        3: "Different Colour Spot",
        4: "Fryum Stuck Together",
        5: "Middle Breakage",
        6: "Similar Colour Spot",
        7: "Small Scratches",
        8: "Other",
    },
    "cashew": {
        1: "Burnt",
        2: "Corner or Edge Breakage",
        3: "Different Colour Spot",
        4: "Middle Breakage",
        5: "Small Holes",
        6: "Small Scratches",
        7: "Stuck Together",
        8: "Same Colour Spot",
        9: "Other",
    },
    "chewinggum": {
        1: "Chunk of Gum Missing",
        2: "Corner Missing",
        3: "Scratches",
        4: "Similar Colour Spot",
        5: "Small Cracks",
        6: "Other",
    },
    "macaroni1": {
        1: "Chip Around Edge and Corner",
        2: "Different Colour Spot",
        3: "Middle Breakage",
        4: "Similar Colour Spot",
        5: "Small Cracks",
        6: "Small Scratches",
        7: "Other",
    },
    "macaroni2": {
        1: "Breakage Down the Middle",
        2: "Colour Spot Similar to Object",
        3: "Different Colour Spot",
        4: "Small Chip Around Edge",
        5: "Small Cracks",
        6: "Small Scratches",
        7: "Other",
    },
    "pcb1": {1: "Bent", 2: "Melt", 3: "Scratch", 4: "Missing"},
    "pcb2": {1: "Bent", 2: "Melt", 3: "Scratch", 4: "Missing"},
    "pcb3": {1: "Bent", 2: "Melt", 3: "Scratch", 4: "Missing"},
    "pcb4": {
        1: "Burnt",
        2: "Scratch",
        3: "Missing",
        4: "Damage",
        5: "Extra",
        6: "Wrong Place",
        7: "Dirt",
    },
    "pipe_fryum": {
        1: "Burnt",
        2: "Corner and Edge Breakage",
        3: "Different Colour Spot",
        4: "Middle Breakage",
        5: "Similar Colour Spot",
        6: "Small Scratches",
        7: "Stuck Together",
        8: "Small Cracks",
        9: "Other",
    },
}


@dataclass(frozen=True)
class VisaInstance:
    object_name: str
    image: str
    mask: str
    class_id: int
    class_name: str
    label: str
    area: int
    bbox: tuple[int, int, int, int]


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask.astype(bool)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def _instance_from_id(
    root: Path, object_name: str, image: str, mask_path: str, class_id: int
) -> VisaInstance:
    values = np.asarray(Image.open(root / mask_path).convert("L"))
    component = largest_component(values == class_id)
    ys, xs = np.nonzero(component)
    if not xs.size:
        raise ValueError(f"empty class id {class_id} in {mask_path}")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    class_name = ID2CLASS[object_name][class_id]
    return VisaInstance(
        object_name=object_name,
        image=image,
        mask=mask_path,
        class_id=class_id,
        class_name=class_name,
        label=f"{object_name}/{class_name}",
        area=int(component.sum()),
        bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
    )


def load_instances(root: Path, min_area: int) -> list[VisaInstance]:
    missing = sorted(name for name in ID2CLASS if not (root / name).is_dir())
    if missing:
        raise FileNotFoundError(f"VisA subsets missing from {root}: {missing}")
    instances: list[VisaInstance] = []
    for object_name in sorted(ID2CLASS):
        csv_path = root / object_name / "image_anno.csv"
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row["mask"]:
                    continue
                values = np.asarray(Image.open(root / row["mask"]).convert("L"))
                for class_id in sorted(int(v) for v in np.unique(values) if v):
                    if class_id not in ID2CLASS[object_name]:
                        raise ValueError(
                            f"unknown id {class_id} for {object_name}: {row['mask']}"
                        )
                    instance = _instance_from_id(
                        root, object_name, row["image"], row["mask"], class_id
                    )
                    if instance.area >= min_area:
                        instances.append(instance)
    return instances


def _interleave_unique_images(
    buckets: dict[str, list[VisaInstance]], n_per: int
) -> tuple[list[VisaInstance], list[str]]:
    labels = sorted(label for label, items in buckets.items() if len(items) >= n_per)
    while labels:
        selected: dict[str, list[VisaInstance]] = defaultdict(list)
        cursors = {label: 0 for label in labels}
        used: set[str] = set()
        sequence: list[VisaInstance] = []
        for _ in range(n_per):
            for label in labels:
                items = buckets[label]
                cursor = cursors[label]
                while cursor < len(items) and items[cursor].image in used:
                    cursor += 1
                cursors[label] = cursor + 1
                if cursor < len(items):
                    item = items[cursor]
                    selected[label].append(item)
                    sequence.append(item)
                    used.add(item.image)
        complete = [label for label in labels if len(selected[label]) == n_per]
        if complete == labels:
            return sequence, labels
        labels = complete
    return [], []


def build_protocol(
    instances: list[VisaInstance], n_per: int = 6, openset_per_object: int = 4
) -> tuple[list[VisaInstance], list[VisaInstance], dict]:
    counts = Counter(item.label for item in instances)
    by_object: dict[str, list[str]] = defaultdict(list)
    for label in counts:
        by_object[label.split("/", 1)[0]].append(label)
    holdout = {
        object_name: max(labels, key=lambda label: (counts[label], label))
        for object_name, labels in sorted(by_object.items())
    }
    holdout_labels = set(holdout.values())
    holdout_images = {
        item.image for item in instances if item.label in holdout_labels
    }

    buckets: dict[str, list[VisaInstance]] = defaultdict(list)
    for item in sorted(instances, key=lambda x: (x.label, x.image)):
        if item.label not in holdout_labels and item.image not in holdout_images:
            buckets[item.label].append(item)
    seen, seen_labels = _interleave_unique_images(buckets, n_per)
    used_seen = {item.image for item in seen}

    openset: list[VisaInstance] = []
    for object_name, label in sorted(holdout.items()):
        candidates = sorted(
            (
                item
                for item in instances
                if item.label == label and item.image not in used_seen
            ),
            key=lambda item: item.image,
        )
        openset.extend(candidates[:openset_per_object])

    manifest = {
        "dataset": {
            "name": "VisA_20220922",
            "source": DATASET_URL,
            "archive_bytes": DATASET_BYTES,
            "license": DATASET_LICENSE,
            "objects": sorted(ID2CLASS),
        },
        "protocol": {
            "n_per_seen_class": n_per,
            "openset_per_object": openset_per_object,
            "holdout_rule": "most frequent eligible defect class per object; ties by label",
            "image_isolation": True,
            "largest_component_per_class_per_image": True,
        },
        "inventory": {
            "instances": len(instances),
            "classes": len(counts),
            "class_counts": dict(sorted(counts.items())),
        },
        "seen": {
            "classes": seen_labels,
            "samples": len(seen),
            "items": [asdict(item) for item in seen],
        },
        "openset": {
            "holdout_by_object": holdout,
            "samples": len(openset),
            "items": [asdict(item) for item in openset],
        },
    }
    return seen, openset, manifest


def load_target(root: Path, item: VisaInstance) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(root / item.image).convert("RGB")
    values = np.asarray(Image.open(root / item.mask).convert("L"))
    return image, largest_component(values == item.class_id)


def predict_step(pred, root: Path, item: VisaInstance, phase: str, step: int):
    image, gt_mask = load_target(root, item)
    x, y = click_point(gt_mask, {"bbox": item.bbox})
    started = time.perf_counter()
    prediction = pred.predict_from_point(image, x, y)
    latency_ms = (time.perf_counter() - started) * 1000
    row = {
        "phase": phase,
        "step": step,
        "object": item.object_name,
        "file": item.image,
        "gt": item.label,
        "pred": "-",
        "conf": 0.0,
        "margin": 0.0,
        "cw": 0.0,
        "tier": "silent",
        "correct": 0,
        "iou": 0.0,
        "fail": "no_mask",
        "cand": "-",
        "cand_correct": 0,
        "taught": "",
        "latency_ms": round(latency_ms, 1),
        "mem": len(pred.memory),
        "area": item.area,
    }
    if prediction is None:
        return None, image, gt_mask, row

    iou = mask_iou(prediction.mask, gt_mask)
    top_k = prediction.top_k or []
    margin = (
        float(top_k[0][1]) - float(top_k[1][1])
        if len(top_k) >= 2
        else (float(top_k[0][1]) if top_k else 0.0)
    )
    candidate = top_k[0][0] if top_k else "-"
    correct = int(prediction.label == item.label) if prediction.label else 0
    fail = ""
    if prediction.tier != "silent" and not correct:
        fail = "mask" if iou < 0.5 else "retrieval"
    row.update(
        {
            "pred": prediction.label or "-",
            "conf": round(float(prediction.confidence), 4),
            "margin": round(margin, 4),
            "cw": round(float(prediction.cw), 4),
            "tier": prediction.tier,
            "correct": correct,
            "iou": round(iou, 3),
            "fail": fail,
            "cand": candidate,
            "cand_correct": int(candidate == item.label),
        }
    )
    return prediction, image, gt_mask, row
