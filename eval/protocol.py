"""Benchmark protocol — Phase A snowball + Phase B open-set probe.

Phase A replays the real labeling loop headlessly: click a point INSIDE the
GT mask (nearest its centroid — measures the pipeline, not bbox-center luck)
-> SAM mask -> predict (label/tier) -> record vs GT (incl. IoU of the SAM
mask against the COCO GT mask, so mask failures aren't blamed on retrieval)
-> teach the SAM mask with the GT label, exactly what a user commit stores.

Phase B probes classes the memory has NEVER seen, drawn from the FULL COCO
val2017 annotations restricted to images that exist on disk, WITHOUT
teaching. Any auto/suggest on them is an open-set false accept.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image


def load_coco(path: Path):
    from pycocotools.coco import COCO
    return COCO(str(path))


# ------------------------------------------------------------- sequences
def build_seen_sequence(coco, classes: list[str], n_per: int, min_area: int,
                        img_dir: Path | None = None) -> list[dict]:
    """Per class: biggest qualifying instance of the first n_per images,
    then interleave classes (c1,c2,...,c1,c2,...) so every class snowballs
    in parallel. Deterministic (COCO ids are sorted). Pass ``img_dir`` when
    the annotation file covers more images than exist locally (custom class
    suites use the full val2017 json, of which only ~800 images are on disk)."""
    per_class: dict[str, list[dict]] = {}
    for c in classes:
        cid = coco.getCatIds(catNms=[c])[0]
        picked: list[dict] = []
        for iid in coco.getImgIds(catIds=[cid]):
            if img_dir is not None and not (
                    img_dir / coco.loadImgs(iid)[0]["file_name"]).is_file():
                continue
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=[cid]))
                    if not a.get("iscrowd", 0) and a["area"] >= min_area]
            if anns:
                picked.append({"iid": iid, "ann": max(anns, key=lambda a: a["area"]),
                               "cls": c})
            if len(picked) >= n_per:
                break
        per_class[c] = picked
    seq: list[dict] = []
    for i in range(n_per):
        for c in classes:
            if i < len(per_class[c]):
                seq.append(per_class[c][i])
    return seq


def build_openset_sequence(coco_full, seen_classes: list[str], img_dir: Path,
                           n: int, min_area: int, seed: int,
                           per_class_cap: int = 4,
                           exclude_iids: set | None = None) -> list[dict]:
    """Instances of UNSEEN categories on locally available images, spread
    across as many distinct categories as possible (round-robin).

    ``exclude_iids``: images already used for phase-A teaching. A probe on a
    taught image retrieves a near-identical same-image crop (blend ~1.0) —
    that is memory leakage, not an open-set verdict."""
    seen = set(seen_classes)
    excl = exclude_iids or set()
    rng = np.random.default_rng(seed)
    buckets: list[list[dict]] = []
    for cat in coco_full.loadCats(coco_full.getCatIds()):
        if cat["name"] in seen:
            continue
        items: list[dict] = []
        for iid in coco_full.getImgIds(catIds=[cat["id"]]):
            if iid in excl:
                continue
            info = coco_full.loadImgs(iid)[0]
            if not (img_dir / info["file_name"]).is_file():
                continue
            anns = [a for a in coco_full.loadAnns(
                        coco_full.getAnnIds(imgIds=iid, catIds=[cat["id"]]))
                    if not a.get("iscrowd", 0) and a["area"] >= min_area]
            if anns:
                items.append({"iid": iid, "ann": max(anns, key=lambda a: a["area"]),
                              "cls": cat["name"]})
            if len(items) >= per_class_cap:
                break
        if items:
            buckets.append(items)
    order = rng.permutation(len(buckets))
    buckets = [buckets[i] for i in order]
    seq: list[dict] = []
    depth = 0
    while len(seq) < n:
        added = False
        for items in buckets:
            if depth < len(items):
                seq.append(items[depth])
                added = True
                if len(seq) >= n:
                    break
        if not added:
            break
        depth += 1
    return seq


# ------------------------------------------------------------- geometry helpers
def click_point(gt_mask: np.ndarray, ann: dict) -> tuple[int, int]:
    """A point guaranteed inside the GT mask: the mask pixel nearest the mask
    centroid (bbox centers fall on background for rings/people/donuts)."""
    ys, xs = np.nonzero(gt_mask)
    if xs.size == 0:
        bx, by, bw, bh = ann["bbox"]
        return int(bx + bw / 2), int(by + bh / 2)
    cy, cx = float(ys.mean()), float(xs.mean())
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(xs[i]), int(ys[i])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


# ------------------------------------------------------------- run loops
def _step_row(pred, coco, item: dict, img_dir: Path, phase: str, step: int):
    """One click -> (Prediction|None, PIL image, file, GT mask, record row)."""
    info = coco.loadImgs(item["iid"])[0]
    img = Image.open(img_dir / info["file_name"]).convert("RGB")
    gt_mask = coco.annToMask(item["ann"]).astype(bool)
    x, y = click_point(gt_mask, item["ann"])
    t0 = time.perf_counter()
    p = pred.predict_from_point(img, x, y)
    ms = (time.perf_counter() - t0) * 1000.0
    gt = item["cls"]
    base = {"phase": phase, "step": step, "file": info["file_name"], "gt": gt,
            "latency_ms": round(ms, 1), "mem": len(pred.memory), "taught": ""}
    if p is None:
        base.update({"pred": "-", "conf": 0.0, "margin": 0.0, "tier": "silent",
                     "correct": 0, "iou": 0.0, "fail": "no_mask",
                     "cand": "-", "cand_correct": 0, "cw": 0.0})
        return None, img, info["file_name"], gt_mask, base
    iou = mask_iou(p.mask, gt_mask)
    correct = int(p.label == gt) if p.label else 0
    tk = p.top_k or []
    margin = (float(tk[0][1]) - float(tk[1][1])) if len(tk) >= 2 else (
        float(tk[0][1]) if tk else 0.0)
    fail = ""
    if p.tier != "silent" and not correct:
        # Attribute the miss: bad SAM cut vs bad retrieval.
        fail = "mask" if iou < 0.5 else "retrieval"
    # cand = the rerank winner regardless of tier gating — lets the offline
    # threshold sweep (eval/sweep.py) re-simulate tiers at ANY threshold.
    cand = tk[0][0] if tk else "-"
    base.update({"pred": p.label or "-", "conf": round(float(p.confidence), 4),
                 "margin": round(margin, 4), "tier": p.tier, "correct": correct,
                 "iou": round(iou, 3), "fail": fail,
                 "cand": cand, "cand_correct": int(cand == gt),
                 "cw": round(float(p.cw), 4)})
    return p, img, info["file_name"], gt_mask, base


def run_snowball(pred, coco, seq: list[dict], img_dir: Path,
                 log=print) -> list[dict]:
    """Phase A: predict -> record -> teach, like a user labeling start-to-end.

    Teaching mirrors what a real user commits: the SAM mask when it's good,
    the FIXED (GT) mask when SAM's cut is bad (IoU < 0.5) — nobody commits a
    garbage mask, they repair it first. The repair is charged by the effort
    model via the ``taught`` / ``iou`` columns."""
    if seq:  # warm up models so step 0 latency isn't model-load time
        info = coco.loadImgs(seq[0]["iid"])[0]
        img = Image.open(img_dir / info["file_name"]).convert("RGB")
        pred.predict_from_point(img, img.width // 2, img.height // 2)
    rows: list[dict] = []
    for step, item in enumerate(seq):
        p, img, fname, gt_mask, row = _step_row(pred, coco, item, img_dir,
                                                "snowball", step)
        rows.append(row)
        if p is not None and row["iou"] >= 0.5:
            pred.remember(img, p.mask, item["cls"], image_uri=fname, verified=True)
            row["taught"] = "sam"
        else:
            pred.remember(img, gt_mask, item["cls"], image_uri=fname, verified=True)
            row["taught"] = "gt_fix"
        row["mem"] = len(pred.memory)
        if step % 10 == 0:
            log(f"  [A] {step:3d}/{len(seq)}  {row['gt']}->{row['pred']} "
                f"({row['tier']}, iou={row['iou']})  mem={row['mem']}")
    return rows


def run_openset(pred, coco_full, seq: list[dict], img_dir: Path,
                log=print) -> list[dict]:
    """Phase B: probe unseen classes; NO teaching. correct is always 0 by
    construction (the GT class is not in memory), so tier is the verdict."""
    rows: list[dict] = []
    for step, item in enumerate(seq):
        _, _, _, _, row = _step_row(pred, coco_full, item, img_dir, "openset", step)
        row["correct"] = 0
        row["fail"] = "" if row["tier"] == "silent" else "openset_accept"
        rows.append(row)
        if step % 10 == 0:
            log(f"  [B] {step:3d}/{len(seq)}  {row['gt']}->{row['pred']} ({row['tier']})")
    return rows
