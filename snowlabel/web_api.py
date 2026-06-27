"""Web-native API for the SnowLabel canvas labeling tool (``labeler.html``).

The Label Studio protocol routes in :mod:`snowlabel.server` wrap every
prediction in LS's RLE-brush format. These endpoints instead speak a plain
JSON + PNG contract that a ``<canvas>`` frontend can consume directly, while
reusing the *same* global :class:`SnowPredictor` — so a label committed here
grows the exact same snowball memory as a click coming from Label Studio.

Endpoints (all under ``/api``):
    GET  /api/dataset          — list images in the configured folder + labeled flag
    GET  /api/image/{id}       — raw image bytes (FileResponse)
    POST /api/segment          — point/box prompt -> mask PNG + predicted label
    POST /api/commit           — store (image, mask, label) into the snowball
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

try:
    import cv2  # ships with ultralytics
except Exception:  # pragma: no cover
    cv2 = None

from .config import get_config
from .predictor import SnowPredictor

log = logging.getLogger("snowlabel.web_api")

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
_MAX_UPLOAD_BYTES = 12 * 1024 * 1024   # 12 MB cap on visitor uploads
_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _user_id(request: Request) -> str:
    """Resolve the calling visitor's isolation id.

    Single-tenant (default): always "" — the X-Snow-User header is ignored, so
    the local car library behaves exactly as before. Multi-tenant public demo
    (SNOWLABEL_MULTIUSER=1): the validated header value scopes both the visitor's
    uploaded images and their memory anchors. Invalid / missing -> "".
    """
    if not get_config().multi_user:
        return ""
    uid = (request.headers.get("x-snow-user") or "").strip()
    if not uid:
        # 浏览器原生请求(<img src>、FileResponse 等)带不了自定义头,但会自动带 cookie。
        # 缩略图/画布走 /api/image,正是靠这个 cookie 兜底才能按本用户目录取到图。
        uid = (request.cookies.get("snow_user") or "").strip()
    return uid if _USER_RE.match(uid) else ""


def mask_to_contour(mask, eps_frac: float = 0.004) -> list[list[int]] | None:
    """Largest external contour as [[x,y],...] in image-pixel coords, simplified
    with approxPolyDP. The frontend strokes it as a thin outline (expanded ~10%
    outward) instead of a filled overlay. None -> frontend falls back to bbox."""
    if cv2 is None:
        return None
    m = (np.asarray(mask) > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
    pts = [[int(p[0][0]), int(p[0][1])] for p in c]
    return pts if len(pts) >= 3 else None


# ----------------------------------------------------------------- mask <-> png
def mask_to_png_b64(mask: np.ndarray) -> str:
    """Bool / 0-1 mask -> base64 single-channel PNG (255 = foreground)."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(m, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_b64_to_mask(s: str) -> np.ndarray:
    """base64 PNG (optionally a ``data:`` URL) -> bool mask."""
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    im = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(im) > 127


# ----------------------------------------------------------------- request bodies
class SegmentRequest(BaseModel):
    image_id: str
    type: str = "point"                       # "point" | "box"
    x: int | None = None                      # single positive click (legacy)
    y: int | None = None
    points: list[list[int]] | None = None     # [[x,y], ...] for multi-point
    labels: list[int] | None = None           # 1 = positive, 0 = negative
    bbox: list[int] | None = None             # [x0, y0, x1, y1]


class CommitRequest(BaseModel):
    image_id: str
    mask_png: str
    label: str
    # telemetry only (optional): what the model had predicted for this mask
    pred_label: str | None = None
    tier: str | None = None
    confidence: float | None = None
    source: str | None = None


class SegmentAllRequest(BaseModel):
    image_id: str
    max_masks: int = 20


class CommitItem(BaseModel):
    mask_png: str
    label: str
    # telemetry only (optional)
    pred_label: str | None = None
    tier: str | None = None
    confidence: float | None = None
    source: str | None = None


class AcceptedMeta(BaseModel):
    """A mask whose prediction the user accepted as-is — NOT stored in memory
    (it would be redundant), but logged so the real acceptance rate is known."""
    label: str | None = None
    pred_label: str | None = None
    tier: str | None = None
    confidence: float | None = None
    source: str | None = None


class CommitBatchRequest(BaseModel):
    image_id: str
    items: list[CommitItem]
    accepted: list[AcceptedMeta] = []


# ----------------------------------------------------------------- telemetry
def _telemetry(event: dict) -> None:
    """Append one labeling event to data/telemetry/events.jsonl. Telemetry
    must never break labeling — failures are logged and swallowed."""
    try:
        tdir = get_config().project_root / "data" / "telemetry"
        tdir.mkdir(parents=True, exist_ok=True)
        event["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(tdir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        log.exception("telemetry write failed")


def _commit_event(image_id: str, label: str, pred_label, tier, confidence,
                  source) -> dict:
    return {
        "type": "commit",
        "image_id": image_id,
        "label": label,
        "pred_label": pred_label,
        "tier": tier,
        "confidence": confidence,
        "source": source,
        # corrected = the model HAD a prediction and the user overrode it
        "corrected": bool(pred_label) and label != pred_label,
    }


# ----------------------------------------------------------------- router factory
def build_router(predictor: SnowPredictor) -> APIRouter:
    """Build the ``/api`` router bound to an existing predictor.

    Using a factory (instead of a module-level router) lets ``server.py`` share
    its single heavy :class:`SnowPredictor` with these routes without a circular
    import (server imports this module; this module never imports server).
    """
    router = APIRouter(prefix="/api", tags=["webtool"])

    def _dataset_dir() -> Path:
        return get_config().dataset_dir

    def _user_dir(user_id: str) -> Path:
        """Image folder for a visitor. Empty user_id -> the shared dataset dir
        (local single-tenant mode); a real user_id -> their private upload
        subfolder (public multi-tenant demo)."""
        base = _dataset_dir()
        return base / user_id if user_id else base

    def _list_images(user_id: str) -> list[Path]:
        d = _user_dir(user_id)
        if not d.is_dir():
            return []
        return sorted(
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXT
        )

    def _resolve_image(image_id: str, user_id: str) -> Path:
        # image_id is a bare filename — reject anything that could escape the dir.
        if "/" in image_id or "\\" in image_id or ".." in image_id:
            raise HTTPException(400, "invalid image_id")
        p = _user_dir(user_id) / image_id
        if not p.is_file():
            raise HTTPException(404, f"image not found: {image_id}")
        return p

    def _labeled_counts(user_id: str) -> Counter:
        """filename -> how many committed memory entries reference it (scoped to
        this visitor when multi-tenant)."""
        c: Counter = Counter()
        for e in predictor.memory.iter_entries():
            if user_id and getattr(e, "user_id", "") != user_id:
                continue
            uri = getattr(e, "image_uri", "") or ""
            if uri:
                c[Path(uri).name] += 1
        return c

    @router.get("/dataset")
    def dataset(request: Request, limit: int = 500, offset: int = 0) -> dict:
        user_id = _user_id(request)
        d = _user_dir(user_id)
        files = _list_images(user_id)
        counts = _labeled_counts(user_id)
        labeled_total = sum(1 for f in files if f.name in counts)
        window = files[offset:offset + limit]
        return {
            "dir": str(d),
            "total": len(files),
            "labeled": labeled_total,
            "limit": limit,
            "offset": offset,
            "images": [
                {
                    "id": f.name,
                    "name": f.name,
                    "url": f"/api/image/{f.name}",
                    "labeled": f.name in counts,
                    "n_labels": int(counts.get(f.name, 0)),
                }
                for f in window
            ],
        }

    @router.get("/image/{image_id}")
    def image(image_id: str, request: Request):
        return FileResponse(_resolve_image(image_id, _user_id(request)))

    @router.post("/upload")
    async def upload(request: Request, file: UploadFile = File(...)) -> dict:
        """Accept a visitor-uploaded image into their (per-user) folder so they
        can label their own pictures. Validates type/size and re-checks it's a
        real image before writing under a random filename."""
        user_id = _user_id(request)
        ext = Path(file.filename or "").suffix.lower()
        if ext not in _IMG_EXT:
            raise HTTPException(400, f"unsupported image type: {ext or '?'}")
        data = await file.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image too large (max 12 MB)")
        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            raise HTTPException(400, "not a valid image")
        d = _user_dir(user_id)
        d.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{ext}"
        (d / name).write_bytes(data)
        _register_cache(d, name, data, ext, user_id)   # 三库登记 + 后台预处理
        return {"ok": True, "id": name, "name": name, "url": f"/api/image/{name}"}

    def _register_cache(d: Path, name: str, data: bytes, ext: str,
                        user_id: str) -> None:
        """三库缓存:登记图片(MinIO+MySQL)并排队后台 SAM 预处理。upload 与
        load_samples 共用,失败只告警不影响标注。"""
        if predictor.cache is None:
            return
        try:
            from .imgcache.cache import image_sha256
            from .imgcache.prefetch import get_prefetcher
            im = Image.open(io.BytesIO(data))
            predictor.cache.put_image(image_sha256(data), data, ext=ext,
                                      orig_name=name, width=im.width,
                                      height=im.height)
            get_prefetcher(predictor).enqueue(str(d / name), user_id)
        except Exception as e:
            log.warning("imgcache 登记/入队失败: %s", e)

    @router.post("/load_samples")
    def load_samples(request: Request) -> dict:
        """一键把内置示例数据集 cub3_demo(3 种鸟:赫氏鸥/裏海燕鸥/仙人掌鹪鹩)
        拷进访客自己的目录,让人零门槛上手 + 顺势演示"越标越准"。已有图就不再重复灌。"""
        user_id = _user_id(request)
        src_root = (get_config().project_root / "data" / "datasets"
                    / "cub3_demo" / "images")
        if not src_root.is_dir():
            raise HTTPException(404, "示例数据集 cub3_demo 不存在")
        d = _user_dir(user_id)
        d.mkdir(parents=True, exist_ok=True)
        existing = len(_list_images(user_id))
        per_class = 5
        copied = 0
        for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            imgs = sorted(p for p in cls_dir.iterdir()
                          if p.suffix.lower() in _IMG_EXT)[:per_class]
            for src in imgs:
                data = src.read_bytes()
                if len(data) > _MAX_UPLOAD_BYTES:
                    continue
                ext = src.suffix.lower()
                name = f"{uuid.uuid4().hex}{ext}"
                (d / name).write_bytes(data)
                copied += 1
                _register_cache(d, name, data, ext, user_id)
        return {"ok": True, "copied": copied, "total": existing + copied}

    @router.post("/segment")
    def segment(req: SegmentRequest, request: Request) -> dict:
        user_id = _user_id(request)
        path = _resolve_image(req.image_id, user_id)
        img = Image.open(path).convert("RGB")
        if req.type == "point":
            if req.points:
                pts = [[int(p[0]), int(p[1])] for p in req.points]
                labs = req.labels if req.labels is not None else [1] * len(pts)
                if len(labs) != len(pts):
                    raise HTTPException(400, "points and labels length mismatch")
                pred = predictor.predict_from_points(img, pts, labs, user_id=user_id)
            elif req.x is not None and req.y is not None:
                pred = predictor.predict_from_point(img, int(req.x), int(req.y),
                                                    user_id=user_id)
            else:
                raise HTTPException(400, "point prompt needs points[] or x,y")
        elif req.type == "box":
            if not req.bbox or len(req.bbox) != 4:
                raise HTTPException(400, "box prompt needs bbox [x0,y0,x1,y1]")
            pred = predictor.predict_from_box(img, tuple(int(v) for v in req.bbox),
                                              user_id=user_id)
        else:
            raise HTTPException(400, f"unknown prompt type: {req.type}")

        if pred is None:
            return {"ok": False}
        return {
            "ok": True,
            "mask_png": mask_to_png_b64(pred.mask),
            "contour": mask_to_contour(pred.mask),
            "bbox": list(pred.bbox) if pred.bbox else None,
            "label": pred.label,
            "confidence": round(float(pred.confidence), 4),
            "tier": pred.tier,
            "top_k": [[lbl, round(float(s), 4)] for lbl, s in pred.top_k],
        }

    @router.post("/commit")
    def commit(req: CommitRequest, request: Request) -> dict:
        user_id = _user_id(request)
        path = _resolve_image(req.image_id, user_id)
        img = Image.open(path).convert("RGB")
        mask = png_b64_to_mask(req.mask_png)
        # The frontend sends the mask at natural image size; resize defensively
        # in case of any drift (e.g. future edited masks at canvas resolution).
        if mask.shape[:2] != (img.height, img.width):
            mask = np.asarray(
                Image.fromarray((mask.astype(np.uint8) * 255)).resize(
                    (img.width, img.height), Image.NEAREST)
            ) > 127
        if int(mask.sum()) == 0:
            raise HTTPException(422, "empty mask — nothing to commit")
        label = req.label.strip()
        if not label:
            raise HTTPException(400, "label is required")
        entry_id = predictor.remember(
            img, mask, label, image_uri=str(path), verified=True, user_id=user_id)
        _telemetry(_commit_event(req.image_id, label, req.pred_label,
                                 req.tier, req.confidence, req.source))
        return {"entry_id": int(entry_id), "memory_size": len(predictor.memory)}

    @router.post("/segment_all")
    def segment_all(req: SegmentAllRequest, request: Request) -> dict:
        """Everything mode: SAM proposes every object mask; the snowball memory
        tags each with a predicted label/tier. Returns all of them at once."""
        user_id = _user_id(request)
        path = _resolve_image(req.image_id, user_id)
        img_bytes = path.read_bytes()
        img = Image.open(path).convert("RGB")
        preds, from_cache = predictor.segment_all_cached(
            img_bytes, img, max_masks=int(req.max_masks), user_id=user_id,
            ext=(path.suffix or ".jpg"), orig_name=path.name)
        return {
            "ok": True,
            "from_cache": from_cache,
            "count": len(preds),
            "masks": [
                {
                    "idx": i,
                    "mask_png": mask_to_png_b64(p.mask),
                    "contour": mask_to_contour(p.mask),
                    "bbox": list(p.bbox) if p.bbox else None,
                    "label": p.label,
                    "confidence": round(float(p.confidence), 4),
                    "tier": p.tier,
                    "top_k": [[lbl, round(float(s), 4)] for lbl, s in p.top_k],
                }
                for i, p in enumerate(preds)
            ],
        }

    @router.post("/commit_batch")
    def commit_batch(req: CommitBatchRequest, request: Request) -> dict:
        """Bulk-commit many (mask, label) pairs from one image — encodes DINO
        once. Powers "segment everything → accept all confident"."""
        user_id = _user_id(request)
        path = _resolve_image(req.image_id, user_id)
        img = Image.open(path).convert("RGB")
        pairs: list[tuple] = []
        for it in req.items:
            mask = png_b64_to_mask(it.mask_png)
            if mask.shape[:2] != (img.height, img.width):
                mask = np.asarray(
                    Image.fromarray((mask.astype(np.uint8) * 255)).resize(
                        (img.width, img.height), Image.NEAREST)
                ) > 127
            lbl = it.label.strip()
            if int(mask.sum()) == 0 or not lbl:
                continue
            pairs.append((mask, lbl))
            _telemetry(_commit_event(req.image_id, lbl, it.pred_label,
                                     it.tier, it.confidence, it.source))
        for acc in req.accepted:
            _telemetry({"type": "accept", "image_id": req.image_id,
                        "label": acc.label, "pred_label": acc.pred_label,
                        "tier": acc.tier, "confidence": acc.confidence,
                        "source": acc.source})
        ids = predictor.remember_batch(img, pairs, image_uri=str(path),
                                       user_id=user_id)
        return {
            "committed": len(ids),
            "entry_ids": [int(i) for i in ids],
            "memory_size": len(predictor.memory),
        }

    @router.post("/reset")
    def reset(request: Request) -> dict:
        """『重新开始』:把当前访客自己的上传图 + 记忆锚点清空,回到全新状态,
        用于演示"从零越标越准"。守卫:仅多租户 Demo 下、且只删 user_id 命中本访客的
        条目——本地单租户(user_id="")会被拒,绝不会误删 owner 的真实记忆库。全局共享的
        imgcache 不动(它只是缓存,且别的访客在用)。"""
        user_id = _user_id(request)
        if not user_id:
            raise HTTPException(
                403, "『重新开始』仅在多租户公网 Demo 下可用(本地单租户不清库,以免误删真实记忆)")
        removed = 0
        for e in list(predictor.memory.iter_entries()):
            if getattr(e, "user_id", "") == user_id:
                if predictor.memory.forget(e.entry_id):
                    removed += 1
        deleted = 0
        for p in _list_images(user_id):
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
        return {"ok": True, "removed_memory": removed, "deleted_images": deleted,
                "memory_size": len(predictor.memory)}

    return router
