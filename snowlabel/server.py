"""FastAPI server that implements the Label Studio ML backend protocol.

We implement the protocol directly (rather than wrapping LabelStudioMLBase)
for three reasons:
  1. Full visibility into request / response shapes for debugging.
  2. We can add SnowLabel-specific routes (`/stats`, `/memory`, `/forget`)
     alongside the LS-required ones.
  3. FastAPI gives us auto-generated OpenAPI docs at `/docs` — useful demo.

LS protocol routes implemented:
    GET  /health        — required for LS to consider the backend healthy
    POST /setup         — LS calls this on project connection
    POST /predict       — the workhorse: returns predictions for tasks
    POST /webhook       — LS pushes annotation events (saves) here

SnowLabel-specific routes:
    GET  /stats         — memory size, classes seen, configuration
    GET  /memory        — list all stored entries (paginated)
    POST /forget        — drop an entry by id
    POST /seed          — bulk-load (image, mask, label) tuples
"""
from __future__ import annotations

import io
import logging
import os
from typing import Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel, Field

from .config import get_config
from .ls_protocol import (
    bbox_from_mask,
    decode_brush_rle,
    encode_mask_as_brush_rle,
    extract_interactive_prompt,
)
from .predictor import Prediction, SnowPredictor
from .sam3_concept import Sam3ConceptSegmenter
from .web_api import build_router, _user_id

log = logging.getLogger("snowlabel.server")

# ----------------------------------------------------------------------
# App initialization
# ----------------------------------------------------------------------
cfg = get_config()
app = FastAPI(
    title="SnowLabel ML Backend",
    description="Progressive auto-labeling: SAM masks + DINO features + "
                "snowball k-NN memory. Label Studio compatible.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DEBUG: log every incoming request path + origin, so we can diagnose
# "is LS even hitting us?" vs "LS hits the wrong path".
@app.middleware("http")
async def _log_every_request(request, call_next):
    try:
        origin = request.headers.get("origin", "-")
        log.info("REQ %s %s origin=%s", request.method, request.url.path, origin)
    except Exception:
        pass
    return await call_next(request)

# Single global predictor — heavy to construct, safe to share.
predictor = SnowPredictor(cfg)

# Web-native /api/* routes for the canvas labeling tool (labeler.html). They
# share the predictor above, so a click here grows the same snowball memory as
# a click coming through the Label Studio protocol path.
app.include_router(build_router(predictor))

# SAM3 concept segmenter for batch pre-annotation (lazy-loaded on first use,
# so the server starts fast and only pays the ~3.4GB VRAM when /preannotate
# is actually called).
_sam3 = Sam3ConceptSegmenter()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_image_from_task(task: dict, image_field: str = "image") -> tuple[Image.Image, str]:
    """Fetch the PIL image referenced by a Label Studio task dict.

    LS image URIs may be local (`/data/upload/xxx.jpg`), absolute URLs, or
    `data:` URIs (base64). We handle the first two; base64 isn't expected here.
    """
    data = task.get("data", {})
    uri = data.get(image_field) or data.get("image") or data.get("img")
    if not uri:
        raise HTTPException(400, f"Task is missing image field '{image_field}'")

    if uri.startswith("http://") or uri.startswith("https://"):
        r = requests.get(uri, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img, uri

    if uri.startswith("/data/"):
        # Local LS upload — fetch via the LS server. Caller must set
        # LABEL_STUDIO_URL env var (e.g. http://localhost:8080) and a token.
        host = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
        token = os.environ.get("LABEL_STUDIO_API_KEY", "")
        headers = {"Authorization": f"Token {token}"} if token else {}
        r = requests.get(host.rstrip("/") + uri, headers=headers, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img, uri

    # Treat anything else as a file path on disk.
    img = Image.open(uri).convert("RGB")
    return img, uri


def _prediction_to_ls_result(
    pred: Prediction,
    *,
    from_name_brush: str,
    from_name_choice: str,
    to_name: str,
    image_size: tuple[int, int],
) -> list[dict]:
    """Convert one SnowLabel Prediction into Label Studio result items."""
    items: list[dict] = []
    H, W = image_size
    rle = encode_mask_as_brush_rle(pred.mask)

    brush_item: dict[str, Any] = {
        "from_name": from_name_brush,
        "to_name": to_name,
        "type": "brushlabels",
        "original_width": W,
        "original_height": H,
        "image_rotation": 0,
        "value": {
            "format": "rle",
            "rle": rle,
        },
        "score": float(pred.confidence),
    }
    # Always attach a label so Label Studio actually renders the brush mask —
    # a brush region with no brushlabels draws nothing. For silent / cold-start
    # clicks (empty memory) we fall back to the closest top-k guess, or "other"
    # when memory is still empty; the user just recolors it to the right class.
    brush_label = pred.label or (pred.top_k[0][0] if pred.top_k else "other")
    brush_item["value"]["brushlabels"] = [brush_label]
    items.append(brush_item)

    # Also surface the prediction as a "choices" item so the user can confirm /
    # change it via a dropdown without messing with the brush color picker.
    if pred.label and pred.tier in ("auto", "suggest"):
        items.append({
            "from_name": from_name_choice,
            "to_name": to_name,
            "type": "choices",
            "value": {"choices": [pred.label]},
            "score": float(pred.confidence),
            "meta": {"snowlabel_tier": pred.tier, "top_k": pred.top_k},
        })

    return items


# ----------------------------------------------------------------------
# Label Studio required routes
# ----------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Heartbeat — LS pings this to verify the backend is up."""
    return {"status": "UP", "model_version": "snowlabel-0.1.0"}


@app.post("/setup")
def setup(request: dict) -> dict:
    """Called by LS when a project connects. We just acknowledge."""
    log.info("Setup called: project=%s", request.get("project"))
    return {
        "model_version": "snowlabel-0.1.0",
        "memory_size": len(predictor.memory),
    }


class PredictRequest(BaseModel):
    tasks: list[dict]
    project: dict | str | None = None
    label_config: str | None = None
    params: dict | None = None
    context: dict | None = None
    model_version: str | None = None


@app.post("/predict")
def predict_endpoint(req: PredictRequest) -> dict:
    """Main predict route — LS calls this for each task or interactive prompt."""
    # DEBUG: dump the raw incoming payload so we can see exactly what LS sends
    # on a smart click (diagnoses why no mask appears).
    try:
        import json as _json
        _dbg = {
            "has_context": req.context is not None,
            "context": req.context,
            "params": req.params,
            "model_version": req.model_version,
            "n_tasks": len(req.tasks),
            "task0_data_keys": list(req.tasks[0].get("data", {}).keys()) if req.tasks else [],
        }
        (cfg.project_root / "data" / "last_predict.json").write_text(
            _json.dumps(_dbg, indent=2, default=str), encoding="utf-8")
        log.info("PREDICT has_context=%s context=%s", req.context is not None, req.context)
    except Exception:
        pass
    cfg_brush = "tag"     # must match the BrushLabels `name` in the LS XML
    cfg_choice = "choice"  # the Choices block name
    cfg_to = "image"

    predictions = []
    for task in req.tasks:
        try:
            img, uri = _load_image_from_task(task)
        except Exception as e:
            log.exception("Failed loading task image: %s", e)
            predictions.append({
                "result": [],
                "model_version": "snowlabel-0.1.0",
                "error": str(e),
            })
            continue

        # Interactive mode?
        # LS protocol quirk: some versions send the interactive `context` at
        # the top level, others nest it inside `params` (alongside login/pw).
        # Try both, top-level first.
        prompt = extract_interactive_prompt(req.context)
        if prompt is None and req.params and isinstance(req.params, dict):
            nested = req.params.get("context")
            if isinstance(nested, dict):
                prompt = extract_interactive_prompt(nested)
        results: list[dict] = []

        if prompt is not None:
            log.info("Interactive predict: %s on %s", prompt, uri[:80])
            if prompt["type"] == "point":
                pred = predictor.predict_from_point(img, prompt["x"], prompt["y"])
                if pred is not None:
                    results = _prediction_to_ls_result(
                        pred,
                        from_name_brush=cfg_brush,
                        from_name_choice=cfg_choice,
                        to_name=cfg_to,
                        image_size=img.size[::-1],
                    )
            elif prompt["type"] == "box":
                pred = predictor.predict_from_box(img, tuple(prompt["bbox"]))
                if pred is not None:
                    results = _prediction_to_ls_result(
                        pred,
                        from_name_brush=cfg_brush,
                        from_name_choice=cfg_choice,
                        to_name=cfg_to,
                        image_size=img.size[::-1],
                    )
        else:
            # Auto mode: propose all masks for the image
            log.info("Auto predict (no prompt) on %s", uri[:80])
            preds = predictor.predict_all(img, max_masks=12)
            for p in preds:
                # Skip masks where we have no label opinion — they would just
                # be visual noise in the LS UI.
                if p.label is None and p.tier == "silent":
                    continue
                results.extend(_prediction_to_ls_result(
                    p,
                    from_name_brush=cfg_brush,
                    from_name_choice=cfg_choice,
                    to_name=cfg_to,
                    image_size=img.size[::-1],
                ))

        # Average confidence over kept items — LS surfaces this in the queue.
        avg_score = (
            float(np.mean([r.get("score", 0.0) for r in results]))
            if results else 0.0
        )
        predictions.append({
            "result": results,
            "score": avg_score,
            "model_version": "snowlabel-0.1.0",
        })

    return {"results": predictions}


@app.post("/webhook")
def webhook(payload: dict) -> dict:
    """LS pushes annotation events here (configured in LS project settings).

    We listen for the ANNOTATION_CREATED / ANNOTATION_UPDATED events and
    snowball the confirmed labels into memory.
    """
    action = payload.get("action", "")
    if action not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
        return {"status": "ignored", "action": action}

    annotation = payload.get("annotation", {})
    task = payload.get("task", {})
    if not task or not annotation:
        return {"status": "ignored", "reason": "missing task or annotation"}

    try:
        img, uri = _load_image_from_task(task)
    except Exception as e:
        log.exception("Webhook image load failed: %s", e)
        return {"status": "error", "reason": str(e)}

    learned = 0
    for item in annotation.get("result", []):
        if item.get("type") != "brushlabels":
            continue
        val = item.get("value", {})
        labels = val.get("brushlabels", [])
        if not labels:
            continue
        try:
            mask = decode_brush_rle(
                val["rle"],
                width=item.get("original_width", img.size[0]),
                height=item.get("original_height", img.size[1]),
            )
        except Exception as e:
            log.warning("Could not decode brush RLE: %s", e)
            continue
        predictor.remember(img, mask, labels[0], image_uri=uri, verified=True)
        learned += 1

    return {
        "status": "ok",
        "action": action,
        "learned": learned,
        "memory_size": len(predictor.memory),
    }


# ----------------------------------------------------------------------
# SAM3 concept batch pre-annotation
# ----------------------------------------------------------------------
class PreannotateRequest(BaseModel):
    tasks: list[dict]
    labels: list[str] | None = None     # default: classes the memory already knows
    score_threshold: float = 0.4
    max_instances: int = 50


@app.post("/preannotate")
def preannotate_endpoint(req: PreannotateRequest) -> dict:
    """SAM3 concept batch pre-annotation.

    The complement to the click-based snowball: once the memory has learned a
    class *name* from your clicks, SAM3 recalls every instance of that concept
    across the image(s) via a text prompt. Returns Label Studio result items
    (brush mask + choice label) just like /predict.
    """
    if not _sam3.available:
        raise HTTPException(
            503,
            "SAM3 weights not found at weights/sam3_hf. "
            "Run scripts/convert_multiplex_to_hf.py first.",
        )
    labels = req.labels or predictor.memory.classes
    if not labels:
        raise HTTPException(
            400, "No labels provided and snowball memory is empty — nothing to recall.",
        )

    cfg_brush, cfg_choice, cfg_to = "tag", "choice", "image"
    predictions = []
    for task in req.tasks:
        try:
            img, uri = _load_image_from_task(task)
        except Exception as e:
            log.exception("preannotate: image load failed: %s", e)
            predictions.append({"result": [], "error": str(e)})
            continue

        W, H = img.size
        items: list[dict] = []
        per_label: dict[str, int] = {}
        for text in labels:
            insts = _sam3.segment_concept(img, text, max_instances=req.max_instances)
            kept = [i for i in insts if i.score >= req.score_threshold]
            per_label[text] = len(kept)
            for inst in kept:
                rle = encode_mask_as_brush_rle(inst.mask)
                items.append({
                    "from_name": cfg_brush, "to_name": cfg_to, "type": "brushlabels",
                    "original_width": W, "original_height": H, "image_rotation": 0,
                    "value": {"format": "rle", "rle": rle, "brushlabels": [text]},
                    "score": float(inst.score),
                })
                items.append({
                    "from_name": cfg_choice, "to_name": cfg_to, "type": "choices",
                    "value": {"choices": [text]}, "score": float(inst.score),
                    "meta": {"source": "sam3_concept"},
                })
        predictions.append({
            "result": items,
            "model_version": "snowlabel-sam3-0.1.0",
            "meta": {"per_label_counts": per_label, "labels_used": labels},
        })

    return {"results": predictions}


# ----------------------------------------------------------------------
# SnowLabel-specific routes (memory inspector / admin)
# ----------------------------------------------------------------------
def _static_html(name: str, *, no_store: bool = True) -> HTMLResponse:
    from pathlib import Path as _P
    html = _P(__file__).parent / "static" / name
    headers = {}
    if no_store:
        headers = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}
    return HTMLResponse(html.read_text(encoding="utf-8"), headers=headers)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """Serve the short public demo landing page."""
    return _static_html("home.html")


@app.get("/try", response_class=HTMLResponse)
@app.get("/label", response_class=HTMLResponse)
@app.get("/labeler", response_class=HTMLResponse)
def labeler() -> HTMLResponse:
    """Serve the canvas labeling tool — the primary SnowLabel UI.
    no-store:demo 前端迭代频繁,禁止浏览器/CDN 缓存此页,免得改了前端用户却拿到旧版。"""
    return _static_html("labeler.html")


@app.get("/inspector", response_class=HTMLResponse)
def inspector() -> HTMLResponse:
    """Serve the Memory Inspector single-page UI (reads the /memory/* APIs)."""
    return _static_html("inspector.html")


@app.get("/stats")
def stats() -> dict:
    return predictor.stats()


@app.get("/memory")
def list_memory(
    limit: int = 50,
    offset: int = 0,
    label: str | None = None,
    verified_only: bool = False,
) -> dict:
    """List memory entries with optional class / verification filters."""
    entries = list(predictor.memory.iter_entries())
    if label:
        entries = [e for e in entries if e.label == label]
    if verified_only:
        entries = [e for e in entries if e.verified]
    total = len(entries)
    window = entries[offset:offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "filter": {"label": label, "verified_only": verified_only},
        "entries": [e.__dict__ for e in window],
    }


@app.get("/memory/by-class/{label}")
def memory_by_class(label: str, limit: int = 100) -> dict:
    """All entries of a single class (newest first by id)."""
    entries = [e for e in predictor.memory.iter_entries() if e.label == label]
    entries.sort(key=lambda e: -e.entry_id)
    return {
        "label": label,
        "total": len(entries),
        "entries": [e.__dict__ for e in entries[:limit]],
    }


@app.get("/memory/distribution")
def memory_distribution(request: Request) -> dict:
    """Class -> count, plus totals. Lighter than /stats — no model info.

    多租户(公网 Demo)下按调用者 user_id 过滤,让每个访客的「雪球记忆」面板
    只显示自己教过的——新访客看到的就是 0,亲眼看它从零长起来(否则会串到
    全局记忆,跟"从零开始、越标越准"的叙事自相矛盾)。单租户(uid="")保持全局。"""
    from collections import Counter
    uid = _user_id(request)
    entries = predictor.memory.iter_entries()
    if uid:
        entries = [e for e in entries if getattr(e, "user_id", "") == uid]
    counts = Counter(e.label for e in entries)
    return {
        "total": sum(counts.values()),
        "dim": predictor.dino.embed_dim if predictor._memory is not None else None,
        "backend": cfg.memory_backend,
        "classes": dict(counts.most_common()),
    }


class SearchRequest(BaseModel):
    """Either pass a feature vector directly, OR pass an image_uri + click,
    in which case the server will extract the feature for you (uses SAM+DINO).
    """
    vector: list[float] | None = None
    image_uri: str | None = None
    click_x: int | None = None
    click_y: int | None = None
    k: int = Field(10, ge=1, le=100)


@app.post("/memory/search")
def memory_search(req: SearchRequest) -> dict:
    """Nearest neighbors in the snowball memory.

    Useful for debugging mis-predictions: "what does my model think is
    closest to this image patch?".
    """
    if req.vector is not None:
        v = np.asarray(req.vector, dtype=np.float32)
        if v.shape[0] != predictor.dino.embed_dim:
            raise HTTPException(
                400,
                f"vector dim {v.shape[0]} != backbone dim {predictor.dino.embed_dim}",
            )
    elif req.image_uri and req.click_x is not None and req.click_y is not None:
        img, _ = _load_image_from_task({"data": {"image": req.image_uri}})
        sm = predictor.sam.predict_from_point(img, req.click_x, req.click_y)
        if sm is None:
            raise HTTPException(422, "SAM produced no mask for that click")
        v = predictor.feat.features(img, sm.mask).v_whole
    else:
        raise HTTPException(
            400,
            "Provide either `vector` OR (`image_uri` + `click_x` + `click_y`)",
        )

    entries, sims = predictor.memory.query(v, k=req.k)
    return {
        "k": req.k,
        "results": [
            {
                "entry_id": e.entry_id,
                "label": e.label,
                "image_uri": e.image_uri,
                "verified": e.verified,
                "similarity": float(s),
            }
            for e, s in zip(entries, sims)
        ],
    }


class ForgetRequest(BaseModel):
    entry_id: int


@app.post("/forget")
def forget(req: ForgetRequest) -> dict:
    ok = predictor.memory.forget(req.entry_id)
    return {"status": "ok" if ok else "not_found", "entry_id": req.entry_id,
            "memory_size": len(predictor.memory)}


class ClearRequest(BaseModel):
    confirm: str = Field(..., description="Must be the exact string 'YES_DELETE_ALL'")


@app.post("/memory/clear")
def clear_memory(req: ClearRequest) -> dict:
    """Drop every entry. Requires explicit confirm token to prevent accidents."""
    if req.confirm != "YES_DELETE_ALL":
        raise HTTPException(400, "confirm must be 'YES_DELETE_ALL'")
    ids_before = [e.entry_id for e in predictor.memory.iter_entries()]
    removed = 0
    for eid in ids_before:
        if predictor.memory.forget(eid):
            removed += 1
    return {"status": "ok", "removed": removed,
            "memory_size_after": len(predictor.memory)}


# Keep the {entry_id} catch-all LAST among /memory/* GETs so the more
# specific routes above don't get hijacked by integer coercion.
@app.get("/memory/{entry_id}")
def get_entry(entry_id: int) -> dict:
    """Look up one entry by id."""
    for e in predictor.memory.iter_entries():
        if e.entry_id == entry_id:
            return e.__dict__
    raise HTTPException(404, f"entry_id {entry_id} not found")


def run() -> None:
    """Entrypoint for `python -m snowlabel.server`."""
    import uvicorn
    logging.basicConfig(
        level=os.environ.get("SNOWLABEL_LOG", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    log.info("Starting SnowLabel server on %s:%d", cfg.host, cfg.port)
    log.info("Memory dir: %s", cfg.memory_dir)
    uvicorn.run(
        "snowlabel.server:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
