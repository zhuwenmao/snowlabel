"""API edge-case battery against the live server. Exercises the paths the UI
can hit but smoke tests don't: bad ids, empty masks, box prompts, negative
points, accepted-only commit_batch, length-mismatched prompts, traversal."""
import base64
import io
import sys
import uuid

import requests
import numpy as np
from PIL import Image

B = "http://localhost:9090"
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def decode_mask(mask_png):
    raw = base64.b64decode(mask_png)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L")) > 0


def main():
    session = requests.Session()
    session.headers["X-Snow-User"] = "api_edges_" + uuid.uuid4().hex
    stats = session.get(B + "/stats").json()
    multi_user = bool(stats.get("multi_user"))
    loaded_samples = False
    try:
        ds = session.get(B + "/api/dataset").json()
        if not ds.get("images") and multi_user:
            response = session.post(B + "/api/load_samples",
                                    json={"dataset": "common_coco_vehicles"})
            response.raise_for_status()
            loaded_samples = True
            ds = session.get(B + "/api/dataset").json()
        if not ds.get("images"):
            raise RuntimeError("dataset is empty; configure SNOWLABEL_DATASET_DIR")
        iid = ds["images"][0]["id"]
        img = Image.open(io.BytesIO(session.get(B + "/api/image/" + iid).content))
        w, h = img.size
        print(f"image={iid} {w}x{h} multi_user={multi_user}")

        run_checks(session, iid, w, h, multi_user)
    finally:
        if multi_user and loaded_samples:
            session.post(B + "/api/reset")

    print("\nEDGES OK" if not FAILS else f"\nEDGE FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


def run_checks(session, iid, w, h, multi_user):
    if multi_user:
        check("memory rejects missing session",
              requests.get(B + "/memory").status_code == 401)
        check("memory rejects invalid session",
              requests.get(B + "/memory", headers={"X-Snow-User": "../bad"}).status_code == 401)
        check("public legacy predict is disabled",
              requests.post(B + "/predict", json={"tasks": []}).status_code == 404)
        check("global clear rejects missing session",
              requests.post(B + "/memory/clear",
                            json={"confirm": "YES_DELETE_ALL"}).status_code == 401)

    # --- invalid image ids
    r = session.get(B + "/api/image/no_such_file.jpg")
    check("404 on missing image", r.status_code == 404)
    r = session.get(B + "/api/image/..%5Csnowlabel%5Cconfig.py")
    check("path traversal rejected", r.status_code in (400, 404), f"({r.status_code})")
    r = session.post(B + "/api/segment", json={"image_id": "nope.jpg",
                                                "type": "point", "x": 1, "y": 1})
    check("segment on missing image -> 404", r.status_code == 404)

    # --- prompt validation
    r = session.post(B + "/api/segment", json={"image_id": iid, "type": "point"})
    check("point without coords -> 400", r.status_code == 400)
    r = session.post(B + "/api/segment", json={"image_id": iid, "type": "point",
                                                "points": [[10, 10], [20, 20]],
                                                "labels": [1]})
    check("points/labels length mismatch -> 400", r.status_code == 400)
    r = session.post(B + "/api/segment", json={"image_id": iid, "type": "box",
                                                "bbox": [10, 10, 50]})
    check("bad bbox -> 400", r.status_code == 400)
    r = session.post(B + "/api/segment", json={"image_id": iid, "type": "wat"})
    check("unknown prompt type -> 400", r.status_code == 400)

    # --- working prompts
    box_seg = session.post(B + "/api/segment", json={
        "image_id": iid, "type": "box",
        "bbox": [w // 4, h // 4, 3 * w // 4, 3 * h // 4]}).json()
    check("box prompt returns mask",
          box_seg.get("ok") and len(box_seg.get("mask_png", "")) > 0)

    box_mask = decode_mask(box_seg["mask_png"])
    ys, xs = np.nonzero(box_mask)
    if not xs.size:
        raise RuntimeError("box prompt returned an empty PNG mask")
    cy, cx = float(ys.mean()), float(xs.mean())
    center_i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    px, py = int(xs[center_i]), int(ys[center_i])
    far_i = int(np.argmax((ys - py) ** 2 + (xs - px) ** 2))
    nx, ny = int(xs[far_i]), int(ys[far_i])

    seg_pos = session.post(B + "/api/segment", json={
        "image_id": iid, "type": "point",
        "points": [[px, py]], "labels": [1]}).json()
    seg_neg = session.post(B + "/api/segment", json={
        "image_id": iid, "type": "point",
        "points": [[px, py], [nx, ny]],
        "labels": [1, 0]}).json()
    both_ok = bool(seg_pos.get("ok") and seg_neg.get("ok"))
    check("pos and pos+neg both ok", both_ok,
          f"(pos={seg_pos.get('ok')} neg={seg_neg.get('ok')})")
    if both_ok:
        a = decode_mask(seg_pos["mask_png"])
        b = decode_mask(seg_neg["mask_png"])
        check("negative point changes the mask", not np.array_equal(a, b),
              f"(areas {int(a.sum())} vs {int(b.sum())})")
    working_mask_png = seg_pos.get("mask_png") or box_seg["mask_png"]

    # --- commit validation
    blank = Image.new("L", (w, h), 0)
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    empty_png = base64.b64encode(buf.getvalue()).decode()
    r = session.post(B + "/api/commit", json={"image_id": iid,
                                               "mask_png": empty_png, "label": "x"})
    check("empty mask commit -> 422", r.status_code == 422)
    r = session.post(B + "/api/commit", json={"image_id": iid,
                                               "mask_png": working_mask_png,
                                               "label": "   "})
    check("blank label commit -> 400", r.status_code == 400)

    # --- accepted-only commit_batch (the all-correct Enter flow)
    n0 = session.get(B + "/memory/distribution").json()["total"]
    r = session.post(B + "/api/commit_batch", json={
        "image_id": iid, "items": [],
        "accepted": [{"label": "cat", "pred_label": "cat", "tier": "auto",
                      "confidence": 0.9, "source": "all"}]}).json()
    n1 = session.get(B + "/memory/distribution").json()["total"]
    check("accepted-only batch commits nothing", r.get("committed") == 0 and n0 == n1)

    # --- skip-invalid behavior in batch
    r = session.post(B + "/api/commit_batch", json={
        "image_id": iid,
        "items": [{"mask_png": empty_png, "label": "skipme"},
                  {"mask_png": working_mask_png, "label": "batch-edge"}]}).json()
    check("batch skips empty mask, commits valid", r.get("committed") == 1)
    for eid in r.get("entry_ids", []):
        session.post(B + "/forget", json={"entry_id": eid})
    nf = session.get(B + "/memory/distribution").json()["total"]
    check("cleanup restored memory", nf == n0, f"(total={nf})")


if __name__ == "__main__":
    main()
