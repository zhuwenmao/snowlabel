"""API edge-case battery against the live server. Exercises the paths the UI
can hit but smoke tests don't: bad ids, empty masks, box prompts, negative
points, accepted-only commit_batch, length-mismatched prompts, traversal."""
import base64
import io
import sys

import requests
from PIL import Image

B = "http://localhost:9090"
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def main():
    ds = requests.get(B + "/api/dataset").json()
    iid = ds["images"][0]["id"]
    img = Image.open(io.BytesIO(requests.get(B + "/api/image/" + iid).content))
    w, h = img.size
    print(f"image={iid} {w}x{h}")

    # --- invalid image ids
    r = requests.get(B + "/api/image/no_such_file.jpg")
    check("404 on missing image", r.status_code == 404)
    r = requests.get(B + "/api/image/..%5Csnowlabel%5Cconfig.py")
    check("path traversal rejected", r.status_code in (400, 404), f"({r.status_code})")
    r = requests.post(B + "/api/segment", json={"image_id": "nope.jpg",
                                                "type": "point", "x": 1, "y": 1})
    check("segment on missing image -> 404", r.status_code == 404)

    # --- prompt validation
    r = requests.post(B + "/api/segment", json={"image_id": iid, "type": "point"})
    check("point without coords -> 400", r.status_code == 400)
    r = requests.post(B + "/api/segment", json={"image_id": iid, "type": "point",
                                                "points": [[10, 10], [20, 20]],
                                                "labels": [1]})
    check("points/labels length mismatch -> 400", r.status_code == 400)
    r = requests.post(B + "/api/segment", json={"image_id": iid, "type": "box",
                                                "bbox": [10, 10, 50]})
    check("bad bbox -> 400", r.status_code == 400)
    r = requests.post(B + "/api/segment", json={"image_id": iid, "type": "wat"})
    check("unknown prompt type -> 400", r.status_code == 400)

    # --- working prompts
    r = requests.post(B + "/api/segment", json={
        "image_id": iid, "type": "box",
        "bbox": [w // 4, h // 4, 3 * w // 4, 3 * h // 4]}).json()
    check("box prompt returns mask", r.get("ok") and len(r.get("mask_png", "")) > 0)

    seg_pos = requests.post(B + "/api/segment", json={
        "image_id": iid, "type": "point",
        "points": [[w // 2, h // 2]], "labels": [1]}).json()
    seg_neg = requests.post(B + "/api/segment", json={
        "image_id": iid, "type": "point",
        "points": [[w // 2, h // 2], [w // 2 + 40, h // 2]],
        "labels": [1, 0]}).json()
    check("pos and pos+neg both ok", seg_pos.get("ok") and seg_neg.get("ok"))
    a = len(base64.b64decode(seg_pos["mask_png"]))
    b = len(base64.b64decode(seg_neg["mask_png"]))
    check("negative point changes the mask", a != b, f"(png {a} vs {b} bytes)")

    # --- commit validation
    blank = Image.new("L", (w, h), 0)
    buf = io.BytesIO(); blank.save(buf, format="PNG")
    empty_png = base64.b64encode(buf.getvalue()).decode()
    r = requests.post(B + "/api/commit", json={"image_id": iid,
                                               "mask_png": empty_png, "label": "x"})
    check("empty mask commit -> 422", r.status_code == 422)
    r = requests.post(B + "/api/commit", json={"image_id": iid,
                                               "mask_png": seg_pos["mask_png"],
                                               "label": "   "})
    check("blank label commit -> 400", r.status_code == 400)

    # --- accepted-only commit_batch (the all-correct Enter flow)
    n0 = requests.get(B + "/memory/distribution").json()["total"]
    r = requests.post(B + "/api/commit_batch", json={
        "image_id": iid, "items": [],
        "accepted": [{"label": "cat", "pred_label": "cat", "tier": "auto",
                      "confidence": 0.9, "source": "all"}]}).json()
    n1 = requests.get(B + "/memory/distribution").json()["total"]
    check("accepted-only batch commits nothing", r.get("committed") == 0 and n0 == n1)

    # --- skip-invalid behavior in batch
    r = requests.post(B + "/api/commit_batch", json={
        "image_id": iid,
        "items": [{"mask_png": empty_png, "label": "skipme"},
                  {"mask_png": seg_pos["mask_png"], "label": "边界批量"}]}).json()
    check("batch skips empty mask, commits valid", r.get("committed") == 1)
    for eid in r.get("entry_ids", []):
        requests.post(B + "/forget", json={"entry_id": eid})
    nf = requests.get(B + "/memory/distribution").json()["total"]
    check("cleanup restored memory", nf == n0, f"(total={nf})")

    print("\nEDGES OK" if not FAILS else f"\nEDGE FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
