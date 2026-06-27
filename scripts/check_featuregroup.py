"""End-to-end smoke test for the crop-encode + feature-group + rerank pipeline.

Exercises the live API: cold predict (open-set silent on empty memory) ->
commit (Chinese label) -> warm predict (rerank should recall it) -> everything
mode (batched crop-encode, timed) -> open-set corner click -> cleanup.

Run with the server up:  python scripts/check_featuregroup.py
"""
import io
import time

import requests
from PIL import Image

B = "http://localhost:9090"


def seg_point(iid, x, y):
    return requests.post(B + "/api/segment", json={
        "image_id": iid, "type": "point", "points": [[x, y]], "labels": [1],
    }).json()


def main():
    ds = requests.get(B + "/api/dataset").json()
    iid = ds["images"][0]["id"]
    img = Image.open(io.BytesIO(requests.get(B + "/api/image/" + iid).content))
    w, h = img.size
    print(f"image={iid} {w}x{h}")

    r0 = seg_point(iid, w // 2, h // 2)
    print(f"[cold]  ok={r0['ok']} label={r0.get('label')} tier={r0.get('tier')} "
          f"conf={r0.get('confidence')} mask_bytes={len(r0.get('mask_png',''))}")
    assert r0["ok"], "no mask from center click"
    assert r0.get("label") is None and r0.get("tier") == "silent", \
        "empty memory should be open-set silent"

    cm = requests.post(B + "/api/commit", json={
        "image_id": iid, "mask_png": r0["mask_png"], "label": "测试猫",
    }).json()
    eid = cm["entry_id"]
    print(f"[commit] entry_id={eid} memory={cm['memory_size']}  label='测试猫'")

    r1 = seg_point(iid, w // 2, h // 2)
    print(f"[warm]  label={r1.get('label')!r} tier={r1.get('tier')} "
          f"conf={r1.get('confidence')} top_k={r1.get('top_k')}")
    ok_recall = (r1.get("label") == "测试猫")
    print(f"        recall correct: {ok_recall}")

    t = time.time()
    sa = requests.post(B + "/api/segment_all", json={"image_id": iid, "max_masks": 20}).json()
    dt = time.time() - t
    print(f"[everything] count={sa['count']} time={dt:.1f}s (batched crop-encode)")
    for m in sa["masks"][:5]:
        print(f"   idx={m['idx']} label={m['label']!r} tier={m['tier']} "
              f"conf={m['confidence']} bbox={m['bbox']}")

    r2 = seg_point(iid, int(w * 0.04), int(h * 0.04))
    print(f"[open-set corner] label={r2.get('label')!r} tier={r2.get('tier')} "
          f"conf={r2.get('confidence')}")

    requests.post(B + "/forget", json={"entry_id": eid})
    total = requests.get(B + "/memory/distribution").json()["total"]
    print(f"[cleanup] memory total={total}  (restored: {total == 0})")
    print("\nSMOKE OK" if ok_recall and total == 0 else "\nCHECK ABOVE")


if __name__ == "__main__":
    main()
