"""Smoke-test the multi-tenant demo server (SNOWLABEL_MULTIUSER=1).

Verifies, against a running server on :9090:
  1. upload isolation  — each X-Snow-User sees only their own uploaded images
  2. memory isolation  — a label one visitor teaches is invisible to another

Run with the demo server up. Temporary; safe to delete.
"""
from __future__ import annotations

from pathlib import Path
import uuid

import requests
from PIL import Image

B = "http://localhost:9090"
imgp = next(Path("demo_images").glob("*.jpg"))
blob = imgp.read_bytes()
W, Hh = Image.open(imgp).size
cx, cy = W // 2, Hh // 2


def session(uid):
    s = requests.Session()
    s.headers["X-Snow-User"] = uid
    return s


def upload(s):
    r = s.post(B + "/api/upload",
               files={"file": (imgp.name, blob, "image/jpeg")})
    r.raise_for_status()
    return r.json()["id"]


def dataset(s):
    return s.get(B + "/api/dataset").json()


def seg(s, img_id):
    return s.post(B + "/api/segment",
                  json={"image_id": img_id, "type": "point",
                        "x": cx, "y": cy}).json()


suffix = uuid.uuid4().hex
alice = session("diag_a_" + suffix)
bob = session("diag_b_" + suffix)

print(f"img={imgp.name} size={W}x{Hh} center=({cx},{cy})")
try:
    # 1) upload isolation
    ida = upload(alice)
    idb = upload(bob)
    da, db = dataset(alice), dataset(bob)
    print(f"visitor A sees {da['total']} image(s)")
    print(f"visitor B sees {db['total']} image(s)")
    assert da["total"] == 1 and db["total"] == 1, "FAIL: upload isolation"
    print("PASS upload isolation\n")

    # 2) memory isolation
    s0 = seg(alice, ida)
    print(f"visitor A first segment: tier={s0.get('tier')} label={s0.get('label')}")
    if s0.get("ok"):
        alice.post(B + "/api/commit",
                   json={"image_id": ida, "mask_png": s0["mask_png"],
                         "label": "diag_thing"}).raise_for_status()
        memory = alice.get(B + "/memory?include_preview=1").json()
        anchor = next(e for e in memory["entries"] if e["label"] == "diag_thing")
        preview = alice.get(B + anchor["preview_url"])
        assert preview.status_code == 200, "FAIL: anchor preview unavailable"
        assert preview.headers.get("content-type", "").startswith("image/webp")
        print(f"anchor preview: #{anchor['entry_id']} {len(preview.content)} bytes")
        s1 = seg(alice, ida)
        sb = seg(bob, idb)
        print(f"visitor A after teaching: tier={s1.get('tier')} label={s1.get('label')}")
        print(f"visitor B same spot:      tier={sb.get('tier')} label={sb.get('label')}")
        assert s1.get("label") == "diag_thing", "FAIL: owner cannot retrieve own label"
        assert sb.get("label") != "diag_thing", "FAIL: memory leaked across users"
        print("\nPASS memory isolation")
finally:
    alice.post(B + "/api/reset")
    bob.post(B + "/api/reset")
