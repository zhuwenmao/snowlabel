"""Smoke-test the multi-tenant demo server (SNOWLABEL_MULTIUSER=1).

Verifies, against a running server on :9090:
  1. upload isolation  — each X-Snow-User sees only their own uploaded images
  2. memory isolation  — a label one visitor teaches is invisible to another

Run with the demo server up. Temporary; safe to delete.
"""
from __future__ import annotations

from pathlib import Path

import requests
from PIL import Image

B = "http://localhost:9090"
imgp = next(Path("demo_images").glob("*.jpg"))
blob = imgp.read_bytes()
W, Hh = Image.open(imgp).size
cx, cy = W // 2, Hh // 2


def H(uid):
    return {"X-Snow-User": uid}


def upload(uid):
    r = requests.post(B + "/api/upload", headers=H(uid),
                      files={"file": (imgp.name, blob, "image/jpeg")})
    r.raise_for_status()
    return r.json()["id"]


def dataset(uid):
    return requests.get(B + "/api/dataset", headers=H(uid)).json()


def seg(uid, img_id):
    return requests.post(B + "/api/segment", headers=H(uid),
                         json={"image_id": img_id, "type": "point",
                               "x": cx, "y": cy}).json()


print(f"img={imgp.name} size={W}x{Hh} center=({cx},{cy})")

# 1) upload isolation
ida = upload("alice")
idb = upload("bob")
da, db = dataset("alice"), dataset("bob")
print(f"alice sees {da['total']} img(s) in {da['dir']}")
print(f"bob   sees {db['total']} img(s) in {db['dir']}")
assert da["total"] == 1 and db["total"] == 1, "FAIL: upload isolation"
print("PASS upload isolation\n")

# 2) memory isolation
s0 = seg("alice", ida)
print(f"alice first segment: tier={s0.get('tier')} label={s0.get('label')} "
      f"(empty store -> expect silent)")
if s0.get("ok"):
    requests.post(B + "/api/commit", headers=H("alice"),
                  json={"image_id": ida, "mask_png": s0["mask_png"],
                        "label": "alice_thing"}).raise_for_status()
    s1 = seg("alice", ida)
    sb = seg("bob", idb)
    print(f"alice after teaching: tier={s1.get('tier')} label={s1.get('label')} "
          f"(expect alice_thing)")
    print(f"bob same spot:        tier={sb.get('tier')} label={sb.get('label')} "
          f"(expect silent / None — must NOT see alice_thing)")
    assert s1.get("label") == "alice_thing", "FAIL: alice cannot retrieve own label"
    assert sb.get("label") != "alice_thing", "FAIL: memory leaked across users!"
    print("\nPASS memory isolation")
print("STATS memory:", requests.get(B + "/stats").json()["memory"])
