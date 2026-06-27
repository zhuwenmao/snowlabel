"""Verify Chinese (and emoji/mixed) labels round-trip through the live web API.

Stores each label via /api/commit_batch, reads it back via /memory/{id}, asserts
an exact match, then forgets it (leaves memory as it found it). Run with the
server up:  python scripts/check_zh_label.py
"""
import io
import sys

import requests
from PIL import Image

B = "http://localhost:9090"

TESTS = [
    "摩托车",                                            # short, common
    "重型自卸式工程运输卡车混凝土搅拌车夜间作业测试标签",   # long: > old 64-byte cap
    "猫咪abc123混合",                                    # CJK + ASCII mix
]


def main() -> None:
    iid = requests.get(B + "/api/dataset").json()["images"][0]["id"]
    img = Image.open(io.BytesIO(requests.get(B + "/api/image/" + iid).content))
    w, h = img.size
    seg = requests.post(B + "/api/segment", json={
        "image_id": iid, "type": "point", "points": [[w // 2, h // 2]], "labels": [1],
    }).json()
    mp = seg["mask_png"]

    all_ok = True
    for lbl in TESTS:
        cb = requests.post(B + "/api/commit_batch", json={
            "image_id": iid, "items": [{"mask_png": mp, "label": lbl}],
        }).json()
        eid = cb["entry_ids"][0]
        got = requests.get(B + "/memory/" + str(eid)).json()["label"]
        ok = (got == lbl)
        all_ok = all_ok and ok
        print(f"[{'OK' if ok else 'FAIL'}] chars={len(lbl):2d} "
              f"bytes={len(lbl.encode('utf-8')):3d}  in={lbl!r}  out={got!r}")
        requests.post(B + "/forget", json={"entry_id": eid})

    print("ALL MATCH:", all_ok)
    print("memory total after cleanup:",
          requests.get(B + "/memory/distribution").json()["total"])
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
