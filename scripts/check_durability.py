"""Milvus durability check, phase 1: commit one entry via the live API.

Run with the server up; then restart the server and run with --verify to
confirm the entry survived (RAM mirror reloads from Milvus at startup —
if milvus-lite didn't persist the insert, it's gone). Cleanup with --cleanup.
"""
import io
import sys

import requests
from PIL import Image

B = "http://localhost:9090"
LABEL = "持久化测试"


def dist():
    return requests.get(B + "/memory/distribution").json()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "commit"
    if mode == "commit":
        ds = requests.get(B + "/api/dataset").json()
        iid = ds["images"][0]["id"]
        img = Image.open(io.BytesIO(requests.get(B + "/api/image/" + iid).content))
        w, h = img.size
        seg = requests.post(B + "/api/segment", json={
            "image_id": iid, "type": "point", "points": [[w // 2, h // 2]],
            "labels": [1]}).json()
        assert seg["ok"], "segment failed"
        cm = requests.post(B + "/api/commit", json={
            "image_id": iid, "mask_png": seg["mask_png"], "label": LABEL}).json()
        print(f"committed entry={cm['entry_id']} memory={cm['memory_size']}")
        print("dist:", dist())
    elif mode == "verify":
        d = dist()
        ok = d["classes"].get(LABEL, 0) >= 1
        print("dist after restart:", d)
        print("DURABILITY OK" if ok else "DURABILITY FAILED — insert lost on restart!")
        sys.exit(0 if ok else 1)
    elif mode == "cleanup":
        rows = requests.get(B + "/memory/by-class/" + LABEL).json()
        n = 0
        for e in rows.get("entries", []):
            requests.post(B + "/forget", json={"entry_id": e["entry_id"]})
            n += 1
        print(f"forgot {n}; dist:", dist())


if __name__ == "__main__":
    main()
