"""Telemetry round-trip check against the live server.

segment -> commit_batch (one corrected item + one accepted meta) ->
events.jsonl gains a "commit" (corrected=true) and an "accept" event ->
cleanup (forget the committed entry). Run with the server up.
"""
import io
import json
from pathlib import Path

import requests
from PIL import Image

B = "http://localhost:9090"
ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "data" / "telemetry" / "events.jsonl"


def main():
    n0 = sum(1 for _ in open(EVENTS, encoding="utf-8")) if EVENTS.is_file() else 0

    ds = requests.get(B + "/api/dataset").json()
    iid = ds["images"][0]["id"]
    img = Image.open(io.BytesIO(requests.get(B + "/api/image/" + iid).content))
    w, h = img.size
    seg = requests.post(B + "/api/segment", json={
        "image_id": iid, "type": "point", "points": [[w // 2, h // 2]],
        "labels": [1]}).json()
    assert seg["ok"], "no mask"

    r = requests.post(B + "/api/commit_batch", json={
        "image_id": iid,
        "items": [{"mask_png": seg["mask_png"], "label": "遥测纠正类",
                   "pred_label": "错误预测", "tier": "suggest",
                   "confidence": 0.66, "source": "point"}],
        "accepted": [{"label": "遥测接受类", "pred_label": "遥测接受类",
                      "tier": "auto", "confidence": 0.91, "source": "all"}],
    }).json()
    print(f"commit_batch: committed={r['committed']} memory={r['memory_size']}")

    lines = [json.loads(x) for x in open(EVENTS, encoding="utf-8")][n0:]
    commits = [e for e in lines if e["type"] == "commit"]
    accepts = [e for e in lines if e["type"] == "accept"]
    ok_c = any(e["label"] == "遥测纠正类" and e["corrected"] for e in commits)
    ok_a = any(e["pred_label"] == "遥测接受类" and e["tier"] == "auto" for e in accepts)
    print(f"new events={len(lines)}  commit(corrected)={ok_c}  accept(auto)={ok_a}")

    for eid in r["entry_ids"]:
        requests.post(B + "/forget", json={"entry_id": eid})
    total = requests.get(B + "/memory/distribution").json()["total"]
    print(f"cleanup: memory total={total}")
    print("\nTELEMETRY OK" if ok_c and ok_a and total == 0 else "\nCHECK ABOVE")


if __name__ == "__main__":
    main()
