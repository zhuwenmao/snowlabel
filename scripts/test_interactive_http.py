"""Hit the running server /predict exactly like LS does for a smart click:
a keypoint context. Confirms the full HTTP path (SAM mask + brush RLE) returns
a mask, isolating whether the no-mask issue is backend or LS frontend rendering.
"""
import json
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "demo_manifest.json").read_text())
catfile = next(m["file"] for m in manifest if m["label_hint"] == "cat")
W, H = Image.open(ROOT / "demo_images" / catfile).size
url = f"http://localhost:8081/{catfile}"
print(f"image: {catfile} ({W}x{H})  -> {url}")

body = {
    "tasks": [{"data": {"image": url}}],
    "context": {"result": [{
        "type": "keypoint",
        "original_width": W, "original_height": H,
        "value": {"x": 55, "y": 55},
    }]},
}
r = requests.post("http://localhost:9090/predict", json=body, timeout=120)
print("HTTP", r.status_code)
data = r.json()
pred = data.get("results", [{}])[0]
items = pred.get("result", [])
print("result items:", len(items), "| avg score:", pred.get("score"))
for it in items:
    t = it.get("type")
    v = it.get("value", {})
    if t == "brushlabels":
        rle = v.get("rle", [])
        print(f"  BRUSH  rle_len={len(rle)}  labels={v.get('brushlabels')}  "
              f"score={it.get('score')}  dims={it.get('original_width')}x{it.get('original_height')}")
    elif t == "choices":
        print(f"  CHOICE {v.get('choices')}  score={it.get('score')}")
    else:
        print(f"  {t}: {str(v)[:80]}")
if not items:
    print("  (empty result — backend returned no mask)")
