"""Unit-test the Label Studio protocol layer (no GPU needed).

Confirms the click→mask→label HTTP path is wired correctly independent of the
browser UI: (1) LS context -> interactive prompt extraction, (2) Prediction ->
LS result items (brush mask + choice label). Model loading is lazy, so importing
the server here does not touch the GPU.
"""
import numpy as np

from snowlabel.ls_protocol import extract_interactive_prompt
from snowlabel.predictor import Prediction
from snowlabel.server import _prediction_to_ls_result

# 1. keypoint context (LS sends percent coords)
ctx = {"result": [{"type": "keypoint", "original_width": 640, "original_height": 480,
                   "value": {"x": 50.0, "y": 25.0}}]}
p = extract_interactive_prompt(ctx)
assert p == {"type": "point", "x": 320, "y": 120}, p
print("keypoint ->", p)

ctx2 = {"result": [{"type": "keypointlabels", "original_width": 100, "original_height": 200,
                    "value": {"x": 10, "y": 20}}]}
assert extract_interactive_prompt(ctx2) == {"type": "point", "x": 10, "y": 40}
print("keypointlabels ->", extract_interactive_prompt(ctx2))

ctx3 = {"result": [{"type": "rectanglelabels", "original_width": 100, "original_height": 100,
                    "value": {"x": 10, "y": 20, "width": 30, "height": 40}}]}
assert extract_interactive_prompt(ctx3) == {"type": "box", "bbox": (10, 20, 40, 60)}
print("rectangle ->", extract_interactive_prompt(ctx3))

assert extract_interactive_prompt(None) is None
assert extract_interactive_prompt({}) is None
assert extract_interactive_prompt({"result": []}) is None
print("empty/None contexts -> None OK")

# 2. Prediction -> LS result items
mask = np.zeros((20, 30), dtype=bool)
mask[5:15, 5:25] = True

pred = Prediction(mask=mask, bbox=(5, 5, 25, 15), label="cat", confidence=0.9,
                  tier="auto", top_k=[("cat", 0.9), ("dog", 0.1)])
items = _prediction_to_ls_result(pred, from_name_brush="tag", from_name_choice="choice",
                                 to_name="image", image_size=(20, 30))
types = [i["type"] for i in items]
assert "brushlabels" in types and "choices" in types, types
brush = next(i for i in items if i["type"] == "brushlabels")
assert brush["value"]["brushlabels"] == ["cat"]
assert brush["original_width"] == 30 and brush["original_height"] == 20
print("auto tier -> items:", types, "| brush label:", brush["value"]["brushlabels"],
      "| score:", brush["score"])

# silent tier: brush present WITH a fallback label (LS renders nothing for an
# unlabeled brush region — see _prediction_to_ls_result), but no choices item.
pred_s = Prediction(mask=mask, bbox=None, label=None, confidence=0.2, tier="silent", top_k=[])
items_s = _prediction_to_ls_result(pred_s, from_name_brush="tag", from_name_choice="choice",
                                   to_name="image", image_size=(20, 30))
types_s = [i["type"] for i in items_s]
assert "choices" not in types_s, types_s
assert items_s[0]["value"]["brushlabels"] == ["other"], items_s[0]["value"]
print("silent tier -> items:", types_s, "(brush w/ fallback label, no choice)")

print("\nALL PROTOCOL TESTS PASSED")
