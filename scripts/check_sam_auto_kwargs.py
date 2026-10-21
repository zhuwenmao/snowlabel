"""Check that SAM automatic-generation options reach Predictor.generate.

The fake Ultralytics model keeps this check fast and GPU-free.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

# The user's editable install may still point at the pre-migration checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snowlabel.models import SamSegmenter


class _Array:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def cpu(self) -> "_Array":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakePredictor:
    last: "_FakePredictor | None" = None

    def __init__(self, overrides: dict, _callbacks: dict) -> None:
        del _callbacks
        self.args = SimpleNamespace(device=overrides["device"])
        self.prompts = {"points": [[1, 1]]}
        self.call_kwargs: dict = {}
        self.setup_model_args: tuple | None = None
        _FakePredictor.last = self

    def setup_model(self, model: object, verbose: bool) -> None:
        self.setup_model_args = (model, verbose)

    def set_prompts(self, prompts: dict) -> None:
        self.prompts = prompts

    def __call__(self, *, source: np.ndarray, stream: bool, **kwargs):
        self.call_kwargs = {"source": source, "stream": stream, **kwargs}
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[0, 1:3, 1:3] = 1
        masks[1, 2:7, 2:7] = 1
        return [
            SimpleNamespace(
                masks=SimpleNamespace(data=_Array(masks)),
                boxes=SimpleNamespace(
                    conf=_Array(np.array([0.91, 0.82], dtype=np.float32))
                ),
            )
        ]


class _FakeSamModel:
    def __init__(self) -> None:
        self.predictor = None
        self.overrides = {}
        self.callbacks = {}
        self.model = object()

    def _smart_load(self, name: str):
        assert name == "predictor"
        return _FakePredictor

    def predict(self, *args, **kwargs):
        raise AssertionError("automatic mode must bypass Model.predict validation")


def main() -> None:
    segmenter = SamSegmenter(device="cpu")
    segmenter.cfg = SimpleNamespace(
        sam_auto_max_masks=80,
        sam_auto_points_per_side=64,
        sam_auto_crop_layers=1,
        sam_auto_conf_thres=0.80,
        sam_auto_stability_thres=0.90,
    )
    model = _FakeSamModel()
    segmenter._model = model

    result = segmenter.predict_all(Image.new("RGB", (8, 8)), max_masks=2)
    predictor = _FakePredictor.last
    assert predictor is model.predictor
    assert predictor is not None
    assert predictor.prompts == {}
    assert predictor.setup_model_args == (model.model, False)
    assert predictor.call_kwargs["stream"] is False
    assert predictor.call_kwargs["points_stride"] == 64
    assert predictor.call_kwargs["crop_n_layers"] == 1
    assert predictor.call_kwargs["conf_thres"] == 0.80
    assert predictor.call_kwargs["stability_score_thresh"] == 0.90
    assert len(result) == 2
    print("PASS: SAM automatic kwargs reach Predictor.generate path")


if __name__ == "__main__":
    main()
