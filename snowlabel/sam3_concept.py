"""SAM3 concept (text-prompted) segmentation — the batch-recall complement to
the click-based SAM2 path.

SnowLabel's core interaction is click -> single mask (SAM2) -> DINO feature ->
snowball memory. That teaches the system *class names* from your clicks without
any predefined text ontology. Once a class is in memory, SAM3 closes the loop:
give it the learned class name and it recalls *every* instance of that concept
across the whole dataset in one shot (open-vocabulary concept segmentation).

This is why "Powered by SAM 3" is real: SAM3 is used for what it's actually
good at (text/concept detection), not forced into a click role it can't do.

Weights: converted HF snapshot at weights/sam3_hf (see
scripts/convert_multiplex_to_hf.py). Lazy-loaded on first use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("snowlabel.sam3_concept")


@dataclass
class ConceptInstance:
    """One detected instance of a text concept."""
    mask: np.ndarray                       # bool H×W
    score: float                           # detection confidence 0..1
    bbox: tuple[int, int, int, int]        # xyxy in image pixels


class Sam3ConceptSegmenter:
    """Text-prompted concept segmentation via HF Sam3Model.

    One instance per process; the model is heavy (~800M params, ~3.4GB VRAM).
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "auto",
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> None:
        default_dir = Path(__file__).resolve().parent.parent / "weights" / "sam3_hf"
        self.model_dir = str(model_dir or default_dir)
        self._device_pref = device
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self._model = None
        self._proc = None

    def _device(self) -> str:
        if self._device_pref == "cpu":
            return "cpu"
        if self._device_pref == "cuda" or torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import Sam3Model, Sam3Processor
            dev = self._device()
            log.info("Loading SAM3 concept model from %s on %s", self.model_dir, dev)
            self._model = Sam3Model.from_pretrained(self.model_dir).to(dev).eval()
            self._proc = Sam3Processor.from_pretrained(self.model_dir)
            log.info("SAM3 concept model ready.")
        return self._model, self._proc

    @property
    def available(self) -> bool:
        """True if the converted HF weights exist on disk."""
        return (Path(self.model_dir) / "model.safetensors").is_file()

    @torch.inference_mode()
    def segment_concept(
        self,
        image: Image.Image,
        text: str,
        max_instances: int = 50,
    ) -> list[ConceptInstance]:
        """Return every instance of `text` concept found in `image`."""
        model, proc = self._ensure_loaded()
        image = image.convert("RGB")
        inputs = proc(images=image, text=text, return_tensors="pt").to(self._device())
        out = model(**inputs)
        res = proc.post_process_instance_segmentation(
            out,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        instances: list[ConceptInstance] = []
        n = min(len(res["scores"]), max_instances)
        for i in range(n):
            m = res["masks"][i].cpu().numpy().astype(bool)
            ys, xs = np.where(m)
            if not xs.size:
                continue
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            instances.append(ConceptInstance(
                mask=m,
                score=float(res["scores"][i]),
                bbox=bbox,
            ))
        return instances
