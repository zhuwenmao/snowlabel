"""Model wrappers — SAM for masks, DINO for features.

Both classes lazy-load on first use, so importing this module is cheap.
The wrappers hide model-specific quirks behind a uniform Python API so the
predictor doesn't care which checkpoint is underneath.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import get_config

log = logging.getLogger("snowlabel.models")


# ====================================================================
# DINO feature extractor
# ====================================================================
@dataclass
class _DinoState:
    model: torch.nn.Module
    patch_size: int
    embed_dim: int
    grid: int
    preprocess: transforms.Compose
    device: torch.device
    num_register_tokens: int = 0   # DINOv2=0, DINOv3=4
    is_v3: bool = False             # routes the forward path


class DinoFeatureExtractor:
    """Mask-pool DINO patch tokens into a single L2-normalized vector.

    Same algorithm Phase 0 validated; production-grade caching layered on top.
    """

    def __init__(self, variant: str | None = None, input_res: int | None = None,
                 device: str = "auto"):
        cfg = get_config()
        self.variant = variant or cfg.dino_variant
        self.input_res = input_res or cfg.input_resolution
        self._device_pref = device if device != "auto" else cfg.device
        self._state: _DinoState | None = None

    # ---------- lazy load ----------
    def _resolve_device(self) -> torch.device:
        if self._device_pref == "cpu":
            return torch.device("cpu")
        if self._device_pref == "cuda" or torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _ensure_loaded(self) -> _DinoState:
        if self._state is not None:
            return self._state
        device = self._resolve_device()
        log.info("Loading DINO backbone %s on %s ...", self.variant, device)

        # `variant` may be:
        #   1. a local directory path holding a HF snapshot (e.g. weights/...dinov3...)
        #   2. a short name like "dinov3_vits16" -> HF repo id (gated)
        #   3. a short name like "dinov2_vitb14" -> torch.hub
        path_like = Path(self.variant)
        is_local_path = path_like.is_dir() and (path_like / "config.json").is_file()
        is_v3 = is_local_path or self.variant.startswith("dinov3")
        num_register_tokens = 0

        if is_local_path:
            # Local HF snapshot — typical for the user-supplied weights/ dir.
            try:
                from transformers import AutoModel
            except ImportError as e:
                raise ImportError(
                    "transformers required to load HF snapshot. "
                    "pip install transformers>=4.45"
                ) from e
            log.info("Loading DINOv3 from local HF snapshot: %s", path_like)
            model = AutoModel.from_pretrained(str(path_like))
            patch_size = int(model.config.patch_size)
            embed_dim = int(model.config.hidden_size)
            num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
        elif self.variant.startswith("dinov2"):
            model = torch.hub.load(
                "facebookresearch/dinov2",
                self.variant,
                trust_repo=True,
            )
            patch_size = int(getattr(model, "patch_size", 14))
            embed_dim = int(getattr(model, "embed_dim", 384))
        elif self.variant.startswith("dinov3"):
            # Requires HuggingFace access approval + `huggingface-cli login`.
            try:
                from transformers import AutoModel
            except ImportError as e:
                raise ImportError(
                    "transformers package required for DINOv3. "
                    "pip install transformers>=4.45"
                ) from e
            hf_id = {
                "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "dinov3_vits16plus": "facebook/dinov3-vitsplus-pretrain-lvd1689m",
            }.get(self.variant)
            if hf_id is None:
                raise ValueError(
                    f"Unknown DINOv3 variant: {self.variant}. "
                    f"Try dinov3_vits16 / dinov3_vitb16 / dinov3_vitl16, "
                    f"or pass a local path to a HF snapshot."
                )
            log.info("Loading DINOv3 from HuggingFace: %s", hf_id)
            model = AutoModel.from_pretrained(hf_id)
            patch_size = int(model.config.patch_size)
            embed_dim = int(model.config.hidden_size)
            num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
        else:
            raise ValueError(f"Unsupported DINO variant: {self.variant}")

        model = model.to(device).eval()
        if self.input_res % patch_size != 0:
            raise ValueError(
                f"input_res={self.input_res} must be divisible by "
                f"patch_size={patch_size}"
            )
        grid = self.input_res // patch_size

        preprocess = transforms.Compose([
            transforms.Resize((self.input_res, self.input_res),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        self._state = _DinoState(
            model=model,
            patch_size=patch_size,
            embed_dim=embed_dim,
            grid=grid,
            preprocess=preprocess,
            device=device,
            num_register_tokens=num_register_tokens,
            is_v3=is_v3,
        )
        log.info(
            "DINO ready. variant=%s patch=%d dim=%d grid=%dx%d reg_tokens=%d device=%s",
            self.variant, patch_size, embed_dim, grid, grid,
            num_register_tokens, device,
        )
        return self._state

    # ---------- public API ----------
    @property
    def embed_dim(self) -> int:
        return self._ensure_loaded().embed_dim

    @property
    def grid(self) -> int:
        return self._ensure_loaded().grid

    @torch.inference_mode()
    def encode_image(self, img_pil: Image.Image) -> torch.Tensor:
        """Forward DINO once; return patch tokens (grid*grid, embed_dim).

        DINOv2 output: dict with 'x_norm_patchtokens' already pre-stripped of CLS.
        DINOv3 output: last_hidden_state has [CLS, register_tokens..., patches...]
                       so we strip the first (1 + num_register_tokens) entries.
        """
        s = self._ensure_loaded()
        x = s.preprocess(img_pil).unsqueeze(0).to(s.device)
        if s.is_v3:
            out = s.model(pixel_values=x)
            skip = 1 + s.num_register_tokens
            tokens = out.last_hidden_state[0, skip:, :]
        else:
            out = s.model.forward_features(x)
            tokens = out["x_norm_patchtokens"][0]
        # Sanity check — failing this means resolution/register-token math drifted.
        expected = s.grid * s.grid
        if tokens.shape[0] != expected:
            raise RuntimeError(
                f"DINO returned {tokens.shape[0]} patch tokens; expected "
                f"{expected} ({s.grid}x{s.grid}). Check input_resolution and "
                f"num_register_tokens settings."
            )
        return tokens

    @torch.inference_mode()
    def encode_batch(self, pil_list: list[Image.Image]) -> torch.Tensor:
        """Forward a *batch* of (already cropped) PILs in one pass.

        Returns patch tokens [B, grid*grid, D] on the model device. Used by the
        crop-encode feature extractor so N mask-crops cost one batched forward
        instead of N sequential ones — the key to keeping "segment everything"
        fast on the GPU.
        """
        s = self._ensure_loaded()
        xs = torch.stack([s.preprocess(im) for im in pil_list]).to(s.device)
        if s.is_v3:
            out = s.model(pixel_values=xs)
            skip = 1 + s.num_register_tokens
            tokens = out.last_hidden_state[:, skip:, :]
        else:
            out = s.model.forward_features(xs)
            tokens = out["x_norm_patchtokens"]
        expected = s.grid * s.grid
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"DINO returned {tokens.shape[1]} patch tokens; expected {expected}."
            )
        return tokens

    def pool_by_mask(
        self,
        patch_tokens: torch.Tensor,
        mask_hw: np.ndarray,
    ) -> np.ndarray:
        """Mask-weighted mean of patch tokens, L2-normalized. Returns CPU ndarray."""
        s = self._ensure_loaded()
        # Resize mask to patch grid resolution
        m = Image.fromarray((mask_hw.astype(np.uint8) > 0).astype(np.uint8) * 255)
        m = m.resize((s.grid, s.grid), Image.BILINEAR)
        w = np.asarray(m, dtype=np.float32) / 255.0
        w_flat = torch.from_numpy(w.flatten()).to(patch_tokens.device)
        if float(w_flat.sum()) < 1e-3:
            feat = patch_tokens.mean(dim=0)
        else:
            feat = (patch_tokens * w_flat.unsqueeze(1)).sum(dim=0) / w_flat.sum()
        feat = feat / (feat.norm() + 1e-8)
        return feat.detach().cpu().numpy().astype(np.float32)

    def feature_for_mask(
        self,
        img_pil: Image.Image,
        mask_hw: np.ndarray,
    ) -> np.ndarray:
        """Convenience: encode image + pool by mask in one shot."""
        tokens = self.encode_image(img_pil)
        return self.pool_by_mask(tokens, mask_hw)


# ====================================================================
# SAM segmenter (via ultralytics)
# ====================================================================
@dataclass
class SamMask:
    mask: np.ndarray  # bool H×W
    score: float      # SAM-reported confidence
    bbox: tuple[int, int, int, int] | None = None  # xyxy


class SamSegmenter:
    """Thin wrapper over ultralytics SAM 2(.1) so we can swap to SAM 3 later.

    For point/box prompts we use ultralytics; for "give me all masks", we use
    the same model's automatic generator.
    """

    def __init__(self, variant: str | None = None, device: str = "auto") -> None:
        cfg = get_config()
        self.variant = variant or cfg.sam_variant
        self._device_pref = device if device != "auto" else cfg.device
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import SAM
            # SAM 3 weights are gated — ultralytics will not auto-download them.
            # We detect that case and emit a clear error before the
            # ultralytics-level FileNotFoundError reaches the user.
            is_sam3 = "sam3" in self.variant.lower()
            if is_sam3:
                from pathlib import Path
                weight_path = Path(self.variant)
                if not weight_path.is_absolute() and not weight_path.exists():
                    # ultralytics will look in cwd; we want to be explicit.
                    raise FileNotFoundError(
                        f"SAM 3 weights '{self.variant}' not found. SAM 3 is "
                        f"gated; you must manually download from "
                        f"https://huggingface.co/facebook/sam3 (after access "
                        f"approval) and place the file in this project root "
                        f"({Path.cwd()}). Then either set "
                        f"SNOWLABEL_SAM={self.variant} or pass the full path. "
                        f"See PHASE1_GUIDE.md for step-by-step."
                    )
            log.info("Loading SAM %s%s ...", self.variant,
                     "" if is_sam3 else " (auto-downloads on first run)")
            self._model = SAM(self.variant)
            log.info("SAM ready.")
        return self._model

    @property
    def device(self) -> str:
        return ("cuda" if (self._device_pref == "cuda" or
                           (self._device_pref == "auto" and torch.cuda.is_available()))
                else "cpu")

    def predict_from_point(
        self,
        img_pil: Image.Image,
        x: int,
        y: int,
    ) -> SamMask | None:
        """Single positive click → best mask SAM proposes."""
        return self.predict_from_points(img_pil, [[x, y]], [1])

    def predict_from_points(
        self,
        img_pil: Image.Image,
        points: Sequence[Sequence[int]],
        labels: Sequence[int],
    ) -> SamMask | None:
        """Multi-point prompt with positive (1) and negative (0) clicks.

        Negative points let the user *subtract* regions SAM wrongly swallowed —
        e.g. click the cat (＋), then right-click the cushion the mask leaked
        onto (－). All points jointly refine one mask.
        """
        if not points:
            return None
        model = self._ensure_loaded()
        img_np = np.asarray(img_pil.convert("RGB"))
        pts = [[int(px), int(py)] for px, py in points]
        labs = [int(v) for v in labels]
        # ultralytics treats a *flat* points list as separate single-point
        # objects (so a negative point would be ignored). Nest one level —
        # points=[pts], labels=[labs] — to declare ONE object refined by all
        # the positive (1) and negative (0) clicks together.
        result = model.predict(
            img_np, points=[pts], labels=[labs],
            device=self.device, verbose=False,
        )
        if not result or result[0].masks is None or len(result[0].masks.data) == 0:
            return None
        masks = result[0].masks.data.cpu().numpy()  # (N, H, W)
        conf = (result[0].boxes.conf.cpu().numpy()
                if result[0].boxes is not None and result[0].boxes.conf is not None
                else None)
        if masks.shape[0] == 1:
            best = 0
        elif any(v == 0 for v in labs) and conf is not None and len(conf) == masks.shape[0]:
            # User is actively disambiguating — trust SAM's top-scored mask
            # rather than the "smallest object" heuristic used for a lone click.
            best = int(np.argmax(conf))
        else:
            # Lone-point ambiguity: the smallest mask is usually the object,
            # not the whole scene. SAM 2 typically returns one for one point.
            sizes = masks.reshape(masks.shape[0], -1).sum(axis=1)
            best = int(np.argmin(sizes))
        mask = masks[best].astype(bool)
        score = float(conf[best]) if conf is not None else 1.0
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size else None
        return SamMask(mask=mask, score=score, bbox=bbox)

    def predict_from_box(
        self,
        img_pil: Image.Image,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> SamMask | None:
        """Box prompt → mask."""
        model = self._ensure_loaded()
        img_np = np.asarray(img_pil.convert("RGB"))
        result = model.predict(
            img_np, bboxes=[list(bbox_xyxy)],
            device=self.device, verbose=False,
        )
        if not result or result[0].masks is None or len(result[0].masks.data) == 0:
            return None
        mask = result[0].masks.data[0].cpu().numpy().astype(bool)
        score = (float(result[0].boxes.conf[0])
                 if result[0].boxes is not None else 1.0)
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size else None
        return SamMask(mask=mask, score=score, bbox=bbox)

    def predict_all(
        self,
        img_pil: Image.Image,
        max_masks: int = 32,
    ) -> list[SamMask]:
        """Automatic mode — let SAM propose all instance masks it sees."""
        model = self._ensure_loaded()
        img_np = np.asarray(img_pil.convert("RGB"))
        result = model.predict(img_np, device=self.device, verbose=False)
        if not result or result[0].masks is None:
            return []
        masks = result[0].masks.data.cpu().numpy()
        # rank by area (largest objects first) but trim to max_masks
        order = np.argsort(-masks.reshape(masks.shape[0], -1).sum(axis=1))[:max_masks]
        out: list[SamMask] = []
        for i in order:
            m = masks[i].astype(bool)
            ys, xs = np.where(m)
            if not xs.size:
                continue
            out.append(SamMask(
                mask=m,
                score=1.0,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            ))
        return out
