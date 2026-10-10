"""Text-prompted object detection with SAM 3, on the GPU."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

DOWNLOAD_HINT = "uvx --from huggingface_hub hf download facebook/sam3 sam3.pt --local-dir models"


@dataclass
class Detection:
    label: str  # the text prompt that matched
    conf: float
    box: np.ndarray  # (4,) xyxy pixels in the image passed to `detect`


class Sam3:
    """SAM 3 through Ultralytics. Only boxes are used; masks are discarded.

    Every image is resized to `imgsz` (a multiple of SAM 3's patch size, 14), so a
    crop of exactly that size keeps small objects like a ball at full resolution.
    """

    def __init__(self, model: str | Path, imgsz: int = 644, conf: float = 0.25, device: str = "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("Sam3 needs a CUDA GPU, but PyTorch can't see one")
        if not Path(model).exists():
            raise FileNotFoundError(f"{model} not found. SAM 3 weights are gated on Hugging Face; download with:\n  {DOWNLOAD_HINT}")
        self.predictor = SAM3SemanticPredictor(
            overrides=dict(
                model=str(model),
                task="segment",
                mode="predict",
                imgsz=imgsz,
                conf=conf,
                device=device,
                quantize=16,
                save=False,
                verbose=False,
            )
        )
        self.predictor.setup_model()

    def detect(self, bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        self.predictor.set_image(bgr)
        boxes = self.predictor(text=prompts)[0].boxes
        return [
            Detection(prompts[k], float(c), b)
            for b, c, k in zip(
                boxes.xyxy.float().cpu().numpy(),
                boxes.conf.float().cpu().numpy(),
                boxes.cls.int().cpu().numpy(),
            )
        ]
