"""Text-prompted object detection with SAM 3, on the GPU."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

from highlight.video import resolve_device

DOWNLOAD_HINT = "uvx --from huggingface_hub hf download facebook/sam3 sam3.pt --local-dir models"


def require_weights(model: str | Path) -> None:
    if not Path(model).is_file():
        raise FileNotFoundError(
            f"{model} not found. SAM 3 weights require approved Hugging Face access. "
            f"Sign in with `uv run hf auth login`, then download with:\n  {DOWNLOAD_HINT}"
        )


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

    def __init__(self, model: str | Path, imgsz: int = 644, conf: float = 0.25, device: str | None = None):
        require_weights(model)
        self.device = resolve_device(device)
        self.calls = 0
        self.inference_s = 0.0
        self.predictor = SAM3SemanticPredictor(
            overrides=dict(
                model=str(model),
                task="segment",
                mode="predict",
                imgsz=imgsz,
                conf=conf,
                device=self.device,
                quantize=16,
                save=False,
                verbose=False,
            )
        )
        self.predictor.setup_model()

    def detect(self, bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        start = time.perf_counter()
        self.predictor.set_image(bgr)
        boxes = self.predictor(text=prompts)[0].boxes
        detections = [
            Detection(prompts[k], float(c), b)
            for b, c, k in zip(
                boxes.xyxy.float().cpu().numpy(),
                boxes.conf.float().cpu().numpy(),
                boxes.cls.int().cpu().numpy(),
            )
        ]
        self.calls += 1
        self.inference_s += time.perf_counter() - start
        return detections
