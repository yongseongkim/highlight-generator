"""Interchangeable detail detectors, run only on shortlisted keyframes."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from ultralytics import YOLO

from highlight.sam import Detection
from highlight.video import resolve_device


class YoloObjects:
    """Use a local detection/segmentation checkpoint, including a trained shuttle model.

    Ordinary COCO weights do not have a badminton shuttlecock class. Class names
    come from the checkpoint, rather than relabeling a sports ball as a shuttle.
    """

    def __init__(self, model: Path, device: str, imgsz: int = 640):
        if not model.is_file():
            raise FileNotFoundError(f"Object checkpoint not found: {model}")
        self.device = resolve_device(device)
        self.model = YOLO(str(model))
        if self.model.task not in {"detect", "segment"}:
            raise ValueError("--object-model must be a YOLO detection or segmentation checkpoint")
        self.imgsz = imgsz
        self.calls = 0
        self.inference_s = 0.0

    def detect(self, bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        # Closed-set YOLO uses its trained classes; text prompts only apply to SAM.
        started = time.perf_counter()
        result = self.model.predict(bgr, imgsz=self.imgsz, device=self.device, quantize=16, verbose=False)[0]
        boxes = result.boxes
        found = [Detection(result.names[int(label)], float(conf), box)
                 for box, conf, label in zip(boxes.xyxy.float().cpu().numpy(),
                                             boxes.conf.float().cpu().numpy(), boxes.cls.int().cpu().numpy())]
        self.calls += 1
        self.inference_s += time.perf_counter() - started
        return found
