"""Lazy, pose-independent YOLO observations in the dense CV decoding pass."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from highlight.discovery_plan import context_crops
from highlight.discovery_sam import digest
from highlight.pose import file_identity


class ObjectScout:
    def __init__(self, model: Path, meta: dict, court, fps: float, budget: int,
                 device: str | None = None, factory=None):
        self.model, self.meta, self.device = model, meta, device
        self.factory, self.detector = factory, None
        self.crop = context_crops(court, 1)[0]
        count = min(budget, meta["frames"], max(1, int(meta["frames"]/meta["fps"]*fps)))
        # A finite cap must leave scouts at the end, not exhaust on the opening.
        step = min(meta["frames"]-1, max(1, round(meta["fps"]/fps)))
        anchors = (np.array([(meta["frames"]-1-step)//2]) if count//2 == 1 else
                   np.rint(np.linspace(0, meta["frames"]-1-step, count//2)).astype(int))
        self.frames = {int(f) for anchor in anchors for f in [anchor, anchor+step]}
        if count % 2:
            self.frames.add(min((f for f in range(meta["frames"]) if f not in self.frames), key=lambda f: abs(f-meta["frames"]//2)))
        self.checkpoint = file_identity(model)
        self.rows, self.classes = [], []
        self.calls = 0
        self.load_s = self.inference_s = self.total_s = 0.

    def __call__(self, bgr, frame: int, t: float) -> None:
        if frame not in self.frames:
            return
        started = time.perf_counter()
        if self.detector is None:
            begin = time.perf_counter()
            if self.factory is None:
                from highlight.objects import YoloObjects
                self.detector = YoloObjects(self.model, self.device)
            else:
                self.detector = self.factory()
            self.load_s += time.perf_counter()-begin
            self.classes = list(self.detector.model.names.values())
        x1, y1, x2, y2 = self.crop
        begin = time.perf_counter()
        found = self.detector.detect(np.ascontiguousarray(bgr[y1:y2, x1:x2]), [])
        self.inference_s += time.perf_counter()-begin
        self.calls += 1
        self.rows.append({"frame_index": frame, "time_s": t, "time_basis": "source_decoder", "crop": list(self.crop),
                          "objects": [{"label": obj.label, "confidence": float(obj.conf),
                                       "box": (np.asarray(obj.box, float)+[x1,y1,x1,y1]).tolist(),
                                       "observation_kind": "observed", "source_track_id": None} for obj in found]})
        self.total_s += time.perf_counter()-started

    def stream(self) -> dict:
        return {"model": {"name": "yolo_object", "checkpoint_id": digest(self.checkpoint), "checkpoint_identity": self.checkpoint,
                          "backend": str(self.detector.device) if self.detector else str(self.device or "not_loaded"),
                          "run_id": digest({"frames": sorted(self.frames), "crop": self.crop, "imgsz": 640, "quantize": 16})[:20],
                          "provenance": "fresh_local", "native_classes": self.classes}, "rows": self.rows}

    def metrics(self) -> dict:
        return {"object_calls": self.calls, "object_load_s": self.load_s, "object_inference_s": self.inference_s,
                "object_observe_s": self.total_s, "object_planned_frames": len(self.frames)}
