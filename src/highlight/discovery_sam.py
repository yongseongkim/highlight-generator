"""Budgeted SAM execution and exact-input inference cache; no pose gate."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from highlight.pose import file_identity


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def detail_source(detail: dict) -> str:
    identity = detail.get("request_id", digest([detail["frame_index"], detail["crop"]])[:20])
    return f"sam-discovery-{identity}"


def validate_detail(detail: dict, meta: dict) -> None:
    frame = detail["frame_index"]
    crop = np.asarray(detail["crop"], dtype=float)
    if not isinstance(frame, int) or not 0 <= frame < meta["frames"] or crop.shape != (4,):
        raise ValueError("Invalid SAM source frame/crop")
    if (not np.isfinite(crop).all() or (crop != np.rint(crop)).any() or (crop[:2] < 0).any()
            or (crop[2:] > [meta["width"], meta["height"]]).any() or (crop[2:] <= crop[:2]).any()):
        raise ValueError("SAM crop is outside the source image")
    t = detail.get("decoder_time_s", frame/meta["fps"])
    if not math.isfinite(t) or t < 0 or t > meta["frames"]/meta["fps"] + 1:
        raise ValueError("Invalid SAM decoder time")
    for obj in detail["objects"]:
        box = np.asarray(obj["box"], dtype=float)
        if (not isinstance(obj["label"], str) or not obj["label"] or box.shape != (4,) or not np.isfinite(box).all()
                or not math.isfinite(obj["confidence"]) or not 0 <= obj["confidence"] <= 1
                or (box[2:] <= box[:2]).any() or (box[:2] < crop[:2]-1).any() or (box[2:] > crop[2:]+1).any()):
            raise ValueError("Invalid SAM raw object")


class SamExecutor:
    def __init__(self, video: Path, meta: dict, model: Path, prompts: list[str], budget: int,
                 cache: Path, device: str | None = None, detector_factory=None):
        self.video, self.meta, self.model, self.prompts = video, meta, model, prompts
        self.budget, self.cache, self.device = budget, cache, device
        self.factory, self.detector = detector_factory, None
        self.identity = {"schema_version": 1, "source_identity": meta["source_identity"],
                         "model_identity": file_identity(model), "prompts": prompts, "imgsz": 644, "confidence": .25}
        self.entries, self.requests, self.details = {}, [], []
        self.calls = self.hits = 0
        self.load_s = self.inference_s = self.decode_s = 0.
        self.by_branch = Counter()
        if cache.exists():
            for line in cache.read_text().splitlines():
                entry = json.loads(line)
                if entry["key"] != digest(entry["input"]):
                    raise ValueError("SAM cache key/input mismatch")
                # A shared cache may contain other sources/checkpoints. They never hit this identity.
                if all(entry["input"].get(k) == v for k, v in self.identity.items()):
                    validate_detail(entry["detail"], meta)
                    if (entry["detail"]["frame_index"] != entry["input"]["frame_index"]
                            or entry["detail"]["crop"] != entry["input"]["crop"]):
                        raise ValueError("SAM cache result/input geometry mismatch")
                    self.entries[entry["key"]] = entry["detail"]

    @property
    def remaining(self):
        return self.budget - self.calls

    def execute(self, probes: list) -> list[dict]:
        results = []
        cap = None
        try:
            for probe in probes:
                requested = {**self.identity, "frame_index": probe.frame_index, "crop": list(probe.crop)}
                key = digest(requested)
                record = {**asdict(probe), "request_id": key[:20]}
                if key in self.entries:
                    detail = dict(self.entries[key])
                    status = "cached"
                    self.hits += 1
                elif not self.remaining:
                    record["status"] = "budget_exhausted"
                    self.requests.append(record)
                    continue
                else:
                    if self.detector is None:
                        started = time.perf_counter()
                        if self.factory is None:
                            from highlight.sam import Sam3
                            self.detector = Sam3(self.model, device=self.device)
                        else:
                            self.detector = self.factory()
                        self.load_s += time.perf_counter() - started
                    if cap is None:
                        cap = cv2.VideoCapture(str(self.video))
                    started = time.perf_counter()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, probe.frame_index)
                    ok, bgr = cap.read()
                    if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1 != probe.frame_index:
                        raise RuntimeError(f"Cannot decode exact SAM frame {probe.frame_index}")
                    decoder_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    x1, y1, x2, y2 = probe.crop
                    self.decode_s += time.perf_counter() - started
                    started = time.perf_counter()
                    self.calls += 1
                    self.by_branch[probe.branch] += 1
                    objects = self.detector.detect(np.ascontiguousarray(bgr[y1:y2, x1:x2]), self.prompts)
                    self.inference_s += time.perf_counter() - started
                    detail = {"frame_index": probe.frame_index, "decoder_time_s": decoder_time,
                              "crop": list(probe.crop), "objects": [{"label": o.label, "confidence": float(o.conf),
                              "box": (np.asarray(o.box, float)+[x1,y1,x1,y1]).tolist()} for o in objects]}
                    validate_detail(detail, self.meta)
                    self.cache.parent.mkdir(parents=True, exist_ok=True)
                    with self.cache.open("a") as output:
                        output.write(json.dumps({"key": key, "input": requested, "detail": detail}, allow_nan=False)+"\n")
                    self.entries[key] = detail
                    status = "fresh"
                record["status"] = status
                detail = {**detail, "request_id": key[:20], "branch": probe.branch, "reason": probe.reason,
                          "cache_status": status}
                self.requests.append(record)
                self.details.append(detail)
                results.append(detail)
        finally:
            if cap is not None:
                cap.release()
        return results

    def metrics(self) -> dict:
        return {"sam_calls": self.calls, "sam_cache_hits": self.hits, "new_calls_by_branch": dict(self.by_branch),
                "sam_load_s": self.load_s, "sam_inference_s": self.inference_s, "sam_decode_s": self.decode_s}
