"""Person detection, 2D pose and tracking with YOLO11-pose, on the GPU."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

# COCO keypoint order used by YOLO pose models.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)  # fmt: skip

# Keypoint index pairs joined by a line when drawing a skeleton.
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),  # legs, hips
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # shoulders, arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # head
)  # fmt: skip

STRIDE = 32  # YOLO input height and width must be multiples of this


@dataclass
class FramePoses:
    """Everyone detected in one frame. Coordinates are in source-video pixels."""

    track_ids: np.ndarray  # (N,) int; -1 when the tracker hasn't matched the person to a track
    boxes: np.ndarray  # (N, 4) float xyxy
    scores: np.ndarray  # (N,) person confidence
    keypoints: np.ndarray  # (N, 17, 3) x, y, confidence


class PoseDetector:
    """Runs YOLO11-pose on GPU frames and attaches ByteTrack IDs, keeping state across calls.

    Frames stay on the GPU end to end: resizing and padding are done with torch, and the
    tensor goes straight to the model, skipping Ultralytics' CPU preprocessing. Use one
    detector per video, since the tracker carries state from frame to frame.

    Every detection at or above `conf` is returned, tracked or not. Ultralytics' own
    track mode drops detections whose track isn't confirmed yet, which loses players
    mid-jump: a fast vertical move breaks the motion prediction and starts a new track.
    """

    def __init__(
        self,
        model: str | Path = "yolo11s-pose.pt",
        input_width: int = 1920,
        device: str = "cuda:0",
        conf: float = 0.25,
        tracker: str = "bytetrack.yaml",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("PoseDetector needs a CUDA GPU, but PyTorch can't see one")
        if input_width % STRIDE:
            raise ValueError(f"input_width must be a multiple of {STRIDE}, got {input_width}")
        self.model = YOLO(str(model))
        self.input_width = input_width
        self.device = device
        self.conf = conf
        tracker_cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker)))
        self.tracker = TRACKER_MAP[tracker_cfg.tracker_type](args=tracker_cfg)
        # ByteTrack's second matching stage uses weak detections to keep tracks through blur and occlusion.
        self.detect_conf = min(conf, tracker_cfg.track_low_thresh)

    def __call__(self, rgb: torch.Tensor) -> FramePoses:
        x, scale = self._preprocess(rgb)
        result = self.model.predict(x, conf=self.detect_conf, device=self.device, quantize=16, verbose=False)[0]
        boxes = result.boxes.cpu().numpy()
        track_ids = np.full(len(boxes), -1, dtype=np.int32)
        tracks = self.tracker.update(boxes)  # rows: x1, y1, x2, y2, track_id, score, cls, detection index
        if len(tracks):
            track_ids[tracks[:, 7].astype(int)] = tracks[:, 4].astype(int)
        # Weak detections are only kept when they continue a track.
        keep = (boxes.conf >= self.conf) | (track_ids >= 0)
        return FramePoses(
            track_ids=track_ids[keep],
            boxes=boxes.xyxy[keep] / scale,
            scores=boxes.conf[keep],
            keypoints=_rescale_keypoints(result.keypoints.data.cpu().numpy()[keep], scale),
        )

    def _preprocess(self, rgb: torch.Tensor) -> tuple[torch.Tensor, float]:
        """(H, W, 3) uint8 -> (1, 3, h, w) float in [0, 1], width = input_width, height padded to STRIDE."""
        height, width = rgb.shape[:2]
        x = rgb.permute(2, 0, 1).unsqueeze(0).float().div_(255)
        scale = self.input_width / width
        new_height = round(height * scale)
        if scale != 1:
            x = F.interpolate(x, size=(new_height, self.input_width), mode="bilinear", antialias=True)
        # Pad only the bottom, so coordinates map back with the scale alone.
        x = F.pad(x, (0, 0, 0, -new_height % STRIDE), value=114 / 255)
        return x, scale


def _rescale_keypoints(keypoints: np.ndarray, scale: float) -> np.ndarray:
    keypoints = keypoints.copy()
    keypoints[..., :2] /= scale
    return keypoints


def save_poses(path: str | Path, sampled: list[tuple[int, float, FramePoses]], meta: dict) -> None:
    """Save poses as one row per detected person per frame.

    `sampled` holds (frame_index, time_s, poses) for every frame pose ran on, including
    frames where nobody was found; those are kept in sampled_frame_index/sampled_time_s.
    """
    rows = [(i, t, p) for i, t, p in sampled for _ in range(len(p.track_ids))]
    np.savez_compressed(
        path,
        sampled_frame_index=np.array([i for i, _, _ in sampled], dtype=np.int32),
        sampled_time_s=np.array([t for _, t, _ in sampled], dtype=np.float64),
        frame_index=np.array([i for i, _, _ in rows], dtype=np.int32),
        time_s=np.array([t for _, t, _ in rows], dtype=np.float64),
        track_id=_concat([p.track_ids for _, _, p in sampled], (0,), np.int32),
        box=_concat([p.boxes for _, _, p in sampled], (0, 4), np.float32),
        score=_concat([p.scores for _, _, p in sampled], (0,), np.float32),
        keypoints=_concat([p.keypoints for _, _, p in sampled], (0, 17, 3), np.float32),
        meta=json.dumps({**meta, "keypoint_names": KEYPOINT_NAMES}),
    )


def load_poses(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"]))
    return arrays, meta


def poses_by_frame(arrays: dict[str, np.ndarray]) -> dict[int, FramePoses]:
    """Regroup a `load_poses` result into one FramePoses per sampled frame index."""
    by_frame = {}
    for frame in arrays["sampled_frame_index"]:
        rows = arrays["frame_index"] == frame
        by_frame[int(frame)] = FramePoses(
            arrays["track_id"][rows], arrays["box"][rows], arrays["score"][rows], arrays["keypoints"][rows]
        )
    return by_frame


def _concat(parts: list[np.ndarray], empty_shape: tuple[int, ...], dtype) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(empty_shape, dtype=dtype)
