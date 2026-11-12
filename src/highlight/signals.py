"""Per-frame pose cues that flag candidate highlight moments.

Each cue scores one person from their keypoints, in units of their torso length so near
and far players are comparable. A frame's score is its top-scoring person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KPT_CONF = 0.5  # keypoints below this confidence are treated as not visible
MIN_TORSO_PX = 15  # smaller people have keypoints too noisy to trust

FACE = (0, 1, 2, 3, 4)  # nose, eyes, ears
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


def torso_length(kpts: np.ndarray) -> float | None:
    """Shoulder-center to hip-center distance in pixels, or None if not measurable."""
    visible = kpts[:, 2] >= KPT_CONF
    if not visible[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all():
        return None
    shoulders = (kpts[L_SHOULDER, :2] + kpts[R_SHOULDER, :2]) / 2
    hips = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2
    torso = float(np.linalg.norm(shoulders - hips))
    return torso if torso >= MIN_TORSO_PX else None


def arms_up(kpts: np.ndarray) -> float:
    """How far the higher wrist is above the head; > 0 means above. Shots, rebounds, blocks."""
    torso = torso_length(kpts)
    visible = kpts[:, 2] >= KPT_CONF
    wrists = [kpts[i, 1] for i in (L_WRIST, R_WRIST) if visible[i]]
    if torso is None or not wrists:
        return np.nan
    face = [kpts[i, 1] for i in FACE if visible[i]]
    head_y = min(face) if face else (kpts[L_SHOULDER, 1] + kpts[R_SHOULDER, 1]) / 2 - 0.5 * torso
    return (head_y - min(wrists)) / torso  # image y grows downward


def foot_up(kpts: np.ndarray) -> float:
    """Height difference between the ankles; high while kicking."""
    torso = torso_length(kpts)
    visible = kpts[:, 2] >= KPT_CONF
    if torso is None or not (visible[L_ANKLE] and visible[R_ANKLE]):
        return np.nan
    return abs(kpts[L_ANKLE, 1] - kpts[R_ANKLE, 1]) / torso


CUES = {"arms_up": arms_up, "foot_up": foot_up}


@dataclass
class CueSeries:
    """One cue over every frame pose ran on."""

    time_s: np.ndarray  # (F,)
    score: np.ndarray  # (F,) top person's score; nan when nobody was measurable
    box: np.ndarray  # (F, 4) xyxy box of the top person; nan when nobody was measurable


def cue_series(poses: dict[str, np.ndarray], cue: str) -> CueSeries:
    """Score every sampled frame of a `pose.load_poses` result with one cue."""
    score_fn = CUES[cue]
    frames = poses["sampled_frame_index"]
    scores = np.full(len(frames), np.nan)
    boxes = np.full((len(frames), 4), np.nan)
    row_scores = np.array([score_fn(k) for k in poses["keypoints"]])
    for i, frame in enumerate(frames):
        rows = np.flatnonzero((poses["frame_index"] == frame) & ~np.isnan(row_scores))
        if len(rows):
            best = rows[np.argmax(row_scores[rows])]
            scores[i] = row_scores[best]
            boxes[i] = poses["box"][best]
    return CueSeries(time_s=poses["sampled_time_s"], score=scores, box=boxes)
