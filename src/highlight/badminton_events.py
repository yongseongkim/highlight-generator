"""Rank pose-based badminton highlight candidates, with explicit work budgets.

These are explainable motion heuristics, not smash/winner or rally-end classifiers.
The detector deliberately retains candidates when a tiny racket/shuttle is missed
by a later detail model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from highlight.candidates import Window
from highlight.court import Court
from highlight.signals import torso_length


@dataclass
class Moment:
    window: Window
    peak_frame: int
    track_id: int
    reason: str
    overhead: float
    wrist_speed: float
    body_speed: float


def player_foot(keypoints: np.ndarray, box: np.ndarray) -> np.ndarray:
    ankles = keypoints[[15, 16]]
    visible = ankles[:, 2] >= 0.5
    return ankles[visible, :2].mean(axis=0) if visible.any() else np.array([(box[0] + box[2]) / 2, box[3]])


def rank_moments(poses: dict[str, np.ndarray], court: Court, duration_s: float,
                 max_highlights: int = 30, budget_s: float = 120, threshold: float = 0.5,
                 pre_s: float = 3, post_s: float = 3) -> tuple[list[Moment], dict]:
    """Select non-overlapping, bounded windows around motion peaks inside the court."""
    times = poses["sampled_time_s"]
    frames = poses["sampled_frame_index"]
    if max_highlights < 0 or budget_s < 0 or threshold < 0 or pre_s < 0 or post_s < 0 or pre_s + post_s <= 0:
        raise ValueError("Highlight budgets/threshold must be nonnegative and window duration positive")
    rows_by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(poses["frame_index"]):
        rows_by_frame.setdefault(int(frame), []).append(row)
    previous = {}
    peaks = []
    trace = []
    kept_rows = excluded_rows = 0
    for frame, time_s in zip(frames, times):
        best = None
        for row in rows_by_frame.get(int(frame), []):
            keypoints, box = poses["keypoints"][row], poses["box"][row]
            if not np.isfinite(box).all() or not np.isfinite(keypoints).all():
                continue
            if not court.contains(player_foot(keypoints, box)):
                excluded_rows += 1
                continue
            kept_rows += 1
            torso = torso_length(keypoints)
            if torso is None:
                continue
            track_id = int(poses["track_id"][row])
            old = previous.get(track_id) if track_id >= 0 else None
            shoulder_y = keypoints[[5, 6], 1].mean()
            wrists = keypoints[[9, 10]]
            visible = wrists[:, 2] >= 0.5
            overhead = max(0.0, float((shoulder_y - wrists[visible, 1].min()) / torso)) if visible.any() else 0.0
            wrist_speed = body_speed = 0.0
            if old is not None:
                old_time, old_kpts, old_torso = old
                dt = float(time_s - old_time)
                if 0 < dt <= 0.4:
                    scale = (torso + old_torso) / 2
                    common = visible & (old_kpts[[9, 10], 2] >= 0.5)
                    if common.any():
                        velocity = np.linalg.norm(wrists[common, :2] - old_kpts[[9, 10]][common, :2], axis=1)
                        wrist_speed = float(velocity.max() / scale / dt)
                    hips = keypoints[[11, 12], :2].mean(axis=0)
                    old_hips = old_kpts[[11, 12], :2].mean(axis=0)
                    body_speed = float(np.linalg.norm(hips - old_hips) / scale / dt)
            if track_id >= 0:
                previous[track_id] = (float(time_s), keypoints, torso)
            # A raised arm by itself is not a stroke. Require temporal movement;
            # Compress large velocities smoothly, preserving ranking differences
            # instead of tying many different motions at exactly 1.0.
            movement = -math.expm1(-max(wrist_speed - 0.5, 0) / 3)
            overhead_motion = -math.expm1(-overhead / 0.6) * movement
            travel = -math.expm1(-max(body_speed - 0.3, 0) / 1.5)
            score = max(0.7 * overhead_motion + 0.2 * movement + 0.1 * travel,
                        0.55 * travel + 0.45 * movement)
            if best is None or score > best[0]:
                best = (score, row, track_id, overhead, wrist_speed, body_speed)
        score = best[0] if best is not None else 0.0
        trace.append({"time_s": float(time_s), "score": float(score)})
        if best is not None and score >= threshold:
            peaks.append((float(time_s), best))

    selected: list[Moment] = []
    selected_seconds = 0.0
    # Give active portions of the whole recording a chance before taking several
    # events from one portion. Empty/quiet bins never create synthetic candidates.
    affordable = min(max_highlights, max(1, math.ceil(budget_s / (pre_s + post_s))))
    bin_count = max(1, min(affordable, math.ceil(duration_s / 30)))
    buckets = [[] for _ in range(bin_count)]
    for peak in sorted(peaks, key=lambda p: (-p[1][0], p[0])):
        index = min(bin_count - 1, max(0, int(peak[0] / max(duration_s, 1e-9) * bin_count)))
        buckets[index].append(peak)
    while len(selected) < max_highlights:
        added = False
        for bucket in sorted((b for b in buckets if b), key=lambda b: -b[0][1][0]):
            while bucket:
                time_s, (score, row, track_id, overhead, wrist_speed, body_speed) = bucket.pop(0)
                start, end = max(0.0, time_s - pre_s), min(duration_s, time_s + post_s)
                if end <= start or selected_seconds + end - start > budget_s + 1e-9:
                    continue
                if any(start < other.window.end_s and end > other.window.start_s for other in selected):
                    continue
                window = Window(start, end, time_s, float(score), poses["box"][row].astype(float).tolist())
                selected.append(Moment(window, int(poses["frame_index"][row]), track_id,
                                       "overhead_motion_candidate" if overhead >= 0.3 else "movement_candidate",
                                       float(overhead), float(wrist_speed), float(body_speed)))
                selected_seconds += end - start
                added = True
                break
            if len(selected) >= max_highlights:
                break
        if not added:
            break
    selected.sort(key=lambda moment: moment.window.start_s)
    return selected, {"candidate_samples": len(peaks), "selected_moments": len(selected),
                      "selected_video_s": selected_seconds, "court_player_rows": kept_rows,
                      "excluded_player_rows": excluded_rows, "temporal_bins": bin_count, "trace": trace}


def keyframe_plan(moments: list[Moment], fps: float, frame_count: int, per_moment: int = 3,
                  max_calls: int = 90) -> list[tuple[int, int]]:
    """(moment index, source frame index), globally bounded and unique per moment.

    Allocate peak frames to the highest-ranked moments before giving any moment
    extra frames. A small call budget therefore covers several moments fairly.
    """
    if fps <= 0 or frame_count < 0 or not 1 <= per_moment <= 9 or max_calls < 0:
        raise ValueError("Invalid keyframe settings")
    if not frame_count or not moments or not max_calls:
        return []
    offsets = [0.0] + sorted(np.linspace(-0.3, 0.3, per_moment if per_moment % 2 else per_moment + 1), key=abs)
    offsets = list(dict.fromkeys(float(x) for x in offsets))[:per_moment]
    ranked = sorted(range(len(moments)), key=lambda i: -moments[i].window.peak_score)
    plan = []
    seen = set()
    for offset in offsets:
        for index in ranked:
            window = moments[index].window
            time_s = np.clip(window.peak_s + offset, window.start_s, window.end_s)
            frame = int(np.clip(moments[index].peak_frame + round((time_s - window.peak_s) * fps), 0, frame_count - 1))
            if (index, frame) not in seen:
                plan.append((index, frame))
                seen.add((index, frame))
            if len(plan) >= max_calls:
                return sorted(plan, key=lambda item: item[1])
    return sorted(plan, key=lambda item: item[1])
