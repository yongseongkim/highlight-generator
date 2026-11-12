"""Temporal badminton pose patterns and conservative SAM object evidence.

These rules describe observable sequences, not learned highlight preferences or
validated stroke/outcome labels. All coordinates remain in source-image pixels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math

import numpy as np

from highlight.badminton_events import player_foot
from highlight.court import Court


@dataclass
class Action:
    pattern: str
    track_id: int
    time_s: float
    frame_index: int
    start_s: float
    end_s: float
    pose_score: float
    box: list[float]
    features: dict
    phases: dict[str, float]
    sam: dict = field(default_factory=dict)

    @property
    def ranking_score(self) -> float:
        return self.pose_score + self.sam.get("ranking_bonus", 0.0)


def _median3(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    padded = np.pad(values, [(1, 1)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    return np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a - b, c - b
    denominator = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    cosine = np.sum(u * v, axis=-1) / np.maximum(denominator, 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def _segments(poses: dict, court: Court) -> tuple[list[np.ndarray], dict]:
    k = poses["keypoints"]
    boxes = poses["box"]
    torso = np.linalg.norm(k[:, [5, 6], :2].mean(axis=1) - k[:, [11, 12], :2].mean(axis=1), axis=1)
    valid = ((poses["track_id"] >= 0) & (poses["score"] >= 0.35) & (torso >= 20)
             & np.isfinite(k).all(axis=(1, 2)) & np.isfinite(boxes).all(axis=1)
             & (k[:, [5, 6, 11, 12], 2] >= 0.5).all(axis=1)
             & (boxes[:, 3] - boxes[:, 1] >= 80))
    court_excluded = 0
    if court.polygon is not None:
        for row in np.flatnonzero(valid):
            if not court.contains(player_foot(k[row], boxes[row]), margin=0):
                valid[row] = False
                court_excluded += 1
    grouped = defaultdict(list)
    for row in np.flatnonzero(valid):
        grouped[int(poses["track_id"][row])].append(int(row))
    segments = []
    discontinuities = 0
    sampled_dt = np.diff(poses["sampled_time_s"])
    nominal_dt = float(np.median(sampled_dt[sampled_dt > 0])) if (sampled_dt > 0).any() else 0.1
    gap_limit = max(0.18, min(0.35, nominal_dt * 2.5))
    for rows in grouped.values():
        rows.sort(key=lambda row: poses["time_s"][row])
        current = []
        for row in rows:
            if current:
                old = current[-1]
                dt = float(poses["time_s"][row] - poses["time_s"][old])
                scale = (torso[row] + torso[old]) / 2
                hip_shift = np.linalg.norm(k[row, [11, 12], :2].mean(axis=0) - k[old, [11, 12], :2].mean(axis=0)) / scale
                stable = (0 < dt <= gap_limit and 0.6 <= torso[row] / torso[old] <= 1 / 0.6
                          and hip_shift <= max(0.6, 6 * dt))
                if not stable:
                    if len(current) >= 8 and poses["time_s"][current[-1]] - poses["time_s"][current[0]] >= 0.7:
                        segments.append(np.asarray(current))
                    discontinuities += 1
                    current = []
            current.append(row)
        if len(current) >= 8 and poses["time_s"][current[-1]] - poses["time_s"][current[0]] >= 0.7:
            segments.append(np.asarray(current))
    return segments, {"pose_rows": len(k), "usable_pose_rows": int(valid.sum()), "court_excluded_rows": court_excluded,
                      "court_filter_active": court.polygon is not None, "net_configured": court.net is not None,
                      "continuous_segments": len(segments), "track_discontinuities": discontinuities,
                      "sample_interval_s": nominal_dt}


def _peaks(values: np.ndarray, times: np.ndarray, threshold: float, gap_s: float = 0.15) -> list[int]:
    candidates = [i for i in range(1, len(values) - 1)
                  if np.isfinite(values[i]) and threshold <= values[i] <= 20
                  and values[i] >= values[i - 1] and values[i] >= values[i + 1]]
    kept = []
    for i in sorted(candidates, key=lambda index: -values[index]):
        if all(abs(times[i] - times[old]) >= gap_s for old in kept):
            kept.append(i)
    return sorted(kept)


def mine_actions(poses: dict, court: Court, duration_s: float) -> tuple[list[Action], dict]:
    """Match ordered preparation/movement/recovery phases within continuous tracks."""
    segments, diagnostics = _segments(poses, court)
    actions = []
    for rows in segments:
        times = poses["time_s"][rows].astype(float)
        raw = poses["keypoints"][rows].astype(float)
        xy = _median3(raw[..., :2])
        shoulders = xy[:, [5, 6]].mean(axis=1)
        hips = xy[:, [11, 12]].mean(axis=1)
        scale = np.linalg.norm(shoulders - hips, axis=1)
        scale = np.maximum(_median3(scale), 20)
        quality = np.mean(raw[:, [5, 6, 11, 12], 2], axis=1) * poses["score"][rows]
        valid_ankles = (raw[:, [15, 16], 2] >= 0.5).all(axis=1)
        spread = np.linalg.norm(xy[:, 15] - xy[:, 16], axis=1) / scale
        knees = np.minimum(_angle(xy[:, 11], xy[:, 13], xy[:, 15]), _angle(xy[:, 12], xy[:, 14], xy[:, 16]))
        valid_knees = (raw[:, [13, 14, 15, 16], 2] >= 0.5).all(axis=1)
        for side, (shoulder, wrist) in enumerate(((5, 9), (6, 10))):
            arm = (xy[:, wrist] - xy[:, shoulder]) / scale[:, None]
            visible = raw[:, wrist, 2] >= 0.5
            speed = np.linalg.norm(np.gradient(arm, times, axis=0), axis=1)
            absolute_speed = np.linalg.norm(np.gradient(xy[:, wrist], times, axis=0), axis=1) / scale
            # No velocity evidence when a wrist reappears after an occlusion.
            speed[~(visible & np.roll(visible, 1) & np.roll(visible, -1))] = 0
            height = -arm[:, 1]
            for i in _peaks(speed, times, 1.8):
                before = np.flatnonzero((times >= times[i] - 0.7) & (times <= times[i] - 0.05) & visible)
                after = np.flatnonzero((times >= times[i] + 0.05) & (times <= times[i] + 0.65) & visible)
                if len(before) < 2 or len(after) < 2 or times[i] - times[0] < 0.2 or times[-1] - times[i] < 0.2:
                    continue
                preparation = before[np.argmax(height[before])]
                follow = after[np.argmin(height[after])]
                raised = float(height[preparation])
                drop = float(raised - height[follow])
                travel = float(np.linalg.norm(arm[follow] - arm[preparation]))
                q = float(np.median(quality[np.r_[before, i, after]]))
                if raised >= 0.5 and drop >= 0.35 and travel >= 0.55 and q >= 0.4:
                    score = (0.4 * q + 0.25 * min(raised / 1.2, 1)
                             + 0.25 * min(drop / 1.2, 1) + 0.1 * min(speed[i] / 8, 1))
                    row = rows[i]
                    actions.append(Action("overhead_swing", int(poses["track_id"][row]), float(times[i]),
                        int(poses["frame_index"][row]), max(0.0, float(times[preparation]) - 1.5),
                        min(duration_s, float(times[follow]) + 1.8), float(score), poses["box"][row].astype(float).tolist(),
                        {"hand": "left" if side == 0 else "right", "raised_torso_lengths": raised,
                         "follow_through_drop": drop, "arm_travel": travel, "wrist_speed": float(speed[i]), "pose_quality": q},
                        {"preparation_s": float(times[preparation]), "swing_s": float(times[i]), "follow_through_s": float(times[follow])}))
                # Fast low reach with bent/wide legs followed by recovery. This
                # deliberately does not call a floor pickup a confirmed defense.
                depth = (xy[:, wrist, 1] - hips[:, 1]) / scale
                reach = np.flatnonzero((times >= times[i]) & (times <= times[i] + 0.25) & visible
                                       & valid_ankles & valid_knees & (depth >= 0.1) & (spread >= 1.0) & (knees <= 145))
                if not len(reach):
                    continue
                j = reach[np.argmax(depth[reach])]
                low = float(depth[j])
                recovery = np.flatnonzero((times >= times[j] + 0.1) & (times <= times[j] + 0.9) & valid_ankles)
                # Leg closure alone also describes bending down to collect a
                # shuttle. Require the hand to return toward a ready position.
                recovered_hand = (xy[recovery, wrist, 1] - hips[recovery, 1]) / scale[recovery]
                recovery = recovery[(recovered_hand <= min(0.7, low - 0.3)) & visible[recovery]]
                if len(recovery) < 2 or q < 0.4:
                    continue
                best = recovery[np.argmin(spread[recovery])]
                closes = float(spread[j] - spread[best])
                rises = float((hips[j, 1] - hips[best, 1]) / scale[j])
                if closes < 0.3 and rises < 0.15:
                    continue
                # There must also be a transition into the reach, not a held squat.
                opens = float(spread[j] - np.median(spread[before]))
                lowers = float((hips[j, 1] - np.median(hips[before, 1])) / scale[j])
                if opens < 0.2 and lowers < 0.15:
                    continue
                score = 0.4 * q + 0.3 * min(max(closes, rises) / 0.8, 1) + 0.2 * min(speed[i] / 6, 1) + 0.1
                row = rows[j]
                actions.append(Action("low_reach_recovery", int(poses["track_id"][row]), float(times[j]),
                    int(poses["frame_index"][row]), max(0.0, float(times[before[0]]) - 1),
                    min(duration_s, float(times[best]) + 1.5), float(score), poses["box"][row].astype(float).tolist(),
                    {"hand": "left" if side == 0 else "right", "ankle_spread": float(spread[j]),
                     "knee_angle_degrees": float(knees[j]), "recovery_closure": closes, "recovery_rise": rises,
                     "hand_recovery": float(low - (xy[best, wrist, 1] - hips[best, 1]) / scale[best]),
                     "wrist_speed": float(speed[i]), "pose_quality": q},
                    {"ready_s": float(times[before[0]]), "reach_s": float(times[j]), "recovery_s": float(times[best])}))
            # Compact strokes are independent of overhead/low-reach gates. A
            # gentle touch can have small arm travel and little or no footwork.
            for i in _peaks(speed, times, 0.45, gap_s=0.2):
                before = np.flatnonzero((times >= times[i] - 0.4) & (times <= times[i] - 0.08) & visible)
                touch = np.flatnonzero((times >= times[i] - 0.02) & (times <= times[i] + 0.25) & visible)
                if len(before) < 2 or not len(touch) or speed[i] > 8 or absolute_speed[i] < 0.25:
                    continue
                ready = before[np.argmin(speed[before])]
                distance = np.linalg.norm(arm[touch] - arm[ready], axis=1)
                j = touch[np.argmax(distance)]
                excursion = float(np.max(distance))
                if not 0.15 <= excursion <= 1.2:
                    continue
                after = np.flatnonzero((times >= times[j] + 0.1) & (times <= times[j] + 0.7) & visible)
                if len(after) < 2:
                    continue
                if np.max(height[np.r_[before, touch, after]]) > 0.5:
                    continue
                recovered = np.linalg.norm(arm[after] - arm[ready], axis=1)
                settling = speed[after] <= max(0.25, 0.45 * speed[i])
                returning = recovered <= 0.65 * excursion
                possible = after[settling | returning]
                if not len(possible):
                    continue
                end = possible[np.argmin(speed[possible])]
                foot = player_foot(raw[j], poses["box"][rows[j]])
                # Floor pickups are not promoted to a compact-stroke label.
                if xy[j, wrist, 1] >= foot[1] - 0.2 * scale[j]:
                    continue
                q = float(np.median(quality[np.r_[before, j, after]]))
                if q < 0.5:
                    continue
                net_distance = court.net_distance(foot, scale[j])
                near_net = net_distance is not None and net_distance <= 1.8
                recovery_fraction = float(np.clip(1 - np.linalg.norm(arm[end] - arm[ready]) / excursion, 0, 1))
                control = min(1.0, excursion / 0.45)
                score = 0.45 * q + 0.25 * control + 0.2 * max(recovery_fraction, float(settling[after == end][0])) + 0.1
                row = rows[j]
                actions.append(Action("net_touch_candidate" if near_net else "compact_stroke", int(poses["track_id"][row]),
                    float(times[j]), int(poses["frame_index"][row]), max(0.0, float(times[ready]) - 1.2),
                    min(duration_s, float(times[end]) + 1.5), float(score), poses["box"][row].astype(float).tolist(),
                    {"hand": "left" if side == 0 else "right", "arm_excursion": excursion,
                     "wrist_speed": float(speed[i]), "return_fraction": recovery_fraction, "pose_quality": q,
                     "net_ground_distance_torso": net_distance, "overhead_required": False,
                     "interpretation": "Compact action near the net is a net-play candidate, not a confirmed hairpin."},
                    {"ready_s": float(times[ready]), "touch_s": float(times[j]), "recovery_s": float(times[end])}))
    # Both hands may explain the same stroke. Keep one sequence per track/time.
    unique = []
    # A small follow-up movement must not replace an already matched full swing.
    for action in sorted(actions, key=lambda action: (action.pattern != "overhead_swing", -action.pose_score)):
        if not any(action.track_id == old.track_id and abs(action.time_s - old.time_s) < 0.65 for old in unique):
            unique.append(action)
    unique.sort(key=lambda action: action.time_s)
    diagnostics["pattern_counts"] = dict(Counter(action.pattern for action in unique))
    return unique, diagnostics


def action_bursts(actions: list[Action], max_span_s: float = 10, max_gap_s: float = 3) -> list[Action]:
    """Repeated multi-track actions are rally-like candidates, not confirmed rallies.

    A court boundary is needed to rule out coincident actions on adjacent courts.
    The anchor track supplies the preview/SAM crop, not every player in the burst.
    """
    ordered = sorted((a for a in actions if a.pattern != "repeated_action_burst"), key=lambda a: a.time_s)
    proposals = []
    for end in range(2, len(ordered)):
        group = [ordered[end]]
        for index in range(end - 1, -1, -1):
            candidate = ordered[index]
            if group[-1].time_s - candidate.time_s > max_gap_s or ordered[end].time_s - candidate.time_s > max_span_s:
                break
            group.append(candidate)
        group.reverse()
        if len(group) < 3 or len({a.track_id for a in group}) < 2 or group[-1].time_s - group[0].time_s < 2:
            continue
        anchor = max(group, key=lambda a: a.pose_score)
        score = 0.8 * float(np.mean([a.pose_score for a in group])) + 0.2 * min(len(group) / 5, 1)
        proposals.append(Action("repeated_action_burst", anchor.track_id, anchor.time_s, anchor.frame_index,
                                group[0].start_s, group[-1].end_s, score, list(anchor.box),
                                {"action_count": len(group), "distinct_track_ids": len({a.track_id for a in group}),
                                 "action_times_s": [a.time_s for a in group], "sam_scope": "anchor_player_only"},
                                {"first_action_s": group[0].time_s, "anchor_s": anchor.time_s,
                                 "last_action_s": group[-1].time_s}))
    kept = []
    for action in sorted(proposals, key=lambda a: -a.pose_score):
        if not any(action.start_s < old.end_s and action.end_s > old.start_s for old in kept):
            kept.append(action)
    return sorted(kept, key=lambda a: a.time_s)


def _point_box_distance(point: np.ndarray, box: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(box[:2] - point, point - box[2:]), 0)))


def sam_support(action: Action, poses: dict, details: list[dict], court: Court | None = None) -> dict:
    """Use hand proximity, object size, and repeated-frame stationarity checks.

    Evidence is attached by source frame and geometry, never by IDs from a different
    pose archive. A missed/tiny shuttle cannot prove a failed shot or lack of action.
    """
    rows = np.flatnonzero(poses["track_id"] == action.track_id)
    relevant = []
    for detail in details:
        if not len(rows):
            break
        frame = detail["frame_index"]
        row = rows[np.argmin(abs(poses["frame_index"][rows] - frame))]
        if court is not None and not court.contains(player_foot(poses["keypoints"][row], poses["box"][row]), margin=0):
            continue
        t = float(poses["time_s"][row])
        if abs(t - action.time_s) > 0.7 or abs(int(poses["frame_index"][row]) - frame) > 6:
            continue
        if detail.get("crop") is not None:
            crop, box = np.asarray(detail["crop"]), poses["box"][row]
            overlap = np.maximum(np.minimum(crop[2:], box[2:]) - np.maximum(crop[:2], box[:2]), 0)
            if np.prod(overlap) < 0.25 * np.prod(box[2:] - box[:2]):
                continue
        relevant.append((detail, row, t))
    racket = []
    shuttles = []
    rejected = Counter()
    for detail, row, t in relevant:
        k = poses["keypoints"][row]
        scale = max(20.0, float(np.linalg.norm(k[[5, 6], :2].mean(axis=0) - k[[11, 12], :2].mean(axis=0))))
        wrists = k[[9, 10]]
        wrists = wrists[wrists[:, 2] >= 0.5, :2]
        if not len(wrists):
            continue
        feet = player_foot(k, poses["box"][row])
        for obj in detail.get("objects", []):
            box = np.asarray(obj["box"], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(obj["confidence"]):
                rejected["invalid"] += 1
                continue
            size = box[2:] - box[:2]
            if min(size) <= 0 or obj["confidence"] < 0.4:
                rejected["low_confidence_or_size"] += 1
                continue
            label = obj["label"].lower()
            center = (box[:2] + box[2:]) / 2
            if "racket" in label:
                close = min(_point_box_distance(wrist, box) for wrist in wrists) <= 0.25 * scale
                plausible = max(size) <= 2.2 * scale and np.prod(size) <= 1.8 * scale**2
                if close and plausible:
                    racket.append({**obj, "frame_index": detail["frame_index"]})
                else:
                    rejected["racket_not_near_hand_or_oversized"] += 1
            elif "shuttle" in label:
                if max(size) > 0.5 * scale or obj["confidence"] < 0.5:
                    rejected["shuttle_size_or_confidence"] += 1
                elif center[1] >= feet[1] - 0.1 * scale:
                    rejected["shuttle_near_floor"] += 1
                elif min(np.linalg.norm(center - wrist) for wrist in wrists) > 2 * scale:
                    rejected["shuttle_far_from_hand"] += 1
                else:
                    shuttles.append((obj, detail["frame_index"], t, center, scale))
    moving = []
    for obj, frame, t, center, scale in shuttles:
        others = [other for other in shuttles if other[1] != frame and abs(other[2] - t) >= 0.1]
        if any(np.linalg.norm(center - other[3]) < max(5, 0.12 * scale) for other in others):
            rejected["stationary_shuttle"] += 1
            continue
        if any(max(5, 0.15 * scale) <= np.linalg.norm(center - other[3]) <= 40 * scale * abs(other[2] - t) for other in others):
            moving.append({**obj, "frame_index": frame})
    unique_frames = len({detail["frame_index"] for detail, _, _ in relevant})
    return {"sampled_frames": unique_frames, "racket_near_hand": len(racket), "moving_small_object_samples": len(moving),
            "rejected": dict(rejected), "accepted_objects": racket + moving,
            "ranking_bonus": (0.07 if racket else 0.0) + (0.03 if moving else 0.0),
            "status": ("racket_near_hand" if racket else "moving_small_object" if moving else
                       "sampled_without_support" if unique_frames else "not_sampled"),
            "interpretation": "Supporting object evidence only; this does not confirm shuttle contact, shot type, or outcome."}


def highlight_candidates(actions: list[Action], min_score: float = 0.7, include_low_reach: bool = False) -> list[Action]:
    """Low reaches remain exploratory: floor pickups share the same pose pattern."""
    return [a for a in actions if a.pose_score >= min_score
            and (include_low_reach or a.pattern != "low_reach_recovery")]


def select_actions(actions: list[Action], max_highlights: int = 5, budget_s: float = 40) -> list[Action]:
    """Prefer strong sequences while avoiding overlapping clips and repetitive picks."""
    if max_highlights < 0 or not math.isfinite(budget_s) or budget_s < 0:
        raise ValueError("Highlight count and duration budget must be nonnegative")
    remaining = list(actions)
    selected = []
    used_s = 0.0
    pattern_counts, time_counts = Counter(), Counter()
    while remaining and len(selected) < max_highlights:
        action = max(remaining, key=lambda a: a.ranking_score / (1 + 0.2 * pattern_counts[a.pattern])
                     / (1 + 0.5 * time_counts[int(a.time_s / 30)]))
        remaining.remove(action)
        length = action.end_s - action.start_s
        if length <= 0 or used_s + length > budget_s + 1e-8:
            continue
        if any(action.start_s < other.end_s and action.end_s > other.start_s for other in selected):
            continue
        selected.append(action)
        used_s += length
        pattern_counts[action.pattern] += 1
        time_counts[int(action.time_s / 30)] += 1
    return sorted(selected, key=lambda action: action.time_s)
