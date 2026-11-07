"""Expand strong court-scoped motion peaks into reviewable context intervals.

Image-plane kinematics propose interesting moments, not shot types or outcomes.
Local lulls, visibility gaps, and search limits are editorial cut suggestions;
none verifies service, ground contact, a complete rally, or a goal.
"""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from highlight.court import Court
from highlight.game_units import GameUnit, SportEvent
from highlight.pattern_signals import _median3, _segments


def motion_samples(poses: dict, court: Court) -> tuple[list[dict], dict]:
    """Measure body/relative-arm movement only within continuous in-court tracks."""
    if court.polygon is None:
        raise ValueError("Motion seeds require a target-court polygon")
    segments, diagnostics = _segments(poses, court)
    observations = defaultdict(list)
    for rows in segments:
        times = poses["time_s"][rows].astype(float)
        raw = poses["keypoints"][rows].astype(float)
        xy = _median3(raw[..., :2])
        hips = xy[:, [11, 12]].mean(axis=1)
        shoulders = xy[:, [5, 6]].mean(axis=1)
        scale = np.maximum(_median3(np.linalg.norm(shoulders - hips, axis=1)), 20)
        # A longer difference interval limits amplification of single-frame jitter.
        left = np.maximum(np.searchsorted(times, times - 0.1, side="right") - 1, 0)
        right = np.minimum(np.searchsorted(times, times + 0.1), len(times) - 1)
        dt = np.maximum(times[right] - times[left], 1e-6)
        velocity = (hips[right] - hips[left]) / scale[:, None] / dt[:, None]
        body = np.linalg.norm(velocity, axis=1)
        arms = (xy[:, [9, 10]] - xy[:, [5, 6]]) / scale[:, None, None]
        hand_speeds = np.linalg.norm(arms[right] - arms[left], axis=2) / dt[:, None]
        visible = raw[:, [9, 10], 2] >= 0.6
        visible &= visible[left] & visible[right]
        hand_speeds[~visible] = 0
        hand = hand_speeds.max(axis=1)
        change = np.linalg.norm(velocity[right] - velocity[left], axis=1) / dt
        quality = raw[:, [5, 6, 11, 12], 2].mean(axis=1) * poses["score"][rows]
        known = ((times - times[0] >= 0.2) & (times[-1] - times >= 0.2)
                 & (quality >= 0.5) & (body <= 6) & (hand <= 20) & (change <= 40))
        energy = _median3(np.maximum.reduce([body / 1.2, hand / 3.5, change / 8]))
        for i in np.flatnonzero(known):
            frame = int(poses["frame_index"][rows[i]])
            observations[frame].append({"track_id": int(poses["track_id"][rows[i]]),
                "energy": float(energy[i]), "body_speed": float(body[i]), "arm_speed": float(hand[i]),
                "body_velocity_change": float(change[i]), "vertical_body_speed": float(abs(velocity[i, 1]))})
    samples = []
    for frame, t in zip(poses["sampled_frame_index"], poses["sampled_time_s"]):
        players = observations[int(frame)]
        anchor = max(players, key=lambda p: p["energy"]) if players else None
        samples.append({"time_s": float(t), "frame_index": int(frame), "known_players": len(players),
                        "energy": anchor["energy"] if anchor else None, "anchor": anchor})
    diagnostics["known_motion_samples"] = sum(sample["energy"] is not None for sample in samples)
    return samples, diagnostics


def find_seeds(samples: list[dict], court_id: str = "main", percentile: float = 95,
               min_energy: float = 1.4, gap_s: float = 2.0) -> tuple[list[SportEvent], float]:
    """Keep sustained local peaks above both an absolute and video-relative gate."""
    if not all(math.isfinite(v) for v in (percentile, min_energy, gap_s)) or not 50 <= percentile < 100 or min_energy <= 0 or gap_s < 0:
        raise ValueError("Seed percentile must be in [50, 100); energy positive and gap nonnegative")
    known = [sample["energy"] for sample in samples if sample["energy"] is not None]
    threshold = max(min_energy, float(np.percentile(known, percentile))) if known else min_energy
    times = np.asarray([s["time_s"] for s in samples])
    candidates = []
    for i in range(1, len(samples) - 1):
        sample = samples[i]
        score = sample["energy"]
        if score is None or score < threshold:
            continue
        neighbors = [samples[j]["energy"] for j in (i - 1, i + 1)]
        if any(value is None or value > score for value in neighbors):
            continue
        nearby = [samples[j] for j in range(max(0, i - 4), min(len(samples), i + 5))
                  if abs(times[j] - times[i]) <= 0.22 and samples[j]["energy"] is not None
                  and samples[j]["energy"] >= 0.6 * score]
        if len(nearby) < 2 or nearby[-1]["time_s"] - nearby[0]["time_s"] < 0.08:
            continue
        local = [s["energy"] for s in samples[max(0, np.searchsorted(times, times[i] - 3)):
                                                      np.searchsorted(times, times[i] + 3)] if s["energy"] is not None]
        if score < 1.3 * float(np.median(local)):
            continue  # sustained ordinary motion alone is not a dramatic peak
        candidates.append(sample)
    kept = []
    for sample in sorted(candidates, key=lambda s: (-s["energy"], s["time_s"])):
        if all(abs(sample["time_s"] - old["time_s"]) >= gap_s for old in kept):
            kept.append(sample)
    events = [SportEvent(sample["time_s"], "motion_peak", court_id, source="court_pose_kinematics",
                         data={**(sample.get("anchor") or {}), "energy": sample["energy"],
                               "seed_threshold": threshold, "frame_index": sample["frame_index"]})
              for sample in sorted(kept, key=lambda s: s["time_s"])]
    return events, threshold


def expand_seeds(samples: list[dict], seeds: list[SportEvent], duration_s: float, sport: str,
                 court_id: str = "main", before_s: float = 12, after_s: float = 8,
                 lull_s: float = 0.6, padding_s: float = 0.75) -> list[GameUnit]:
    """Search away from peaks for local lulls; cap and merge candidate contexts."""
    if not math.isfinite(duration_s) or duration_s <= 0 or not all(math.isfinite(v) and v > 0 for v in (before_s, after_s, lull_s)):
        raise ValueError("Duration, search limits, and lull duration must be finite and positive")
    if not math.isfinite(padding_s) or padding_s < 0:
        raise ValueError("Context padding must be finite and nonnegative")
    if not samples:
        return []
    times = np.asarray([sample["time_s"] for sample in samples])
    kept = []
    span_limit = before_s + after_s
    for seed in sorted((s for s in seeds if s.court_id == court_id and 0 <= s.time_s <= duration_s),
                       key=lambda e: (-e.data["energy"], e.time_s)):
        lower, upper = max(0, seed.time_s - before_s), min(duration_s, seed.time_s + after_s)
        local = [s["energy"] for s in samples[np.searchsorted(times, lower):np.searchsorted(times, upper, side="right")]
                 if s["known_players"] >= 2 and s["energy"] is not None]
        low = max(0.25, min(0.65 * seed.data["energy"], float(np.percentile(local, 35)))) if local else 0.25

        def boundary(direction: int):
            limit = lower if direction < 0 else upper
            quiet_start = unknown_start = None
            indices = np.flatnonzero((times >= lower) & (times <= upper)
                                     & ((times < seed.time_s) if direction < 0 else (times > seed.time_s)))
            if direction < 0:
                indices = indices[::-1]
            for index in indices:
                sample = samples[index]
                t = sample["time_s"]
                if abs(t - seed.time_s) < min(1.0, before_s if direction < 0 else after_s):
                    continue
                if sample["known_players"] == 0 or sample["energy"] is None:
                    quiet_start = None
                    unknown_start = t if unknown_start is None else unknown_start
                    if abs(t - unknown_start) >= max(0.8, lull_s):
                        return unknown_start, "visibility_gap"
                    continue
                unknown_start = None
                # A single visible player cannot establish a lull for the court.
                if sample["known_players"] >= 2 and sample["energy"] <= low:
                    quiet_start = t if quiet_start is None else quiet_start
                    if abs(t - quiet_start) >= lull_s:
                        cut = quiet_start + direction * padding_s
                        return float(np.clip(cut, lower, upper)), "local_motion_lull"
                else:
                    quiet_start = None
            return limit, ("video_boundary" if limit in (0, duration_s) else "context_limit")

        start, start_reason = boundary(-1)
        end, end_reason = boundary(1)
        unit = GameUnit(sport, "motion_candidate", start, end, start, end, start_reason, end_reason,
                        False, False, court_id, [seed], features={"motion_seed_score": -math.expm1(-seed.data["energy"] / seed.data["seed_threshold"]),
                        "seed_count": 1, "local_lull_energy": low,
                        "interpretation": "Motion context proposal; service, landing, rally completeness, and outcome are unknown."})
        # Strongest peaks own their context. Merge intersecting proposals only
        # when the union stays bounded; otherwise suppress the weaker overlap.
        overlapping = []
        while True:
            additional = [old for old in kept if old not in overlapping and start < old.end_s and end > old.start_s]
            if not additional:
                break
            overlapping.extend(additional)
            start = min(start, *(old.start_s for old in additional))
            end = max(end, *(old.end_s for old in additional))
        if end - start > span_limit + 1e-8:
            continue
        if overlapping:
            first = min([unit, *overlapping], key=lambda u: u.start_s)
            last = max([unit, *overlapping], key=lambda u: u.end_s)
            unit.start_s = unit.play_start_s = start
            unit.end_s = unit.play_end_s = end
            unit.start_reason, unit.end_reason = first.start_reason, last.end_reason
            unit.events += [event for old in overlapping for event in old.events]
            unit.features["motion_seed_score"] = max(unit.features["motion_seed_score"], *(old.features["motion_seed_score"] for old in overlapping))
            kept = [old for old in kept if old not in overlapping]
        unit.events.sort(key=lambda e: e.time_s)
        unit.features["seed_count"] = len(unit.events)
        kept.append(unit)
    return sorted(kept, key=lambda unit: unit.start_s)
