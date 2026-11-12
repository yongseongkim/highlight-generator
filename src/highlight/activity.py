"""Court-only pose activity proposals, never inferred serves or ground contacts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from highlight.badminton_events import player_foot
from highlight.court import Court
from highlight.game_units import SportEvent
from highlight.pattern_signals import _median3


def court_activity(poses: dict, court: Court) -> list[dict]:
    """Aggregate motion over ~250 ms, with explicit visibility/unknown samples."""
    if court.polygon is None:
        raise ValueError("Court activity requires a target-court polygon")
    k = poses["keypoints"]
    boxes = poses["box"]
    scale = np.linalg.norm(k[:, [5, 6], :2].mean(axis=1) - k[:, [11, 12], :2].mean(axis=1), axis=1)
    valid = ((poses["track_id"] >= 0) & (poses["score"] >= 0.35) & (scale >= 20)
             & np.isfinite(k).all(axis=(1, 2)) & np.isfinite(boxes).all(axis=1)
             & (k[:, [5, 6, 11, 12], 2] >= 0.5).all(axis=1))
    groups, visible = defaultdict(list), defaultdict(int)
    for row in np.flatnonzero(valid):
        if court.contains(player_foot(k[row], boxes[row]), margin=0):
            groups[int(poses["track_id"][row])].append(int(row))
            visible[int(poses["frame_index"][row])] += 1
    velocities = defaultdict(list)
    for rows in groups.values():
        rows = np.asarray(sorted(rows, key=lambda row: poses["time_s"][row]))
        if len(rows) < 4:
            continue
        times = poses["time_s"][rows]
        xy = _median3(k[rows, :, :2].astype(float))
        sizes = np.maximum(_median3(scale[rows]), 20)
        hips = xy[:, [11, 12]].mean(axis=1)
        old = np.maximum(np.searchsorted(times, times - 0.25, side="right") - 1, 0)
        dt = times - times[old]
        known = (dt >= 0.15) & (dt <= 0.45) & (sizes / sizes[old] >= 0.6) & (sizes / sizes[old] <= 1 / 0.6)
        safe_dt = np.maximum(dt, 1e-6)
        body = np.linalg.norm(hips - hips[old], axis=1) / sizes / safe_dt
        wrists = (xy[:, [9, 10]] - xy[:, [5, 6]]) / sizes[:, None, None]
        hands = np.linalg.norm(wrists - wrists[old], axis=2) / safe_dt[:, None]
        wrist_known = (k[rows][:, [9, 10], 2] >= 0.5) & (k[rows[old]][:, [9, 10], 2] >= 0.5)
        hands[~wrist_known] = 0
        hand = hands.max(axis=1)
        known &= (body <= 6) & (hand <= 20) & wrist_known.any(axis=1)
        # Long invisible gaps must not become quiet observations after a track ID
        # reappears. IDs are evidence within their continuous sampled run only.
        gaps = np.r_[0.0, np.diff(times)]
        cumulative = np.cumsum(gaps > 0.2)
        known &= cumulative == cumulative[old]
        body, hand = _median3(body), _median3(hand)
        for i in np.flatnonzero(known):
            velocities[int(poses["frame_index"][rows[i]])].append((float(body[i]), float(hand[i])))
    result = []
    for frame, time_s in zip(poses["sampled_frame_index"], poses["sampled_time_s"]):
        values = velocities[int(frame)]
        result.append({"time_s": float(time_s), "visible_players": visible[int(frame)], "known_players": len(values),
                       "active_players": sum(body >= 0.6 or hand >= 1.6 for body, hand in values),
                       "quiet_players": sum(body < 0.25 and hand < 0.7 for body, hand in values),
                       "max_body_speed": max((body for body, _ in values), default=None),
                       "max_hand_speed": max((hand for _, hand in values), default=None)})
    return result


def activity_events(samples: list[dict], court_id: str = "main", idle_s: float = 2.5,
                    onset_s: float = 0.25) -> list[SportEvent]:
    """Propose active spans. Unknown/occluded samples never count as quiet time."""
    if idle_s <= 0 or onset_s < 0:
        raise ValueError("Idle threshold must be positive and onset nonnegative")
    events = []
    active = False
    onset = quiet = None
    for sample in samples:
        t = sample["time_s"]
        known = sample["known_players"] >= 2
        moving = known and sample["active_players"] >= 1
        still = known and sample["quiet_players"] == sample["known_players"]
        if not known:
            onset = quiet = None
            continue
        if not active:
            if moving:
                onset = t if onset is None else onset
                if t - onset >= onset_s:
                    events.append(SportEvent(max(0.0, onset - 0.5), "activity_start", court_id,
                                             source="court_pose_activity", data={"boundary_estimate": True}))
                    active = True
                    onset = None
            else:
                onset = None
        elif still:
            quiet = t if quiet is None else quiet
            if t - quiet >= idle_s:
                events.append(SportEvent(quiet, "activity_pause", court_id, source="court_pose_inactivity",
                                         data={"observed_until_s": t, "boundary_estimate": True,
                                               "not_a_ground_contact_observation": True}))
                active = False
                onset = quiet = None
        else:
            quiet = None
    return events
