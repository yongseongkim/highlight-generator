"""Timestamped multi-rate observations; no outcome or highlight certainty claims."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import cv2
import numpy as np

from highlight.badminton_events import player_foot
from highlight.court import Court


@dataclass(frozen=True)
class SportProfile:
    prompts: tuple[str, ...]
    native_labels: tuple[str, ...]
    expected_objects: tuple[str, ...]
    surface: str


PROFILES = {
    "badminton": SportProfile(("badminton racket", "shuttlecock"),
        ("badminton racket", "racket", "tennis racket", "shuttlecock", "shuttle"),
        ("shuttlecock", "badminton racket"), "indoor_lines"),
    "basketball": SportProfile(("basketball", "basketball hoop"),
        ("sports ball", "ball", "basketball", "basketball hoop", "hoop"),
        ("basketball", "basketball hoop"), "indoor_lines"),
    "football": SportProfile(("soccer ball", "soccer goal"),
        ("sports ball", "ball", "football", "soccer ball", "soccer goal", "goal"),
        ("soccer ball", "soccer goal"), "grass_and_lines"),
}


class PoseIndex:
    """Zero-margin foot filtering plus past-only bounded-age lookups."""
    def __init__(self, poses: dict, court: Court):
        self.poses, self.court = poses, court
        self.frames = np.asarray(poses["sampled_frame_index"])
        self.times = np.asarray(poses["sampled_time_s"])
        self.rows = defaultdict(list)
        for row, frame in enumerate(poses["frame_index"]):
            k, box = poses["keypoints"][row], poses["box"][row]
            if (poses["score"][row] >= 0.35 and np.isfinite(k).all() and np.isfinite(box).all()
                    and court.contains(player_foot(k, box), margin=0)):
                self.rows[int(frame)].append(row)

    def at(self, frame: int, max_age_s: float = 0.2) -> tuple[int | None, list[int]]:
        index = int(np.searchsorted(self.frames, frame, side="right")) - 1
        if index < 0 or (frame - self.frames[index]) / self.court.fps > max_age_s:
            return None, []
        source = int(self.frames[index])
        return source, self.rows[source]


def object_observations(raw: list[dict], index: PoseIndex, frame: int, time_s: float,
                        source: str) -> list[dict]:
    """Preserve native labels and raw boxes; gate display by plausible association.

    Flying objects are not excluded merely for appearing above a ground polygon.
    Ground-ROI membership is a weak association, never physical court identity.
    """
    _, rows = index.at(frame)
    p, court = index.poses, index.court
    output = []
    for n, item in enumerate(raw):
        box = np.asarray(item["box"], dtype=float)
        confidence = float(item["confidence"])
        if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(confidence) or not 0 <= confidence <= 1 or (box[2:] <= box[:2]).any():
            continue
        label = item["label"]
        center = (box[:2] + box[2:]) / 2
        hand_distance, nearest_track, torso = math.inf, None, None
        for row in rows:
            k = p["keypoints"][row]
            scale = max(20.0, float(np.linalg.norm(k[[5, 6], :2].mean(axis=0) - k[[11, 12], :2].mean(axis=0))))
            wrists = k[[9, 10]]
            for wrist in wrists[wrists[:, 2] >= 0.5, :2]:
                distance = np.linalg.norm(np.maximum(np.maximum(box[:2] - wrist, wrist - box[2:]), 0)) / scale
                if distance < hand_distance:
                    hand_distance, nearest_track, torso = float(distance), int(p["track_id"][row]), scale
        racket = "racket" in label.lower()
        shuttle = "shuttle" in label.lower()
        near_hand = hand_distance <= (0.6 if racket else 3)
        inside_ground = court.contains(center, margin=0)
        plausible = (not racket or (torso is not None and max(box[2:] - box[:2]) <= 2.5 * torso))
        plausible &= not shuttle or max(box[2:] - box[:2]) <= 0.035 * court.width
        accepted = confidence >= 0.4 and plausible and (near_hand if racket else near_hand or inside_ground)
        output.append({"observation_id": f"{source}:{frame}:{n}", "source": source,
            "frame_index": frame, "time_s": round(time_s, 6), "native_label": label,
            "confidence": confidence, "box": box.tolist(),
            "box_normalized": (box / [court.width, court.height, court.width, court.height]).round(5).tolist(),
            "center_normalized": (center / [court.width, court.height]).round(5).tolist(),
            "accepted_for_display": bool(accepted), "nearest_player_track_id": nearest_track,
            "hand_distance_torso": hand_distance if math.isfinite(hand_distance) else None,
            "association": "near_in_court_player" if near_hand else "inside_ground_roi" if inside_ground else "unassociated",
            "tracklet_id": None})
    return output


class ObjectTracklets:
    """Conservative same-label nearest-neighbor links; no gap interpolation."""
    def __init__(self, width: int, height: int, max_gap_s: float = 0.65):
        self.diagonal = math.hypot(width, height)
        self.max_gap_s = max_gap_s
        self.active = {}
        self.count = 0

    def update(self, objects: list[dict], time_s: float) -> None:
        self.active = {key: value for key, value in self.active.items() if 0 <= time_s - value["time_s"] <= self.max_gap_s}
        used = set()
        for obj in sorted(objects, key=lambda o: -o["confidence"]):
            if not obj["accepted_for_display"]:
                continue
            center = np.asarray(obj["center_normalized"])
            candidates = [(float(np.linalg.norm(center - old["center_normalized"])), key)
                          for key, old in self.active.items() if key not in used and old["native_label"] == obj["native_label"]]
            distance, match = min(candidates, default=(math.inf, None))
            if distance > 0.15:
                self.count += 1
                match = f"yolo-track-{self.count}"
            obj["tracklet_id"] = match
            self.active[match] = obj
            used.add(match)


def mark_stationary_sam(rows: list[dict], width: int) -> int:
    """Repeated fixed shuttle-like boxes are counterevidence, not a flight path.

    This is an uncertain stationary-object flag, not proof of a spare shuttle or
    of no play. Raw boxes remain available for inspection.
    """
    objects = [obj for row in rows for obj in row["objects"] if "shuttle" in obj["native_label"].lower()]
    rejected = 0
    for obj in objects:
        center = (np.array(obj["box"][:2]) + obj["box"][2:]) / 2
        repeated = any(other["frame_index"] != obj["frame_index"] and 0.1 <= abs(other["time_s"]-obj["time_s"]) <= 0.8
                       and np.linalg.norm(center - (np.array(other["box"][:2]) + other["box"][2:]) / 2) <= max(5, width*0.005)
                       for other in objects)
        if repeated:
            obj["stationary_candidate"] = True
            obj["display_rejection_reason"] = "repeated_stationary_small_object"
            if obj["accepted_for_display"]:
                rejected += 1
            obj["accepted_for_display"] = False
    return rejected


class RateSampler:
    """Low baseline cadence; denser candidate windows; hard image-call budget."""
    def __init__(self, base_fps: float, peak_fps: float, windows: list[dict], budget: int):
        if not all(math.isfinite(v) and v > 0 for v in (base_fps, peak_fps)) or peak_fps < base_fps or budget < 0:
            raise ValueError("Sampling rates must be positive, peak >= base, budget nonnegative")
        self.base, self.peak, self.windows, self.budget = base_fps, peak_fps, windows, budget
        self.last, self.calls = -math.inf, 0

    def due(self, t: float) -> bool:
        fps = self.peak if any(w["start_s"] <= t <= w["end_s"] for w in self.windows) else self.base
        if self.calls >= self.budget or t - self.last + 1e-9 < 1 / fps:
            return False
        self.last, self.calls = t, self.calls + 1
        return True


class DenseCv:
    """Per-frame ROI differences; periodic line/grass/camera diagnostics."""
    def __init__(self, court: Court, profile: SportProfile, width: int = 640, line_interval_s: float = 2):
        self.court, self.profile = court, profile
        self.scale = min(1.0, width / court.width)
        self.size = (round(court.width * self.scale), round(court.height * self.scale))
        self.mask = np.zeros(self.size[::-1], np.uint8)
        cv2.fillPoly(self.mask, [np.rint(np.array(court.polygon) * self.scale).astype(np.int32)], 255)
        self.inside = self.mask > 0
        self.previous = self.reference = None
        self.last_lines = -math.inf
        self.line_interval = line_interval_s
        self.lines, self.camera_shift = [], None

    def observe(self, bgr: np.ndarray, frame: int, t: float) -> tuple[dict, np.ndarray, bool]:
        small = cv2.resize(bgr, self.size)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        diff = cv2.absdiff(gray, self.previous) if self.previous is not None else np.zeros_like(gray)
        changed = ((diff > 14) & self.inside).astype(np.uint8) * 255
        ratio = float(np.mean(changed[self.inside] > 0))
        updated = t - self.last_lines >= self.line_interval
        if updated:
            edges = cv2.bitwise_and(cv2.Canny(gray, 50, 130), self.mask)
            segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=45, maxLineGap=12)
            self.lines = [] if segments is None else (segments.reshape(-1, 4)[:16] / self.scale).round(1).tolist()
            self.camera_shift = None
            if self.reference is not None:
                shift, response = cv2.phaseCorrelate(self.reference.astype(np.float32), gray.astype(np.float32))
                self.camera_shift = {"dx": round(shift[0] / self.scale, 2), "dy": round(shift[1] / self.scale, 2), "response": round(response, 3)}
            self.reference = gray.copy()
            self.last_lines = t
        grass = None
        if self.profile.surface == "grass_and_lines":
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            grass = float(np.mean(cv2.inRange(hsv, (35, 35, 35), (95, 255, 255))[self.inside] > 0))
        observation = {"frame_index": frame, "time_s": round(t, 6), "roi_motion_fraction": round(ratio, 5),
            "roi_brightness": round(float(gray[self.inside].mean()), 2),
            "scene_change_candidate": bool(diff.mean() > 25 and ratio > 0.5),
            "grass_fraction": round(grass, 4) if grass is not None else None}
        self.previous = gray
        return observation, changed, updated


def bounded_sample(records: list, limit: int) -> list:
    if len(records) <= limit:
        return records
    return [records[i] for i in np.linspace(0, len(records) - 1, limit).astype(int)]


def evidence_packet(unit: dict, cv_rows: list[dict], pose_index: PoseIndex, object_rows: list[dict],
                    sam_rows: list[dict], images: list[dict], coverage: dict) -> dict:
    start, end = unit["start_s"], unit["end_s"]
    cv = [r for r in cv_rows if start <= r["time_s"] <= end]
    object_samples = [r for r in object_rows if start <= r["time_s"] <= end]
    sam_samples = [r for r in sam_rows if start <= r["time_s"] <= end]
    sampled = np.flatnonzero((pose_index.times >= start) & (pose_index.times <= end))
    tracks = defaultdict(list)
    rejected = []
    for sample in object_samples + sam_samples:
        for obj in sample["objects"]:
            if obj["accepted_for_display"]:
                tracks[obj["tracklet_id"] or obj["observation_id"]].append(obj)
            else:
                rejected.append({key: obj.get(key) for key in ("observation_id", "native_label", "time_s", "confidence", "association", "display_rejection_reason", "stationary_candidate")})
    summaries = []
    for track, objects in tracks.items():
        objects.sort(key=lambda o: o["time_s"])
        points = np.asarray([o["center_normalized"] for o in objects])
        duration = objects[-1]["time_s"] - objects[0]["time_s"]
        displacement = float(np.linalg.norm(points[-1] - points[0]))
        summaries.append({"tracklet_id": track, "native_label": objects[0]["native_label"], "source": objects[0]["source"],
            "observations": len(objects), "observed_duration_s": round(duration, 3),
            "displacement_normalized_xy": round(displacement, 5),
            "possibly_stationary": bool(len(objects) >= 3 and duration >= 0.5 and np.max(np.linalg.norm(points - points[0], axis=1)) < 0.008),
            "samples": [{key: o[key] for key in ("observation_id", "time_s", "center_normalized", "confidence", "hand_distance_torso")}
                        for o in bounded_sample(objects, 6)]})
    players = []
    context_samples = set(bounded_sample(sampled.tolist(), 4))
    seeds = [e for e in unit.get("events", []) if e["kind"] == "motion_peak"]
    if seeds:
        peak = max(seeds, key=lambda e: e["data"]["energy"])["time_s"]
        for delta in [-0.3, 0, 0.3]:
            if len(sampled):
                context_samples.add(int(sampled[np.argmin(abs(pose_index.times[sampled] - peak - delta))]))
    for sample in sorted(context_samples):
        frame = int(pose_index.frames[sample])
        visible = []
        for row in pose_index.rows[frame]:
            p = pose_index.poses
            k = p["keypoints"][row]
            torso = max(20, float(np.linalg.norm(k[[5, 6], :2].mean(axis=0) - k[[11, 12], :2].mean(axis=0))))
            foot = player_foot(k, p["box"][row])
            wrist = k[[9, 10]]
            known = wrist[wrist[:, 2] >= 0.5]
            visible.append({"track_id": int(p["track_id"][row]),
                "foot_normalized": (foot / [pose_index.court.width, pose_index.court.height]).round(4).tolist(),
                "hand_height_above_foot_torso": round(float(np.min(foot[1] - known[:, 1]) / torso), 3) if len(known) else None})
        players.append({"frame_index": frame, "time_s": round(float(pose_index.times[sample]), 4), "players": visible})
    summaries.sort(key=lambda s: (-s["observations"], -s["observed_duration_s"]))
    return {"candidate_id": unit["candidate_id"], "sport": coverage["sport"], "interval_s": [start, end],
        "boundary_status": unit.get("boundary_status", "estimated"),
        "cut_reasons": [unit.get("start_reason"), unit.get("end_reason")],
        "motion_seeds": [{"time_s": e["time_s"], "measurements": e.get("data", {})} for e in unit.get("events", []) if e["kind"] == "motion_peak"],
        "coverage": {"cv_frames": len(cv), "pose_samples": len(sampled), "object_samples": len(object_samples), "sam_images": len(sam_samples),
                     "object_model_status": coverage["object_model_status"],
                     "object_checkpoint_classes": coverage["object_checkpoint_classes"], "missing_expected_classes": coverage["missing_expected_classes"],
                     "unknown_expected_classes": coverage["unknown_expected_classes"]},
        "opencv": {"mean_motion_fraction": round(float(np.mean([r["roi_motion_fraction"] for r in cv])), 5) if cv else None,
                   "scene_change_candidates": sum(r["scene_change_candidate"] for r in cv)},
        "player_context": players, "object_tracklets": summaries[:16], "omitted_tracklets": max(0, len(summaries)-16),
        "rejected_object_count": len(rejected), "rejected_object_observations": bounded_sample(rejected, 8), "keyframes": images,
        "unknowns": ["A 2D ground ROI does not establish an airborne object's court or its 3D height.",
                     "Sampled or missing detections do not confirm contact, ground impact, rally boundaries, or goals.",
                     "Tracklets are conservative observed associations, not verified identity or interpolated flight paths."]}
