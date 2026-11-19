"""Source metadata and dense observations without requiring pose detections."""
from __future__ import annotations

import math
from pathlib import Path
import time

import cv2
import numpy as np

from highlight.fusion_signals import DenseCv, PROFILES, PoseIndex
from highlight.pose import file_identity, load_poses
from highlight.motion import motion_samples, find_seeds


def video_metadata(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    try:
        meta = {"video": str(video.resolve()), "source_identity": file_identity(video),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        if not cap.isOpened() or min(meta["width"], meta["height"], meta["frames"]) <= 0 or not math.isfinite(meta["fps"]) or meta["fps"] <= 0:
            raise ValueError(f"Invalid source video: {video}")
        return meta
    finally:
        cap.release()


def empty_poses() -> dict:
    return {"sampled_frame_index": np.array([], dtype=int), "sampled_time_s": np.array([], dtype=float),
            "frame_index": np.array([], dtype=int), "time_s": np.array([], dtype=float),
            "track_id": np.array([], dtype=int), "box": np.empty((0, 4)),
            "score": np.array([], dtype=float), "keypoints": np.empty((0, 17, 3))}


def source_poses(path: Path | None, meta: dict) -> dict:
    if path is None:
        return empty_poses()
    poses, saved = load_poses(path)
    if Path(saved["video"]).resolve() != Path(meta["video"]).resolve():
        raise ValueError("Pose archive belongs to another video")
    if saved.get("source_identity") is not None and saved["source_identity"] != meta["source_identity"]:
        raise ValueError("Pose archive source changed")
    if any(saved[key] != meta[key] for key in ("width", "height", "frames")) or abs(saved["fps"] - meta["fps"]) > .001:
        raise ValueError("Pose archive image geometry/frame rate differs")
    return poses


def pose_discovery_seeds(poses: dict, court, court_id: str) -> list[dict]:
    if not len(poses["frame_index"]):
        return []
    samples, _ = motion_samples(poses, court)
    events, _ = find_seeds(samples, court_id)
    return [{"frame_index": e.data["frame_index"], "time_s": e.time_s, "source": "pose",
             "score": e.data["energy"]/(1+e.data["energy"]), "reason": "court_pose_motion_peak",
             "evidence_refs": [f"pose:{e.data['frame_index']}"], "court_association": "in_court_player"} for e in events]


def scan_video(video: Path, index: PoseIndex, sport: str, meta: dict, width: int = 640,
               observer=None) -> tuple[list[dict], float]:
    analyzer = DenseCv(index.court, PROFILES[sport], width)
    cap = cv2.VideoCapture(str(video))
    rows = []
    started, progress = time.perf_counter(), time.perf_counter()
    unstable = False
    background_reference = None
    # Leave a buffer for bodies/arms extending above their ground-level ROI.
    background_height = max(0, int(.5*min(p[1] for p in index.court.polygon)*analyzer.scale))
    camera_shift = None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame = len(rows)
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if rows and t <= rows[-1]["time_s"]:
                t = frame / meta["fps"]
            row, _, refreshed = analyzer.observe(bgr, frame, t)
            if refreshed:
                # A moving player can dominate full-image phase correlation. Use
                # textured background above the ROI; insufficient texture is unknown.
                background = analyzer.reference[:background_height].astype(np.float32)
                camera_shift = None
                if (background_height >= 16 and background.std() > 5 and background_reference is not None
                        and background_reference.std() > 5):
                    shift, response = cv2.phaseCorrelate(background_reference, background)
                    camera_shift = {"dx": shift[0]/analyzer.scale, "dy": shift[1]/analyzer.scale, "response": response}
                background_reference = background.copy()
                unstable = bool(camera_shift and camera_shift["response"] > .3 and
                                math.hypot(camera_shift["dx"], camera_shift["dy"]) > .02 * meta["width"])
            _, players = index.at(frame)
            usable = any(np.any(index.poses["keypoints"][p, [9, 10], 2] >= .5) for p in players)
            row.update(camera_unstable=unstable, camera_shift=camera_shift, pose_players=len(players), usable_pose=usable)
            rows.append(row)
            if observer is not None:
                observer(bgr, frame, t)
            if time.perf_counter() - progress >= 30:
                print(f"discovery CV: {t:.1f}s / frame {frame}", flush=True)
                progress = time.perf_counter()
    finally:
        cap.release()
    if len(rows) != meta["frames"]:
        raise RuntimeError(f"Decoded {len(rows)} frames; expected {meta['frames']}")
    return rows, time.perf_counter() - started
