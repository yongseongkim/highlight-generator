"""Draw YOLO-pose skeletons and SAM 3 shot tracking on frames with OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from highlight.candidates import Window, window_at
from highlight.pose import SKELETON, FramePoses
from highlight.shots import BallSample, Hoop, ShotResult
from highlight.video import NvdecReader

# BGR colors, one per track ID (cycled).
PALETTE = (
    (255, 56, 56), (56, 56, 255), (56, 200, 56), (255, 157, 0), (200, 56, 255),
    (0, 200, 255), (255, 56, 200), (0, 128, 255), (128, 255, 0), (255, 255, 0),
)  # fmt: skip
UNTRACKED = (160, 160, 160)
HOOP_COLOR = (0, 255, 0)
BALL_COLOR = (0, 0, 255)
TRAIL_COLOR = (0, 215, 255)
TRAIL_S = 1.0  # seconds of ball path to draw
VERDICTS = {"made": "MADE", "miss": "MISSED", "noshot": "NO SHOT"}


def draw_poses(bgr: np.ndarray, poses: FramePoses, kpt_conf: float = 0.5) -> np.ndarray:
    """Draw each person's box, track ID and skeleton onto `bgr` in place and return it."""
    for track_id, box, kpts in zip(poses.track_ids, poses.boxes, poses.keypoints):
        color = PALETTE[track_id % len(PALETTE)] if track_id >= 0 else UNTRACKED
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"#{track_id}" if track_id >= 0 else "?"
        cv2.putText(bgr, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        visible = kpts[:, 2] >= kpt_conf
        points = kpts[:, :2].astype(int)
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                cv2.line(bgr, tuple(points[a]), tuple(points[b]), color, 3, cv2.LINE_AA)
        for (x, y), vis in zip(points, visible):
            if vis:
                cv2.circle(bgr, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return bgr


def draw_ball(bgr: np.ndarray, hoop: Hoop, track: list[BallSample], time_s: float) -> np.ndarray:
    """Draw the hoop, the ball (if found in this frame) and its path over the last TRAIL_S seconds."""
    x1, y1, x2, y2 = hoop.box.astype(int)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), HOOP_COLOR, 2)
    recent = [s for s in track if time_s - TRAIL_S <= s.time_s <= time_s]
    points = np.array([hoop.to_pixels(s.x, s.y) for s in recent], dtype=np.int32)
    if len(points) > 1:
        cv2.polylines(bgr, [points], False, TRAIL_COLOR, 2, cv2.LINE_AA)
    if recent and recent[-1].time_s == time_s:
        cv2.circle(bgr, tuple(points[-1]), round(0.3 * hoop.width), BALL_COLOR, 2, cv2.LINE_AA)
    return bgr


def add_rim_zoom(bgr: np.ndarray, hoop: Hoop, size: int = 540, margin: int = 16) -> np.ndarray:
    """Copy an enlarged view of the area around the rim into the top corner away from the hoop."""
    height, width = bgr.shape[:2]
    rim_x, rim_y = hoop.rim
    side = int(6 * hoop.width)
    x0 = int(np.clip(rim_x - side / 2, 0, width - side))
    y0 = int(np.clip(rim_y - 0.6 * side, 0, height - side))  # more room above the rim for the arc
    zoom = cv2.resize(bgr[y0 : y0 + side, x0 : x0 + side], (size, size), interpolation=cv2.INTER_CUBIC)
    x = margin if rim_x > width / 2 else width - size - margin
    bgr[margin : margin + size, x : x + size] = zoom
    cv2.rectangle(bgr, (x, margin), (x + size, margin + size), (255, 255, 255), 2)
    return bgr


def draw_banner(bgr: np.ndarray, text: str, scale: float = 1.2) -> np.ndarray:
    """Text on a dark strip at the bottom-left."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = bgr.shape[0] - 24
    cv2.rectangle(bgr, (16, y - h - 14), (16 + w + 20, y + base + 6), (0, 0, 0), -1)
    cv2.putText(bgr, text, (26, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def render_shot_videos(
    video: str,
    windows: list[Window],
    shots: list[ShotResult],
    hoop: Hoop | None,
    poses: dict[int, FramePoses],
    stride: int,
    out_paths: list[Path],
) -> None:
    """One video per window: YOLO-pose skeletons plus SAM 3's hoop, ball path and verdict.

    Rendered on the frames SAM 3 analyzed (every `stride`-th). Pose ran on a different
    stride, so each frame shows the skeletons from the nearest pose frame, at most one
    frame away.
    """
    reader = NvdecReader(video)
    pose_frames = np.array(sorted(poses))
    writers: list[cv2.VideoWriter | None] = [None] * len(windows)
    for frame in reader.frames(stride):
        if frame.time_s > windows[-1].end_s:
            break
        i = window_at(windows, frame.time_s)
        if i is None:
            continue
        bgr = np.ascontiguousarray(frame.rgb.cpu().numpy()[..., ::-1])
        nearest = int(pose_frames[np.abs(pose_frames - frame.index).argmin()])
        if abs(nearest - frame.index) <= 1:
            draw_poses(bgr, poses[nearest])
        shot = shots[i]
        if hoop is not None:
            draw_ball(bgr, hoop, shot.track, frame.time_s)
            add_rim_zoom(bgr, hoop)
        verdict = VERDICTS[shot.label]
        if shot.made and frame.time_s >= shot.made_at_s:
            verdict += f" at {shot.made_at_s:.1f} s"
        draw_banner(bgr, f"{frame.time_s:5.2f} s   SAM 3: {verdict}")
        if writers[i] is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writers[i] = cv2.VideoWriter(str(out_paths[i]), fourcc, reader.info.fps / stride, (bgr.shape[1], bgr.shape[0]))
        writers[i].write(bgr)
    for writer in writers:
        if writer is not None:
            writer.release()
