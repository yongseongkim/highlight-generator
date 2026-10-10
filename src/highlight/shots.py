"""Basketball shot outcomes from SAM 3 ball and hoop detections.

The hoop is found once (the camera is assumed fixed). The ball is then searched for in a
crop around the hoop, where it is large enough to see and where the outcome is decided.
Ball positions are measured from the rim center in rim widths, so the rules below don't
depend on how far the camera is from the hoop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from highlight.candidates import Window, window_at
from highlight.sam import Sam3
from highlight.video import NvdecReader

BALL = "basketball"
HOOP = "basketball hoop"
CROP = 644  # matches Sam3's default imgsz, so crops are not resized

# A detection that stays within STATIC_RADIUS of its position in at least
# max(STATIC_MIN_FRAMES, STATIC_FRACTION of the frames) is not the game ball:
# spare balls on the floor, or a part of the rim that SAM 3 mistakes for one.
STATIC_RADIUS = 0.1  # rim widths
STATIC_MIN_FRAMES = 10
STATIC_FRACTION = 0.3


@dataclass(frozen=True)
class Hoop:
    box: np.ndarray  # xyxy, full-frame pixels

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def rim(self) -> tuple[float, float]:
        """Rim center. The prompt returns either the rim alone or rim + net, so measure down from the top."""
        return float(self.box[0] + self.box[2]) / 2, float(self.box[1]) + 0.15 * self.width

    def to_pixels(self, x: float, y: float) -> tuple[int, int]:
        """Rim-relative position in rim widths -> full-frame pixel."""
        rim_x, rim_y = self.rim
        return round(rim_x + x * self.width), round(rim_y + y * self.width)

    def crop_window(self, frame_w: int, frame_h: int) -> tuple[int, int, int]:
        """(x0, y0, size) of a square crop around the rim, with room above it for the ball's arc."""
        size = min(CROP, frame_w, frame_h)
        rim_x, rim_y = self.rim
        x0 = int(np.clip(rim_x - size / 2, 0, frame_w - size))
        y0 = int(np.clip(rim_y - 0.55 * size, 0, frame_h - size))
        return x0, y0, size


@dataclass
class BallSample:
    time_s: float
    x: float  # ball center minus rim center, in rim widths; + is right
    y: float  # + is below the rim
    conf: float


@dataclass
class ShotResult:
    attempt: bool  # the ball went above the rim, near the hoop
    made: bool  # the ball dropped through the rim
    made_at_s: float | None
    track: list[BallSample]  # the game ball near the hoop, one sample per frame it was found

    @property
    def label(self) -> str:
        return "made" if self.made else "miss" if self.attempt else "noshot"


def find_hoop(sam: Sam3, bgr: np.ndarray, min_conf: float = 0.4) -> Hoop | None:
    max_width = 0.1 * bgr.shape[1]  # rejects backboard and whole-frame boxes
    hoops = [d for d in sam.detect(bgr, [HOOP]) if d.conf >= min_conf and d.box[2] - d.box[0] < max_width]
    return Hoop(max(hoops, key=lambda d: d.conf).box) if hoops else None


def ball_samples(sam: Sam3, crop_bgr: np.ndarray, x0: int, y0: int, hoop: Hoop, time_s: float) -> list[BallSample]:
    """Every ball-sized detection in a crop whose top-left is (x0, y0) in the full frame."""
    rim_x, rim_y = hoop.rim
    samples = []
    for d in sam.detect(crop_bgr, [BALL]):
        if not 0.2 * hoop.width < d.box[2] - d.box[0] < 1.5 * hoop.width:  # specks and whole-crop boxes
            continue
        x = float(d.box[0] + d.box[2]) / 2 + x0
        y = float(d.box[1] + d.box[3]) / 2 + y0
        samples.append(BallSample(time_s, (x - rim_x) / hoop.width, (y - rim_y) / hoop.width, d.conf))
    return samples


def ball_track(samples: list[BallSample]) -> list[BallSample]:
    """One game-ball position per frame: drop detections that never move, keep the most confident."""
    if not samples:
        return []
    xy = np.array([(s.x, s.y) for s in samples])
    n_frames = len({s.time_s for s in samples})
    neighbors = (np.linalg.norm(xy[:, None] - xy[None], axis=2) < STATIC_RADIUS).sum(axis=1)
    static = neighbors >= max(STATIC_MIN_FRAMES, STATIC_FRACTION * n_frames)
    best: dict[float, BallSample] = {}
    for sample, is_static in zip(samples, static):
        if not is_static and (sample.time_s not in best or sample.conf > best[sample.time_s].conf):
            best[sample.time_s] = sample
    return [best[t] for t in sorted(best)]


def classify_shot(track: list[BallSample]) -> ShotResult:
    """Made: the ball crosses rim height well inside the rim, then keeps falling through the net.

    A ball rolling off the rim crosses rim height at its edge (|x| ~ 0.5), which the
    0.35 limit rejects; a make crosses near the middle.
    """
    attempt = any(s.y < -0.3 and abs(s.x) < 2.0 for s in track)
    made_at = None
    for above, below in zip(track, track[1:]):
        crosses_inside = (
            above.y < 0 <= below.y
            and below.time_s - above.time_s <= 0.25
            and abs(above.x) < 0.35
            and abs(below.x) < 0.35
        )
        falls_through = any(
            0 <= s.time_s - below.time_s <= 0.5 and s.y > 0.5 and abs(s.x) < 0.4 for s in track
        )
        if crosses_inside and falls_through:
            made_at = below.time_s
            break
    return ShotResult(attempt or made_at is not None, made_at is not None, made_at, track)


def analyze_shots(video: str, windows: list[Window], sam: Sam3, stride: int = 2) -> tuple[list[ShotResult], Hoop | None]:
    """Classify the shot in each window, decoding the video once.

    `stride` 2 (15 fps for 30 fps video) still sees a ball dropping through the rim on
    2-3 frames. The hoop is detected on the first window frame where SAM 3 finds it.
    """
    if not windows:
        return [], None
    samples: list[list[BallSample]] = [[] for _ in windows]
    hoop = None
    for frame in NvdecReader(video).frames(stride):
        if frame.time_s > windows[-1].end_s:
            break
        i = window_at(windows, frame.time_s)
        if i is None:
            continue
        if hoop is None:
            hoop = find_hoop(sam, _to_bgr(frame.rgb))
            if hoop is None:
                continue
            x0, y0, size = hoop.crop_window(frame.rgb.shape[1], frame.rgb.shape[0])
        crop = _to_bgr(frame.rgb[y0 : y0 + size, x0 : x0 + size])
        samples[i].extend(ball_samples(sam, crop, x0, y0, hoop, frame.time_s))
    return [classify_shot(ball_track(s)) for s in samples], hoop


def _to_bgr(rgb: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(rgb.cpu().numpy()[..., ::-1])
