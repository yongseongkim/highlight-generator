"""Turn a pose cue series into candidate highlight windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highlight.signals import CueSeries


@dataclass(frozen=True)
class Sport:
    cue: str
    threshold: float  # cue score that counts as an active frame


# Thresholds tuned on the first sample videos (fixed camera, 1080p, stride 3).
SPORTS = {
    "basketball": Sport(cue="arms_up", threshold=0.3),  # shots, rebounds, blocks
    "football": Sport(cue="foot_up", threshold=0.6),  # kicks; running strides reach ~0.5
}


@dataclass
class Window:
    start_s: float
    end_s: float
    peak_s: float
    peak_score: float
    peak_box: list[float]  # xyxy of the top person at the peak, for the next stage to focus on


def window_at(windows: list[Window], time_s: float) -> int | None:
    """Index of the window containing `time_s`, if any."""
    return next((i for i, w in enumerate(windows) if w.start_s <= time_s <= w.end_s), None)


def find_windows(
    series: CueSeries,
    threshold: float,
    duration_s: float,
    max_gap_s: float = 0.5,
    min_frames: int = 2,
    pre_s: float = 1.5,
    post_s: float = 2.0,
) -> list[Window]:
    """Group active frames into windows, padded so a shot's ball flight is included.

    Active frames closer than `max_gap_s` form one event; events with fewer than
    `min_frames` active frames are dropped as keypoint noise. Overlapping padded
    windows are merged.
    """
    active = np.flatnonzero(np.nan_to_num(series.score, nan=-np.inf) >= threshold)
    events: list[list[int]] = []
    for i in active:
        if events and series.time_s[i] - series.time_s[events[-1][-1]] <= max_gap_s:
            events[-1].append(i)
        else:
            events.append([i])

    windows: list[Window] = []
    for event in events:
        if len(event) < min_frames:
            continue
        peak = event[int(np.argmax(series.score[event]))]
        window = Window(
            start_s=max(0.0, float(series.time_s[event[0]]) - pre_s),
            end_s=min(duration_s, float(series.time_s[event[-1]]) + post_s),
            peak_s=float(series.time_s[peak]),
            peak_score=float(series.score[peak]),
            peak_box=series.box[peak].tolist(),
        )
        if windows and window.start_s <= windows[-1].end_s:
            last = windows[-1]
            last.end_s = max(last.end_s, window.end_s)
            if window.peak_score > last.peak_score:
                last.peak_s, last.peak_score, last.peak_box = window.peak_s, window.peak_score, window.peak_box
        else:
            windows.append(window)
    return windows
