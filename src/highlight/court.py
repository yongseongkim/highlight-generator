"""Cheap court-line evidence and explicit court calibration for fixed cameras.

Hough lines alone cannot distinguish a badminton boundary from basketball markings
or an adjacent court. We expose the evidence and only filter players when a polygon
has been supplied, rather than silently choosing the wrong court.
"""

from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from highlight.court_geometry import draw_geometry, geometry_label, load_calibration


@dataclass
class Court:
    width: int
    height: int
    fps: float
    polygon: list[list[float]] | None
    lines: list[list[float]]
    source: str
    note: str
    net: list[list[float]] | None = None  # ground-level segment between this court's net posts
    calibration: dict | None = None  # floor template fitted to explicit landmarks, never to the ROI

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        if self.polygon is None:
            return True
        distance = cv2.pointPolygonTest(np.asarray(self.polygon, np.float32), tuple(map(float, point)), True)
        return distance >= -margin * min(self.width, self.height)

    def net_distance(self, point: np.ndarray, torso: float) -> float | None:
        """Distance to the finite ground-level net segment, in torso lengths."""
        if self.net is None:
            return None
        a, b = np.asarray(self.net, dtype=float)
        vector = b - a
        along = np.clip(np.dot(point - a, vector) / np.dot(vector, vector), 0, 1)
        return float(np.linalg.norm(point - a - along * vector) / max(torso, 1))


def load_polygon(path: Path, width: int, height: int) -> list[list[float]]:
    """Load four perimeter-ordered corners; coordinates can be normalized or pixels."""
    config = json.loads(path.read_text())
    polygon = np.asarray(config["polygon"], dtype=np.float64)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("Court polygon must contain four finite [x, y] corners in perimeter order")
    if config.get("normalized", False):
        polygon *= [width, height]
    if (polygon < 0).any() or (polygon > [width, height]).any():
        raise ValueError("Court corners must be inside the video image")
    contour = polygon.astype(np.float32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < width * height * 0.005:
        raise ValueError("Court corners must form a convex, non-crossing quadrilateral with nonzero area")
    return polygon.tolist()


def load_net(path: Path, width: int, height: int) -> list[list[float]] | None:
    config = json.loads(path.read_text())
    if config.get("net") is None:
        return None
    net = np.asarray(config["net"], dtype=float)
    if net.shape != (2, 2) or not np.isfinite(net).all():
        raise ValueError("Net ground segment must contain two finite [x, y] endpoints")
    if config.get("normalized", False):
        net *= [width, height]
    if (net < 0).any() or (net > [width, height]).any() or np.linalg.norm(net[1] - net[0]) < 0.05 * min(width, height):
        raise ValueError("Net ground segment must be inside the image and have nonzero length")
    return net.tolist()


def find_lines(bgr: np.ndarray) -> list[list[float]]:
    """Find long line segments at <=960 px width, without fitting a fictitious court."""
    height, width = bgr.shape[:2]
    scale = min(1.0, 960 / width)
    image = cv2.resize(bgr, (round(width * scale), round(height * scale)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 130)
    # Indoor fixed-camera recordings: suppress most wall/ceiling edges. This is
    # only a diagnostic, never an automatic player-exclusion boundary.
    edges[: round(image.shape[0] * 0.4)] = 0
    segments = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=65,
                               minLineLength=max(30, round(image.shape[1] * 0.12)), maxLineGap=18)
    if segments is None:
        return []
    candidates = sorted(segments.reshape(-1, 4), key=lambda s: -np.linalg.norm(s[2:] - s[:2]))
    kept = []
    for segment in candidates:
        a, b = segment[:2].astype(float), segment[2:].astype(float)
        direction = (b - a) / np.linalg.norm(b - a)
        duplicate = False
        for old in kept:
            c, d = old[:2], old[2:]
            old_direction = (d - c) / np.linalg.norm(d - c)
            distance = abs(direction[0] * (c - a)[1] - direction[1] * (c - a)[0])
            if abs(np.dot(direction, old_direction)) > 0.998 and distance < 6:
                duplicate = True
                break
        if not duplicate:
            kept.append(segment.astype(float))
        if len(kept) == 40:
            break
    return [(segment / scale).tolist() for segment in kept]


def inspect_court(video: Path, out_dir: Path, config: Path | None = None,
                  show_line_candidates: bool = False) -> Court:
    """Median of a few opening frames reduces moving-player edges; costs no model calls."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(f"Invalid video geometry/frame rate: {video}")
        samples = []
        last = min(max(count - 1, 0), round(fps * 4))
        for index in np.unique(np.linspace(0, last, 5).astype(int)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                samples.append(frame)
        if not samples:
            raise ValueError(f"Cannot read a court preview: {video}")
    finally:
        capture.release()
    background = np.median(np.stack(samples), axis=0).astype(np.uint8)
    polygon = load_polygon(config, width, height) if config is not None else None
    court = Court(width, height, fps, polygon, find_lines(background),
                  "manual" if polygon is not None else "lines_only",
                  "Player feet are filtered strictly to the supplied polygon."
                  if polygon is not None else
                  "Line candidates are diagnostic only. Multiple sports/courts are ambiguous; supply --court to restrict players.")
    court.net = load_net(config, width, height) if config is not None else None
    court.calibration = load_calibration(config, width, height) if config is not None else None
    preview = background.copy()
    draw_geometry(preview, court, show_line_candidates=show_line_candidates)
    label = geometry_label(court)
    if show_line_candidates:
        label += " | yellow: raw edges (debug)"
    cv2.putText(preview, label, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(preview, label, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_dir / f"{video.stem}_court.jpg"), preview):
        raise OSError("Could not save court preview")
    (out_dir / f"{video.stem}_court.json").write_text(json.dumps(asdict(court), indent=2, allow_nan=False))
    return court


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect court-line candidates with OpenCV; no GPU or model weights needed.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--court", type=Path, help="optional JSON with four court corners")
    parser.add_argument("--show-line-candidates", action="store_true", help="draw ambiguous raw edges for diagnostics only")
    args = parser.parse_args(argv)
    court = inspect_court(args.video, args.out_dir, args.court, args.show_line_candidates)
    print(f"{len(court.lines)} line candidates; {court.note}")
    print(f"saved {args.out_dir / (args.video.stem + '_court.jpg')}")


if __name__ == "__main__":
    main()
