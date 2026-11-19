"""Explicit floor landmark calibration, separate from player selection and Hough lines."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# Nominal BWF Diagram A dimensions (metres). These are display guides, not
# millimetre-accurate painted line edges or an automatic court identification.
BADMINTON_LINES = [
    [[0, 0], [6.1, 0]], [[0, 13.4], [6.1, 13.4]],
    [[0, 0], [0, 13.4]], [[6.1, 0], [6.1, 13.4]],
    [[0.46, 0], [0.46, 13.4]], [[5.64, 0], [5.64, 13.4]],
    [[0, 0.76], [6.1, 0.76]], [[0, 12.64], [6.1, 12.64]],
    [[0, 4.72], [6.1, 4.72]], [[0, 8.68], [6.1, 8.68]],
    [[3.05, 0], [3.05, 4.72]], [[3.05, 8.68], [3.05, 13.4]],
]


def project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mapped = np.c_[points.reshape(-1, 2), np.ones(points.size // 2)] @ homography.T
    if not np.isfinite(mapped).all() or (np.abs(mapped[:, 2]) < 1e-8).any():
        raise ValueError("Court projection reaches the camera horizon")
    return (mapped[:, :2] / mapped[:, 2:]).reshape(points.shape)


def load_calibration(path: Path, width: int, height: int) -> dict | None:
    """Fit only supplied floor correspondences; never infer identity from the ROI.

    image coordinates use the containing config's `normalized` convention.
    Optional check_landmarks are excluded from fitting and reported separately.
    Four fitting points alone cannot validate a court assignment.
    """
    config = json.loads(path.read_text())
    calibration = config.get("calibration")
    if calibration is None:
        return None
    if calibration.get("sport") != "badminton":
        raise ValueError("Court line calibration currently supports badminton only")

    def correspondences(rows):
        world = np.asarray([row["court_m"] for row in rows], dtype=float)
        image = np.asarray([row["image"] for row in rows], dtype=float)
        if world.ndim != 2 or world.shape[1:] != (2,) or image.shape != world.shape:
            raise ValueError("Landmarks need court_m and image [x, y] pairs")
        if config.get("normalized", False):
            image *= [width, height]
        if (not np.isfinite(world).all() or not np.isfinite(image).all()
                or (world < 0).any() or (world > [6.1, 13.4]).any()
                or (image < 0).any() or (image > [width, height]).any()):
            raise ValueError("Landmarks must be finite and inside the court/image")
        if len(np.unique(world, axis=0)) != len(world) or len(np.unique(image, axis=0)) != len(image):
            raise ValueError("Court landmarks must be distinct")
        return world, image

    world, image = correspondences(calibration.get("landmarks", []))
    if len(world) < 4:
        raise ValueError("Court calibration needs at least four floor landmarks")
    # Reject a degenerate DLT, including three collinear points in a four-point fit.
    def normalize(points):
        centered = points - points.mean(axis=0)
        spread = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
        if spread < 1e-8:
            raise ValueError("Degenerate court landmarks")
        return centered / spread
    a, b = normalize(world), normalize(image)
    design = []
    for (x, y), (u, v) in zip(a, b):
        design.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                       [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("Court landmarks cannot determine a homography")
    homography, _ = cv2.findHomography(world, image, method=0)
    if homography is None or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("Invalid court homography")
    corners = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])
    denominators = np.c_[corners, np.ones(4)] @ homography[2]
    if (np.abs(denominators) < 1e-8).any() or denominators.min() * denominators.max() <= 0:
        raise ValueError("Court spans the camera horizon")
    # Full corners may be outside the image. Only clicked landmarks must be visible.
    projected = project(corners, homography)
    if (np.abs(projected) > max(width, height) * 1000).any():
        raise ValueError("Court projection is unstable near the camera horizon")
    if not cv2.isContourConvex(projected.astype(np.float32)):
        raise ValueError("Court projection is not a convex quadrilateral")
    fit_error = np.linalg.norm(project(world, homography) - image, axis=1)
    tolerance = max(3.0, width * 0.004)
    if fit_error.max() > tolerance:
        raise ValueError("Court landmarks disagree; inspect the floor intersections")
    checks = calibration.get("check_landmarks", [])
    check_error = None
    canonical_checks = []
    if checks:
        check_world, check_image = correspondences(checks)
        if any(np.all(world == point, axis=1).any() for point in check_world):
            raise ValueError("Check landmarks must be independent of fitting landmarks")
        check_error = np.linalg.norm(project(check_world, homography) - check_image, axis=1)
        canonical_checks = [{"court_m": w.tolist(), "image": p.tolist()} for w, p in zip(check_world, check_image)]
        if check_error.max() > tolerance:
            raise ValueError("Court calibration does not match the independent check landmarks")
    reviewed = calibration.get("reviewed", False)
    if not isinstance(reviewed, bool):
        raise ValueError("calibration.reviewed must be true or false")
    return {"sport": "badminton", "homography": homography.tolist(),
            "landmarks": [{"court_m": w.tolist(), "image": p.tolist()} for w, p in zip(world, image)],
            "check_landmarks": canonical_checks, "reviewed": reviewed,
            "fit_error_px": fit_error.tolist(), "check_error_px": check_error.tolist() if checks else None,
            "note": "Floor template only. Reprojection error does not verify court identity or shuttle landing."}


def dashed_line(image, a, b, color, thickness=1, dash=10):
    ok, a, b = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), tuple(map(int, a)), tuple(map(int, b)))
    if not ok:
        return
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    length = np.linalg.norm(b - a)
    for start in np.arange(0, length, dash * 2):
        p = a + (b - a) * start / max(length, 1)
        q = a + (b - a) * min(start + dash, length) / max(length, 1)
        cv2.line(image, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)), color, thickness)


def draw_geometry(image, court, *, show_line_candidates=False, lines=None):
    """ROI is dashed grey; calibrated floor lines are cyan; diagnostics opt in."""
    scale = np.array([image.shape[1] / court.width, image.shape[0] / court.height])
    if court.polygon is not None:
        polygon = np.rint(np.asarray(court.polygon) * scale).astype(int)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            dashed_line(image, a, b, (180, 180, 180))
    calibration = court.calibration
    if calibration is not None:
        segments = project(np.array(BADMINTON_LINES), np.array(calibration["homography"])) * scale
        for a, b in np.rint(segments).astype(int):
            if calibration["reviewed"]:
                ok, a, b = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), tuple(a), tuple(b))
                if ok:
                    cv2.line(image, a, b, (255, 220, 0), 2)
            else:
                dashed_line(image, a, b, (255, 220, 0), 2)
    # A manually supplied net segment is also an estimate, not a painted floor line.
    if court.net is not None:
        a, b = np.rint(np.asarray(court.net) * scale).astype(int)
        dashed_line(image, a, b, (220, 160, 60), 1, 6)
    if show_line_candidates:
        for line in court.lines if lines is None else lines:
            a, b = np.rint(np.asarray(line).reshape(2, 2) * scale).astype(int)
            cv2.line(image, tuple(a), tuple(b), (0, 190, 230), 1)


def geometry_label(court):
    net = " | blue dashed: net ground estimate" if court.net is not None else ""
    if court.calibration is None:
        return "Court UNCALIBRATED | dashed grey: player ROI (not court lines)" + net
    status = "reviewed" if court.calibration["reviewed"] else "PROPOSED / needs review"
    return f"Cyan: floor template ({status}) | dashed grey: player ROI" + net
