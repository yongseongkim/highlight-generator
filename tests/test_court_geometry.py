import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from highlight.court import Court
from highlight.court_geometry import BADMINTON_LINES, draw_geometry, load_calibration, project


class CourtGeometryTests(unittest.TestCase):
    def load(self, landmarks, checks=None, normalized=False, **extra):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "court.json"
            path.write_text(json.dumps({"normalized": normalized, "calibration": {"sport": "badminton", "landmarks": landmarks,
                                                       "check_landmarks": checks or [], **extra}}))
            return load_calibration(path, 640, 480)

    def landmarks(self, points):
        h = np.array([[60, 8, 20], [0, 25, 30], [0.002, 0.003, 1.]])
        return [{"court_m": p, "image": q.tolist()} for p, q in zip(points, project(np.array(points), h))]

    def test_service_landmarks_fit_a_court_with_offscreen_back_corners(self):
        # Only visible service intersections are needed, not invented frame corners.
        points = [[0, 4.72], [6.1, 4.72], [6.1, 8.68], [0, 8.68]]
        rows = [{"court_m": p, "image": [p[0]*60 + 30, p[1]*60 - 160]} for p in points]
        calibration = self.load(rows, [{"court_m": [3.05, 4.72], "image": [213, 123.2]}])
        mapped = project(np.array([[0, 0], [6.1, 13.4]]), np.array(calibration["homography"]))
        np.testing.assert_allclose(mapped, [[30, -160], [396, 644]], atol=0.001)
        self.assertFalse(calibration["reviewed"])
        self.assertLess(max(calibration["check_error_px"]), 0.001)
        court = Court(640, 480, 30, None, [], "test", "test", calibration=calibration)
        image = np.zeros((480, 640, 3), np.uint8)
        draw_geometry(image, court)
        self.assertGreater(np.count_nonzero(image), 0)

    def test_bad_or_degenerate_landmarks_and_independent_check_fail(self):
        rows = self.landmarks([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])
        with self.assertRaises(ValueError):
            self.load(rows[:3])
        with self.assertRaises(ValueError):
            self.load(rows[:3] + [rows[0]])
        with self.assertRaises(ValueError):
            self.load(self.landmarks([[0, 0], [0, 1], [0, 2], [0, 3]]))
        with self.assertRaises(ValueError):
            self.load(rows, [{"court_m": [3.05, 4.72], "image": [100, 400]}])
        with self.assertRaises(ValueError):
            self.load(rows, [rows[0]])
        rows[0]["image"] = [float("nan"), 2]
        with self.assertRaises(ValueError):
            self.load(rows)

    def test_four_point_fit_is_not_automatic_review_or_player_roi(self):
        calibration = self.load(self.landmarks([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]]))
        self.assertFalse(calibration["reviewed"])
        self.assertIsNone(calibration["check_error_px"])
        court = Court(640, 480, 30, [[400, 0], [640, 0], [640, 480], [400, 480]], [], "test", "test", calibration=calibration)
        self.assertFalse(court.contains(np.array([50, 50])))
        self.assertTrue(court.contains(np.array([500, 50])))

    def test_normalized_fit_and_check_landmarks_are_stored_in_source_pixels(self):
        rows = self.landmarks([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])
        checks = self.landmarks([[3.05, 4.72]])
        expected = np.array(checks[0]["image"])
        for row in rows + checks:
            row["image"] = (np.array(row["image"]) / [640, 480]).tolist()
        calibration = self.load(rows, checks, normalized=True)
        np.testing.assert_allclose(calibration["check_landmarks"][0]["image"], expected)
        np.testing.assert_allclose(project(np.array([[3.05, 4.72]]), np.array(calibration["homography"]))[0], expected, atol=0.001)

    def test_raw_edges_are_opt_in_and_roi_is_not_a_solid_court(self):
        court = Court(640, 480, 30, [[10, 10], [630, 10], [630, 470], [10, 470]],
                      [[100, 200, 500, 200]], "test", "test")
        image = np.zeros((480, 640, 3), np.uint8)
        draw_geometry(image, court)
        self.assertFalse(image[200, 300].any())
        self.assertGreater(np.count_nonzero(image[10, 10:630, 0] == 0), 100)
        draw_geometry(image, court, show_line_candidates=True)
        self.assertTrue(image[200, 300].any())

    def test_template_keeps_center_service_lines_out_of_net_area(self):
        center = [line for line in BADMINTON_LINES if line[0][0] == line[1][0] == 3.05]
        self.assertEqual(center, [[[3.05, 0], [3.05, 4.72]], [[3.05, 8.68], [3.05, 13.4]]])


if __name__ == "__main__":
    unittest.main()
