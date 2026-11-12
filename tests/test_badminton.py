"""Behavior checks for selection, calibration, detail budgets, and cache safety."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from highlight.badminton import analyze_details
from highlight.badminton_events import Moment, keyframe_plan, rank_moments
from highlight.candidates import Window
from highlight.court import Court, find_lines, load_polygon
from highlight.detect import _poses_match, main as detect_main
from highlight.pose import file_identity, save_poses
from highlight.sam import Detection


def poses_fixture(seconds=20, moving=True, x=100):
    count = seconds * 10
    keypoints = np.zeros((count, 17, 3), np.float32)
    keypoints[..., 0] = x
    keypoints[..., 1] = 100
    keypoints[..., 2] = 1
    keypoints[:, [11, 12], 1] = 150
    keypoints[:, [15, 16], 1] = 200
    keypoints[:, [9, 10], 1] = (np.where(np.arange(count) % 2, 20, 70) if moving else np.full(count, 20))[:, None]
    return {"sampled_frame_index": np.arange(count) * 6, "sampled_time_s": np.arange(count) / 10,
            "frame_index": np.arange(count) * 6, "time_s": np.arange(count) / 10,
            "track_id": np.ones(count, np.int32), "keypoints": keypoints,
            "box": np.tile([x - 30, 20, x + 30, 200], (count, 1)), "score": np.ones(count)}


def court_fixture(polygon=None):
    return Court(640, 360, 60, polygon, [], "manual" if polygon else "lines_only", "test")


def moment_fixture(peak=2.0, frame=120, score=0.8):
    return Moment(Window(max(0, peak - 1), peak + 1, peak, score, [20, 30, 50, 80]), frame, 1,
                  "overhead_motion_candidate", 0.8, 4, 0)


class SelectionTests(unittest.TestCase):
    def test_static_raised_arms_are_not_strokes(self):
        moments, _ = rank_moments(poses_fixture(moving=False), court_fixture(), 20)
        self.assertEqual(moments, [])

    def test_outside_court_player_cannot_trigger_highlight(self):
        court = court_fixture([[200, 0], [600, 0], [600, 350], [200, 350]])
        moments, selection = rank_moments(poses_fixture(x=100), court, 20)
        self.assertEqual(moments, [])
        self.assertEqual(selection["excluded_player_rows"], 200)

    def test_highlight_duration_count_and_overlap_budgets(self):
        moments, selection = rank_moments(poses_fixture(seconds=40), court_fixture(), 40,
                                          max_highlights=3, budget_s=12)
        self.assertGreater(len(moments), 0)
        self.assertLessEqual(len(moments), 3)
        self.assertLessEqual(selection["selected_video_s"], 12)
        for a, b in zip(moments, moments[1:]):
            self.assertLessEqual(a.window.end_s, b.window.start_s)
        for moment in moments:
            self.assertGreaterEqual(moment.window.start_s, 0)
            self.assertLessEqual(moment.window.end_s, 40)
        json.dumps([asdict(m) for m in moments], allow_nan=False)

    def test_new_or_stale_tracks_do_not_invent_temporal_velocity(self):
        poses = poses_fixture(seconds=1)
        poses["track_id"] = np.arange(10)
        moments, _ = rank_moments(poses, court_fixture(), 1)
        self.assertEqual(moments, [])
        poses = poses_fixture(seconds=1)
        poses["sampled_time_s"] = np.arange(10, dtype=float)
        moments, _ = rank_moments(poses, court_fixture(), 10)
        self.assertEqual(moments, [])

    def test_long_game_selection_covers_later_active_periods(self):
        poses = poses_fixture(seconds=180)
        moments, selection = rank_moments(poses, court_fixture(), 180, max_highlights=6, budget_s=36)
        self.assertEqual(selection["temporal_bins"], 6)
        occupied = {min(5, int(moment.window.peak_s / 30)) for moment in moments}
        self.assertEqual(occupied, set(range(6)))
        self.assertTrue(any(moment.window.peak_s > 150 for moment in moments))

    def test_invalid_keypoints_and_empty_videos_produce_strict_json(self):
        poses = poses_fixture(seconds=1)
        poses["keypoints"][:] = np.nan
        moments, selection = rank_moments(poses, court_fixture(), 1)
        self.assertEqual(moments, [])
        json.dumps(selection, allow_nan=False)
        empty, selection = rank_moments(poses_fixture(seconds=0), court_fixture(), 0)
        self.assertEqual(empty, [])
        json.dumps(selection, allow_nan=False)


class KeyframeTests(unittest.TestCase):
    def test_hard_call_budget_prioritizes_best_moments(self):
        moments = [moment_fixture(2, 120, 0.6), moment_fixture(6, 360, 0.9), moment_fixture(10, 600, 0.8)]
        plan = keyframe_plan(moments, 60, 1200, per_moment=3, max_calls=2)
        self.assertEqual({i for i, _ in plan}, {1, 2})
        self.assertEqual(len(plan), 2)
        self.assertEqual(keyframe_plan(moments, 60, 1200, max_calls=0), [])

    def test_frame_limits_deduplication_and_pts_independent_peak(self):
        moments = [moment_fixture(peak=10, frame=50)]
        plan = keyframe_plan(moments, 2, 51, per_moment=9)
        self.assertIn((0, 50), plan)
        self.assertEqual(len(plan), len(set(plan)))
        self.assertTrue(all(0 <= frame < 51 for _, frame in plan))

    def test_object_coordinates_return_to_source_image_and_misses_are_retained(self):
        class FakeDetector:
            calls = 0

            def detect(self, image, prompts):
                self.calls += 1
                self.image_shape = image.shape
                return [Detection("shuttlecock", 0.8, np.array([1, 2, 5, 6]))] if self.calls == 1 else []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (160, 120))
            for _ in range(40):
                writer.write(np.full((120, 160, 3), 100, np.uint8))
            writer.release()
            poses = poses_fixture(seconds=1)
            poses["box"] = np.tile([20, 30, 50, 80], (10, 1))
            detector = FakeDetector()
            details = analyze_details(path, poses, [moment_fixture(frame=12)], [(0, 2), (0, 20)], detector,
                                      ["shuttlecock"], Path(directory), False)
            self.assertEqual(detector.calls, 2)
            self.assertEqual(len(details[0]), 2)
            self.assertEqual(details[0][1]["objects"], [])
            crop = details[0][0]["crop"]
            self.assertEqual(details[0][0]["objects"][0]["box"], [crop[0] + 1, crop[1] + 2, crop[0] + 5, crop[1] + 6])


class CourtTests(unittest.TestCase):
    def test_lines_are_detected_without_inventing_polygon(self):
        image = np.zeros((540, 960, 3), np.uint8)
        cv2.line(image, (100, 500), (350, 230), (255, 255, 255), 4)
        cv2.line(image, (800, 500), (600, 230), (255, 255, 255), 4)
        self.assertGreater(len(find_lines(image)), 0)
        self.assertEqual(find_lines(np.zeros_like(image)), [])

    def test_calibration_rejects_crossed_and_out_of_bounds_corners(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "court.json"
            for polygon in ([[0, 0], [1, 1], [1, 0], [0, 1]], [[0, 0], [2, 0], [1, 1], [0, 1]]):
                path.write_text(json.dumps({"normalized": True, "polygon": polygon}))
                with self.assertRaises(ValueError):
                    load_polygon(path, 640, 360)
            path.write_text(json.dumps({"normalized": True, "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}))
            self.assertEqual(load_polygon(path, 640, 360)[2], [640, 360])


class CacheTests(unittest.TestCase):
    def test_source_settings_and_checkpoint_changes_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, model, archive = root / "video.mp4", root / "weights.pt", root / "poses.npz"
            video.write_bytes(b"first video")
            model.write_bytes(b"first model")
            meta = {"stride": 6, "input_width": 1280, "conf": 0.25,
                    "source_identity": file_identity(video), "model_identity": file_identity(model)}
            save_poses(archive, [], meta)
            self.assertTrue(_poses_match(archive, 6, str(model), video, 1280, 0.25))
            self.assertFalse(_poses_match(archive, 6, str(model), video, 1920, 0.25))
            self.assertFalse(_poses_match(archive, 3, str(model), video, 1280, 0.25))
            other = root / "other.mp4"
            other.write_bytes(video.read_bytes())
            self.assertFalse(_poses_match(archive, 6, str(model), other, 1280, 0.25))
            video.write_bytes(b"replacement video")
            self.assertFalse(_poses_match(archive, 6, str(model), video, 1280, 0.25))
            video.write_bytes(b"first video")
            meta["source_identity"] = file_identity(video)
            save_poses(archive, [], meta)
            model.write_bytes(b"replacement model")
            self.assertFalse(_poses_match(archive, 6, str(model), video, 1280, 0.25))


class ExistingSportTests(unittest.TestCase):
    def test_football_keeps_original_defaults_and_reuses_matching_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, model = root / "video.mp4", root / "weights.pt"
            video.write_bytes(b"source")
            model.write_bytes(b"checkpoint")
            calls = []

            def fake_extract(argv):
                calls.append(argv)
                save_poses(root / "video_poses.npz", [],
                           {"video": str(video), "fps": 30, "frames": 90, "stride": 3,
                            "model": model.name, "input_width": 1920, "conf": 0.25,
                            "source_identity": file_identity(video), "model_identity": file_identity(model)})

            argv = [str(video), "--sport", "football", "--model", str(model), "--out-dir", str(root),
                    "--no-sam", "--no-overlay", "--no-clips"]
            with patch("highlight.detect.resolve_device", return_value="mps"), patch("highlight.detect.extract_pose.main", side_effect=fake_extract):
                detect_main(argv)
                detect_main(argv)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index("--stride") + 1], "3")
            self.assertEqual(calls[0][calls[0].index("--input-width") + 1], "1920")
            report = json.loads((root / "video_highlights.json").read_text())
            self.assertTrue(report["timing"]["pose_reused"])
            self.assertEqual(report["sport"], "football")
            self.assertEqual(report["events"], [])
            self.assertEqual(report["play_candidates"], [])


if __name__ == "__main__":
    unittest.main()
