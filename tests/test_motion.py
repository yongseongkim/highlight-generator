import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from highlight.court import Court
from highlight.game_units import SportEvent, rank_units
from highlight.motion import expand_seeds, find_seeds, motion_samples
from highlight.plays import main as plays_main


def pose_fixture(moving=True, scale=1):
    t = np.arange(0, 12, 0.05)
    k = np.zeros((len(t), 17, 3), np.float32)
    k[..., :2] = [200, 100]
    k[..., 2] = 0.95
    k[:, [11, 12], 1] = 180
    k[:, [9, 10], 1] = 150
    k[:, [15, 16], 1] = 260
    if moving:
        k[:, 10, 0] += np.interp(t, [0, 4, 4.3, 4.6, 12], [0, 0, 200, 0, 0])
    k[..., :2] *= scale
    return {"time_s": t, "sampled_time_s": t, "frame_index": np.arange(len(t)),
            "sampled_frame_index": np.arange(len(t)), "track_id": np.ones(len(t), np.int32),
            "box": np.tile(np.array([100, 50, 450, 280]) * scale, (len(t), 1)),
            "keypoints": k, "score": np.full(len(t), 0.95)}


def target_court(scale=1):
    return Court(640 * scale, 360 * scale, 20,
                 [[0, 0], [640 * scale, 0], [640 * scale, 360 * scale], [0, 360 * scale]],
                 [], "manual", "fixture")


def curve(end=30):
    return [{"time_s": float(t), "frame_index": i, "known_players": 2,
             "energy": 0.2, "anchor": {"track_id": 1}} for i, t in enumerate(np.arange(0, end + 0.01, 0.1))]


def peak(t, energy=3, court_id="main"):
    return SportEvent(t, "motion_peak", court_id, source="test_kinematics",
                      data={"energy": energy, "seed_threshold": 1.4})


class MotionSignalTests(unittest.TestCase):
    def test_fast_arm_without_overhead_produces_unverified_seed(self):
        samples, _ = motion_samples(pose_fixture(), target_court())
        seeds, _ = find_seeds(samples)
        self.assertTrue(seeds)
        self.assertTrue(all(4 <= seed.time_s <= 4.7 for seed in seeds))
        self.assertTrue(all(not seed.verified and seed.confidence is None for seed in seeds))
        self.assertEqual(find_seeds(motion_samples(pose_fixture(False), target_court())[0])[0], [])

    def test_scale_and_neighboring_court_do_not_change_seeds(self):
        baseline = pose_fixture()
        samples, _ = motion_samples(baseline, target_court())
        expected = find_seeds(samples)[0]
        scaled = find_seeds(motion_samples(pose_fixture(scale=2), target_court(scale=2))[0])[0]
        self.assertEqual([s.time_s for s in expected], [s.time_s for s in scaled])
        combined = {key: value.copy() for key, value in baseline.items()}
        for key in ("frame_index", "time_s", "track_id", "box", "keypoints", "score"):
            other = baseline[key].copy()
            if key == "track_id":
                other[:] = 9
            elif key == "keypoints":
                other[..., 0] += 700
            elif key == "box":
                other[:, [0, 2]] += 700
            combined[key] = np.concatenate([baseline[key], other])
        actual, diagnostics = motion_samples(combined, target_court())
        self.assertEqual(actual, samples)
        self.assertGreater(diagnostics["court_excluded_rows"], 0)

    def test_identity_jump_and_hidden_wrist_do_not_create_motion_peak(self):
        poses = pose_fixture(False)
        poses["keypoints"][100:, :, 0] += 250
        samples, diagnostics = motion_samples(poses, target_court())
        self.assertGreater(diagnostics["track_discontinuities"], 0)
        self.assertEqual(find_seeds(samples)[0], [])
        poses = pose_fixture()
        poses["keypoints"][:, 10, 2] = 0.1
        self.assertEqual(find_seeds(motion_samples(poses, target_court())[0])[0], [])

    def test_fast_body_lunge_can_seed_without_arm_motion_but_slow_walk_cannot(self):
        poses = pose_fixture(False)
        t = poses["time_s"]
        poses["keypoints"][:, :, 0] += np.interp(t, [0, 4, 4.4, 4.8, 12], [0, 0, 120, 0, 0])[:, None]
        seeds = find_seeds(motion_samples(poses, target_court())[0])[0]
        self.assertTrue(seeds)
        self.assertTrue(all(seed.data["arm_speed"] < 0.01 for seed in seeds))
        poses = pose_fixture(False)
        poses["keypoints"][:, :, 0] += (poses["time_s"] * 20)[:, None]
        self.assertEqual(find_seeds(motion_samples(poses, target_court())[0])[0], [])

    def test_isolated_jitter_and_constant_high_activity_are_not_dramatic_seeds(self):
        samples = curve()
        samples[100]["energy"] = 10
        self.assertEqual(find_seeds(samples)[0], [])
        for sample in samples:
            sample["energy"] = 5
        self.assertEqual(find_seeds(samples)[0], [])


class MotionContextTests(unittest.TestCase):
    def test_context_extends_through_lower_motion_and_stops_at_local_lulls(self):
        samples = curve()
        for sample in samples:
            if 6 <= sample["time_s"] <= 15:
                sample["energy"] = 1
        units = expand_seeds(samples, [peak(10)], 30, "badminton")
        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertLess(unit.start_s, 6)
        self.assertGreater(unit.end_s, 15)
        self.assertEqual((unit.start_reason, unit.end_reason), ("local_motion_lull", "local_motion_lull"))
        self.assertEqual(unit.boundary_status, "estimated")
        self.assertEqual(rank_units(units, [], verified_only=True), [])

    def test_no_lull_remains_bounded_without_fabricated_serve_or_landing(self):
        samples = curve(100)
        for sample in samples:
            sample["energy"] = 2
        units = expand_seeds(samples, [peak(50, energy=2.5)], 100, "badminton")
        unit = units[0]
        self.assertEqual((unit.start_s, unit.end_s), (38, 58))
        self.assertEqual(unit.start_reason, "context_limit")
        self.assertEqual(unit.end_reason, "context_limit")
        self.assertEqual([event.kind for event in unit.events], ["motion_peak"])

    def test_unknown_frames_are_visibility_gap_not_motion_lull(self):
        samples = curve()
        for sample in samples:
            sample["energy"] = 2
            if sample["time_s"] >= 13:
                sample["known_players"] = 0
                sample["energy"] = None
        unit = expand_seeds(samples, [peak(10)], 30, "badminton")[0]
        self.assertEqual(unit.end_reason, "visibility_gap")
        self.assertAlmostEqual(unit.end_s, 13)
        self.assertFalse(unit.end_verified)

    def test_overlapping_peaks_merge_without_duplicate_selected_footage(self):
        samples = curve()
        for sample in samples:
            if 7 <= sample["time_s"] <= 16:
                sample["energy"] = 1
        units = expand_seeds(samples, [peak(10, 4), peak(14, 3)], 30, "badminton")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].features["seed_count"], 2)
        self.assertLessEqual(units[0].end_s - units[0].start_s, 20)
        self.assertEqual(expand_seeds(samples, [peak(10, court_id="other")], 30, "badminton"), [])

    def test_invalid_context_settings_are_rejected(self):
        for settings in [{"before_s": 0}, {"after_s": float("nan")}, {"padding_s": -1}]:
            with self.assertRaises(ValueError):
                expand_seeds(curve(), [peak(10)], 30, "badminton", **settings)


class MotionCliTests(unittest.TestCase):
    def test_cli_produces_candidates_from_cached_motion_without_boundary_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "synthetic.mp4"
            # The archive supplies geometry/duration; no new model inference.
            video.touch()
            poses = pose_fixture()
            archive = root / "poses.npz"
            np.savez_compressed(archive, **poses, meta=json.dumps({"video": str(video), "fps": 20, "frames": 240,
                                                                  "width": 640, "height": 360}))
            court_path = root / "court.json"
            court_path.write_text(json.dumps({"polygon": target_court().polygon}))
            output = root / "out"
            plays_main([str(video), "--sport", "badminton", "--motion-seeds", "--poses", str(archive),
                        "--court", str(court_path), "--out-dir", str(output), "--no-clips"])
            report = json.loads((output / "synthetic_plays.json").read_text())
            self.assertEqual(report["mode"], "motion_seed_expansion")
            self.assertTrue(report["selected_units"])
            self.assertTrue(all(u["unit_type"] == "motion_candidate" and u["boundary_status"] == "estimated"
                                for u in report["selected_units"]))
            self.assertTrue(all(event["kind"] == "motion_peak" for event in report["events"]))
            self.assertEqual(report["timing"]["sam_calls"], report["timing"]["yolo_calls"])


if __name__ == "__main__":
    unittest.main()
