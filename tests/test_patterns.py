from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from highlight.court import Court, load_net
from highlight.pattern_signals import Action, action_bursts, highlight_candidates, mine_actions, sam_support, select_actions
from highlight.patterns import already_sampled, load_evidence


def fixture(swing=True, translation=False, scale=1):
    times = np.arange(0, 2.51, 0.05)
    count = len(times)
    k = np.zeros((count, 17, 3), np.float32)
    k[..., 0] = 100
    k[..., 1] = 100
    k[..., 2] = 0.95
    k[:, 5, 0], k[:, 6, 0] = 80, 120
    k[:, [11, 12], 1] = 160
    k[:, 11, 0], k[:, 12, 0] = 85, 115
    k[:, [13, 14], 1] = 190
    k[:, 13, 0], k[:, 14, 0] = 80, 120
    k[:, [15, 16], 1] = 220
    k[:, 15, 0], k[:, 16, 0] = 75, 125
    k[:, 9, 0], k[:, 10, 0] = 75, 125
    k[:, [9, 10], 1] = 160
    if swing:
        k[:, 10, 1] = np.interp(times, [0, 0.3, 0.5, 0.85, 1.1, 2.5], [160, 160, 40, 40, 170, 170])
    if translation:
        k[..., 0] += (times * 40)[:, None]
    k[..., :2] *= scale
    return {"frame_index": np.arange(count), "sampled_frame_index": np.arange(count),
            "time_s": times, "sampled_time_s": times, "track_id": np.ones(count, np.int32),
            "keypoints": k, "box": np.tile(np.array([50, 20, 150, 225]) * scale, (count, 1)),
            "score": np.full(count, 0.95)}


def court(scale=1):
    return Court(640 * scale, 360 * scale, 20, None, [], "lines_only", "test")


def action():
    return Action("overhead_swing", 1, 1, 20, 0, 2.5, 0.8, [50, 20, 150, 225], {}, {})


class TemporalPatternTests(unittest.TestCase):
    def test_small_other_hand_movement_cannot_replace_a_full_swing(self):
        poses = fixture()
        poses["keypoints"][:, 9, 1] = 130
        poses["keypoints"][:, 9, 0] += np.interp(poses["time_s"], [0, 0.7, 0.95, 1.1, 1.4, 2.5], [0, 0, 18, 18, 0, 0])
        actions, _ = mine_actions(poses, court(), 2.5)
        self.assertTrue(any(a.pattern == "overhead_swing" for a in actions))

    def test_compact_net_touch_without_overhead_or_fast_swing(self):
        poses = fixture(swing=False)
        poses["keypoints"][:, 10, 1] = 130
        poses["keypoints"][:, 10, 0] += np.interp(poses["time_s"], [0, 0.6, 0.9, 1.05, 1.35, 2.5], [0, 0, 18, 18, 0, 0])
        c = court()
        c.net = [[30, 205], [200, 205]]
        actions, _ = mine_actions(poses, c, 2.5)
        net = [a for a in actions if a.pattern == "net_touch_candidate"]
        self.assertTrue(net)
        self.assertTrue(all(a.features["wrist_speed"] < 1.8 for a in net))
        self.assertFalse(any(a.pattern == "overhead_swing" for a in actions))
        c.net = [[30, 20], [200, 20]]
        actions, _ = mine_actions(poses, c, 2.5)
        self.assertFalse(any(a.pattern == "net_touch_candidate" for a in actions))
        self.assertTrue(any(a.pattern == "compact_stroke" for a in actions))

    def test_other_court_actions_cannot_change_patterns_or_create_bursts(self):
        poses = fixture()
        c = court()
        c.polygon = [[0, 0], [200, 0], [200, 300], [0, 300]]
        baseline, _ = mine_actions(poses, c, 2.5)
        combined = {key: value.copy() for key, value in poses.items()}
        for key in ("frame_index", "time_s", "track_id", "keypoints", "box", "score"):
            other = poses[key].copy()
            if key == "track_id":
                other[:] = 2
            elif key == "keypoints":
                other[..., 0] += 101  # feet are one pixel beyond the boundary
            elif key == "box":
                other[:, [0, 2]] += 101
            combined[key] = np.concatenate([poses[key], other])
        actual, diagnostics = mine_actions(combined, c, 2.5)
        self.assertEqual([(a.pattern, a.time_s, a.track_id) for a in actual],
                         [(a.pattern, a.time_s, a.track_id) for a in baseline])
        self.assertGreater(diagnostics["court_excluded_rows"], 0)
        self.assertEqual(action_bursts(actual), [])

    def test_net_calibration_has_finite_separate_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "court.json"
            path.write_text(json.dumps({"normalized": True, "net": [[0.2, 0.5], [0.8, 0.5]]}))
            self.assertEqual(load_net(path, 640, 360), [[128, 180], [512, 180]])
            path.write_text(json.dumps({"net": [[1, 1], [1, 1]]}))
            with self.assertRaises(ValueError):
                load_net(path, 640, 360)

    def test_ordered_raise_swing_followthrough_is_detected(self):
        actions, _ = mine_actions(fixture(), court(), 2.5)
        self.assertTrue(any(a.pattern == "overhead_swing" for a in actions))
        for a in actions:
            if a.pattern == "overhead_swing":
                self.assertLess(a.phases["preparation_s"], a.phases["swing_s"])
                self.assertLess(a.phases["swing_s"], a.phases["follow_through_s"])
                self.assertGreaterEqual(a.features["follow_through_drop"], 0.35)

    def test_static_arm_and_whole_body_translation_are_not_swings(self):
        for translation in (False, True):
            actions, _ = mine_actions(fixture(swing=False, translation=translation), court(), 2.5)
            self.assertEqual(actions, [])

    def test_motion_without_followthrough_is_not_a_completed_pattern(self):
        poses = fixture()
        poses["keypoints"][:, 10, 1] = np.interp(poses["time_s"], [0, 0.3, 0.5, 2.5], [160, 160, 40, 40])
        actions, _ = mine_actions(poses, court(), 2.5)
        self.assertEqual(actions, [])

    def test_scaling_does_not_change_pattern_score(self):
        a, _ = mine_actions(fixture(), court(), 2.5)
        b, _ = mine_actions(fixture(scale=2), court(2), 2.5)
        self.assertEqual(len(a), len(b))
        self.assertGreater(len(a), 0)
        self.assertAlmostEqual(a[0].pose_score, b[0].pose_score, places=5)

    def test_low_reach_requires_hand_recovery_not_just_leg_closure(self):
        poses = fixture(swing=False)
        t = poses["time_s"]
        intensity = np.interp(t, [0, 0.3, 0.55, 1.05, 1.25, 2.5], [0, 0, 1, 1, 0, 0])
        hand = np.interp(t, [0, 0.6, 0.85, 1, 1.2, 2.5], [0, 0, 1, 1, 0, 0])
        k = poses["keypoints"]
        k[:, [5, 6, 11, 12], 1] += (20 * intensity)[:, None]
        k[:, 15, 0] = 75 - 55 * intensity
        k[:, 16, 0] = 125 + 55 * intensity
        k[:, 10, 1] = 160 + 90 * hand
        actions, _ = mine_actions(poses, court(), 2.5)
        self.assertTrue(any(a.pattern == "low_reach_recovery" for a in actions))
        k[t >= 0.75, 10, 1] = 250  # stays bent/reaching after the legs close
        actions, _ = mine_actions(poses, court(), 2.5)
        self.assertFalse(any(a.pattern == "low_reach_recovery" for a in actions))

    def test_missing_wrist_or_identity_jump_cannot_create_a_swing(self):
        poses = fixture()
        poses["keypoints"][:, 10, 2] = 0.1
        self.assertEqual(mine_actions(poses, court(), 2.5)[0], [])
        poses = fixture(swing=False)
        poses["keypoints"][20:, :, 0] += 200
        actions, diagnostics = mine_actions(poses, court(), 2.5)
        self.assertEqual(actions, [])
        self.assertGreater(diagnostics["track_discontinuities"], 0)

    def test_court_boundary_filters_actions(self):
        c = court()
        c.polygon = [[300, 0], [600, 0], [600, 350], [300, 350]]
        actions, diagnostics = mine_actions(fixture(), c, 2.5)
        self.assertEqual(actions, [])
        self.assertGreater(diagnostics["court_excluded_rows"], 0)


class ObjectEvidenceTests(unittest.TestCase):
    def test_outside_court_player_cannot_supply_sam_support(self):
        poses = fixture()
        x, y = poses["keypoints"][20, 10, :2]
        detail = {"frame_index": 20, "objects": [{"label": "badminton racket", "confidence": 0.9,
                   "box": [float(x - 5), float(y - 10), float(x + 5), float(y + 5)]}]}
        c = court()
        c.polygon = [[0, 0], [200, 0], [200, 200], [0, 200]]  # player's feet are below this court
        evidence = sam_support(action(), poses, [detail], c)
        self.assertEqual(evidence["ranking_bonus"], 0)
        self.assertEqual(evidence["status"], "not_sampled")

    def test_saved_frame_reuse_requires_crop_to_cover_target_player(self):
        details = [{"frame_index": 20, "crop": [0, 0, 200, 300], "objects": []}]
        self.assertTrue(already_sampled(20, np.array([50, 20, 150, 225]), details))
        self.assertFalse(already_sampled(21, np.array([50, 20, 150, 225]), details))
        self.assertFalse(already_sampled(20, np.array([400, 20, 500, 225]), details))

    def test_evidence_chain_reuse_and_source_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            original, derived = root / "original.json", root / "derived.json"
            detail = {"frame_index": 20, "crop": [0, 0, 200, 300], "objects": []}
            original.write_text(json.dumps({"video": str(video), "windows": [{"details": [detail]}]}))
            derived.write_text(json.dumps({"video": str(video), "sam_evidence": str(original), "fresh_sam_details": [detail]}))
            _, details = load_evidence(derived, video)
            self.assertEqual(len(details), 1)
            with self.assertRaises(ValueError):
                load_evidence(derived, root / "other.mp4")
            original.write_text(json.dumps({"video": str(video), "sam_evidence": str(derived)}))
            with self.assertRaises(ValueError):
                load_evidence(derived, video)

    def test_floor_and_stationary_shuttles_do_not_add_ranking_support(self):
        details = [{"frame_index": frame, "objects": [
            {"label": "shuttlecock", "confidence": 0.9, "box": [90, 80, 94, 84]},
            {"label": "shuttlecock", "confidence": 0.9, "box": [90, 218, 94, 222]},
        ]} for frame in (14, 20, 26)]
        evidence = sam_support(action(), fixture(), details)
        self.assertEqual(evidence["ranking_bonus"], 0)
        self.assertEqual(evidence["moving_small_object_samples"], 0)
        self.assertGreater(evidence["rejected"].get("shuttle_near_floor", 0), 0)
        self.assertGreater(evidence["rejected"].get("stationary_shuttle", 0), 0)

    def test_only_a_plausible_racket_near_the_hand_adds_support(self):
        poses = fixture()
        x, y = poses["keypoints"][20, 10, :2]
        details = [{"frame_index": 20, "objects": [
            {"label": "badminton racket", "confidence": 0.8, "box": [float(x - 5), float(y - 15), float(x + 8), float(y + 3)]},
            {"label": "badminton racket", "confidence": 0.99, "box": [400, 20, 430, 60]},
        ]}]
        evidence = sam_support(action(), poses, details)
        self.assertEqual(evidence["racket_near_hand"], 1)
        self.assertAlmostEqual(evidence["ranking_bonus"], 0.07)
        self.assertEqual(evidence["rejected"]["racket_not_near_hand_or_oversized"], 1)

    def test_unsampled_action_is_unknown_rather_than_rejected(self):
        evidence = sam_support(action(), fixture(), [])
        self.assertEqual(evidence["status"], "not_sampled")
        self.assertEqual(evidence["ranking_bonus"], 0)

    def test_sam_crop_from_another_player_does_not_count_as_coverage(self):
        evidence = sam_support(action(), fixture(), [{"frame_index": 20, "crop": [400, 0, 600, 300], "objects": []}])
        self.assertEqual(evidence["status"], "not_sampled")


class SelectionTests(unittest.TestCase):
    def test_net_exchange_burst_does_not_need_an_overhead(self):
        a = replace(action(), pattern="net_touch_candidate")
        actions = [replace(a, time_s=2, start_s=1, end_s=3),
                   replace(a, track_id=2, time_s=4, start_s=3, end_s=5),
                   replace(a, time_s=6, start_s=5, end_s=7)]
        bursts = action_bursts(highlight_candidates(actions))
        self.assertEqual(len(bursts), 1)
        self.assertEqual(bursts[0].features["action_count"], 3)

    def test_ambiguous_low_reaches_are_review_only_by_default(self):
        swing = action()
        reach = replace(swing, pattern="low_reach_recovery", pose_score=0.99)
        self.assertEqual(highlight_candidates([swing, reach]), [swing])
        self.assertEqual(highlight_candidates([swing, reach], include_low_reach=True), [swing, reach])

    def test_repeated_actions_require_multiple_tracks_and_temporal_continuity(self):
        a = action()
        actions = [replace(a, time_s=2, start_s=1, end_s=3),
                   replace(a, track_id=2, time_s=4, start_s=3, end_s=5),
                   replace(a, time_s=6, start_s=5, end_s=7)]
        bursts = action_bursts(actions)
        self.assertEqual(len(bursts), 1)
        self.assertEqual(bursts[0].features["action_count"], 3)
        self.assertEqual(action_bursts([replace(a, track_id=1) for a in actions]), [])
        self.assertEqual(action_bursts([actions[0], actions[1], replace(actions[2], time_s=20)]), [])

    def test_clip_and_duration_budgets_without_overlapping_clips(self):
        a = action()
        actions = [a, replace(a, time_s=1.2, start_s=0.2, end_s=2.7),
                   replace(a, time_s=5, start_s=4, end_s=6), replace(a, time_s=9, start_s=8, end_s=10)]
        selected = select_actions(actions, max_highlights=3, budget_s=4.5)
        self.assertEqual(len(selected), 2)
        self.assertLessEqual(sum(a.end_s - a.start_s for a in selected), 4.5)
        self.assertTrue(all(a.end_s <= b.start_s for a, b in zip(selected, selected[1:])))


if __name__ == "__main__":
    unittest.main()
