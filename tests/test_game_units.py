import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from highlight.activity import activity_events
from highlight.game_units import SportEvent, badminton_units, rank_units, scoring_units
from highlight.pattern_signals import Action
from highlight.plays import main as plays_main, read_events


def event(time, kind, court="main", verified=True, **kwargs):
    return SportEvent(time, kind, court, source="test_observation", verified=verified, **kwargs)


class RallyTests(unittest.TestCase):
    def test_serve_to_landing_keeps_every_stroke_in_one_rally(self):
        observations = [event(3, "serve"), event(4, "hit", data={"stroke": "hairpin"}),
                        event(6, "hit", data={"stroke": "drive"}), event(8, "shuttle_landed")]
        units = badminton_units(observations, 20)
        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertEqual((unit.play_start_s, unit.play_end_s), (3, 8))
        self.assertEqual(unit.boundary_status, "verified")
        self.assertEqual(unit.unit_type, "rally")
        self.assertEqual([e.data.get("stroke") for e in unit.events if e.kind == "hit"], ["hairpin", "drive"])

    def test_brief_tracking_loss_does_not_end_or_split_a_rally(self):
        observations = [event(1, "serve"), event(2, "shuttle_missing"), event(4, "occluded"),
                        event(5, "tracking_resumed"), event(7, "shuttle_landed")]
        units = badminton_units(observations, 10)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].play_end_s, 7)

    def test_neighboring_court_cannot_start_or_end_selected_rally(self):
        observations = [event(1, "serve"), event(2, "serve", court="other"),
                        event(3, "shuttle_landed", court="other"), event(9, "shuttle_landed")]
        units = badminton_units(observations, 20)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].play_end_s, 9)
        self.assertTrue(all(e.court_id == "main" for e in units[0].events))

    def test_pose_pause_never_becomes_ground_contact_for_observed_serve(self):
        observations = [event(1, "serve"), event(3, "activity_pause", verified=False)]
        units = badminton_units(observations, 10, allow_activity_estimates=True)
        self.assertEqual(units[0].end_reason, "video_end_unknown")
        self.assertFalse(units[0].end_verified)

    def test_net_touch_is_not_net_fault(self):
        units = badminton_units([event(1, "serve"), event(3, "net_touch_candidate"), event(7, "fault")], 10)
        self.assertEqual(units[0].play_end_s, 7)
        self.assertEqual(units[0].end_reason, "fault")

    def test_missing_terminal_before_next_serve_remains_unverified(self):
        units = badminton_units([event(1, "serve"), event(6, "serve"), event(9, "shuttle_landed")], 12)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0].end_reason, "next_serve_without_terminal")
        self.assertFalse(units[0].end_verified)
        self.assertEqual(units[1].boundary_status, "verified")
        self.assertEqual(len(badminton_units([event(1, "serve"), event(1, "serve"), event(4, "fault")], 8)), 1)

    def test_duplicate_ground_frames_do_not_invent_new_rallies(self):
        units = badminton_units([event(1, "serve"), event(5, "shuttle_landed"), event(5.1, "shuttle_landed")], 10)
        self.assertEqual(len(units), 1)

    def test_mid_rally_video_start_is_not_a_verified_serve(self):
        units = badminton_units([event(1, "in_play"), event(5, "shuttle_landed")], 10)
        self.assertEqual(units[0].boundary_status, "partially_observed")
        self.assertFalse(units[0].start_verified)

    def test_padding_does_not_merge_distinct_adjacent_rallies(self):
        observations = [event(1, "serve"), event(4, "fault"), event(4.5, "serve"), event(7, "shuttle_landed")]
        units = badminton_units(observations, 10, pre_s=1, post_s=1)
        self.assertEqual(len(units), 2)
        self.assertGreater(units[0].end_s, units[1].start_s)
        self.assertEqual(units[0].play_end_s, 4)
        self.assertEqual(units[1].play_start_s, 4.5)

    def test_budget_skips_a_long_rally_instead_of_trimming_it(self):
        units = badminton_units([event(0, "serve"), event(80, "shuttle_landed"),
                                 event(90, "serve"), event(100, "shuttle_landed")], 120, pre_s=0, post_s=0)
        selected = rank_units(units, [], max_highlights=2, budget_s=20)
        self.assertEqual(len(selected), 1)
        self.assertEqual((selected[0].play_start_s, selected[0].play_end_s), (90, 100))
        self.assertEqual(units[0].play_end_s, 80)

    def test_estimated_motion_boundaries_are_explicit_opt_in(self):
        observations = [event(1, "activity_start", verified=False), event(8, "activity_pause", verified=False)]
        self.assertEqual(badminton_units(observations, 10), [])
        units = badminton_units(observations, 10, allow_activity_estimates=True)
        self.assertEqual(units[0].unit_type, "rally_candidate")
        self.assertEqual(units[0].boundary_status, "estimated")
        self.assertEqual(rank_units(units, [], verified_only=True), [])

    def test_sam_support_ranks_whole_rallies_without_changing_boundaries(self):
        units = badminton_units([event(1, "serve", verified=False), event(5, "shuttle_landed", verified=False),
                                 event(11, "serve", verified=False), event(15, "shuttle_landed", verified=False)], 20,
                                pre_s=0, post_s=0)
        actions = [Action("net_touch_candidate", 1, t, int(t * 30), t - 0.2, t + 0.2, 0.75,
                          [0, 0, 20, 100], {}, {}) for t in [3, 13]]
        actions[1].sam = {"ranking_bonus": 0.07, "status": "racket_near_hand"}
        selected = rank_units(units, actions, max_highlights=1)
        self.assertEqual((selected[0].play_start_s, selected[0].play_end_s), (11, 15))
        self.assertEqual(selected[0].features["sam_supported_action_count"], 1)
        self.assertEqual(selected[0].boundary_status, "estimated")
        self.assertEqual(rank_units(units, actions, verified_only=True), [])


class ScoringTests(unittest.TestCase):
    def test_basketball_scoring_candidate_includes_lead_up(self):
        units = scoring_units([event(20, "basket_made")], "basketball", 30)
        self.assertEqual((units[0].start_s, units[0].play_end_s, units[0].end_s), (12, 20, 23))

    def test_football_goal_uses_observed_attack_start(self):
        observations = [event(2, "attack_start"), event(24, "goal"), event(25, "goal", court="other")]
        units = scoring_units(observations, "football", 30)
        self.assertEqual(len(units), 1)
        self.assertEqual((units[0].start_s, units[0].play_end_s), (2, 24))
        self.assertEqual(units[0].boundary_status, "verified")

    def test_pose_and_ball_detections_cannot_manufacture_a_goal(self):
        observations = [event(1, "pose_activity"), event(2, "ball_detected"), event(3, "shot_attempt")]
        for sport in ["basketball", "football"]:
            self.assertEqual(scoring_units(observations, sport, 10), [])

    def test_duplicate_score_frames_collapse_but_distinct_ids_survive(self):
        self.assertEqual(len(scoring_units([event(10, "goal"), event(10.1, "goal")], "football", 20)), 1)
        observations = [event(10, "basket_made", event_id="score-1"), event(10.5, "basket_made", event_id="score-2")]
        self.assertEqual(len(scoring_units(observations, "basketball", 20)), 2)


class ActivityAndInputTests(unittest.TestCase):
    def test_cli_keeps_whole_event_defined_rallies_and_ignores_other_court(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "synthetic.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (160, 120))
            for _ in range(60):
                writer.write(np.full((120, 160, 3), 80, np.uint8))
            writer.release()
            events = root / "events.json"
            events.write_text(json.dumps({"video": str(video), "sport": "badminton", "source": "synthetic_fixture",
                                         "events": [{"time_s": 0.2, "kind": "serve", "verified": True},
                                                    {"time_s": 0.7, "kind": "shuttle_landed", "court_id": "other", "verified": True},
                                                    {"time_s": 1.0, "kind": "shuttle_landed", "verified": True},
                                                    {"time_s": 1.5, "kind": "serve", "verified": True},
                                                    {"time_s": 2.5, "kind": "fault", "verified": True}]}))
            plays_main([str(video), "--sport", "badminton", "--events", str(events), "--out-dir", str(root / "out"),
                        "--verified-only", "--no-clips"])
            report = json.loads((root / "out/synthetic_plays.json").read_text())
            self.assertEqual([(u["play_start_s"], u["play_end_s"]) for u in report["selected_units"]], [(0.2, 1.0), (1.5, 2.5)])
            self.assertTrue(all(u["boundary_status"] == "verified" for u in report["selected_units"]))
            self.assertEqual(report["timing"]["yolo_calls"], 0)
            self.assertEqual(report["timing"]["sam_calls"], 0)

    def test_missing_players_are_unknown_not_quiet(self):
        samples = [{"time_s": t, "known_players": 2, "active_players": 1, "quiet_players": 0} for t in [0, 0.3]]
        samples += [{"time_s": t, "known_players": 0, "active_players": 0, "quiet_players": 0} for t in [1, 2, 5]]
        self.assertEqual([e.kind for e in activity_events(samples)], ["activity_start"])
        samples += [{"time_s": t, "known_players": 2, "active_players": 0, "quiet_players": 2} for t in [6, 9]]
        events = activity_events(samples)
        self.assertEqual(events[-1].kind, "activity_pause")
        self.assertFalse(events[-1].verified)
        self.assertTrue(events[-1].data["not_a_ground_contact_observation"])

    def test_event_metadata_cannot_retag_other_court_as_ours(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "video.mp4"
            path = root / "events.json"
            path.write_text(json.dumps({"video": str(source), "sport": "badminton", "court_id": "other",
                                        "events": [{"time_s": 1, "kind": "serve"}, {"time_s": 5, "kind": "fault"}]}))
            events = read_events(path, source, "badminton")
            self.assertEqual(badminton_units(events, 10, court_id="main"), [])
            with self.assertRaises(ValueError):
                read_events(path, root / "different.mp4", "badminton")

    def test_invalid_event_timestamps_and_confidence_rejected(self):
        for t in [float("nan"), float("inf"), -1]:
            with self.assertRaises(ValueError):
                SportEvent(t, "serve")
        with self.assertRaises(ValueError):
            SportEvent(1, "serve", confidence=1.5)
        with self.assertRaises(ValueError):
            SportEvent(1, "serve", verified="false")


if __name__ == "__main__":
    unittest.main()
