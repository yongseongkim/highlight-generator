import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from highlight.court import Court
from highlight.fusion import REVIEW_SCHEMA, main
from highlight.fusion_review import validate_reviews
from highlight.fusion_signals import PROFILES, DenseCv, ObjectTracklets, PoseIndex, RateSampler, evidence_packet, mark_stationary_sam, object_observations


def fixture():
    t = np.arange(0, 3, 0.2)
    k = np.zeros((len(t), 17, 3), np.float32)
    k[..., :2] = [80, 55]
    k[..., 2] = 0.95
    k[:, [5, 6], 1] = 35
    k[:, [11, 12], 1] = 65
    k[:, [9, 10], 1] = 45
    k[:, [15, 16], 1] = 100
    return {"sampled_frame_index": np.arange(0, 30, 2), "sampled_time_s": t,
            "frame_index": np.arange(0, 30, 2), "time_s": t, "keypoints": k,
            "track_id": np.ones(len(t), np.int32), "score": np.full(len(t), 0.95),
            "box": np.tile([60, 20, 100, 105], (len(t), 1))}


def court():
    return Court(160, 120, 10, [[0, 0], [160, 0], [160, 120], [0, 120]], [], "manual", "test")


class SamplingTests(unittest.TestCase):
    def test_sampling_has_low_base_high_candidate_rate_and_hard_budget(self):
        sampler = RateSampler(2, 8, [{"start_s": 1, "end_s": 2}], 5)
        self.assertTrue(sampler.due(0))
        self.assertFalse(sampler.due(0.2))
        self.assertTrue(sampler.due(0.5))
        self.assertTrue(sampler.due(1))
        self.assertTrue(sampler.due(1.125))
        self.assertTrue(sampler.due(1.25))
        self.assertFalse(sampler.due(1.5))
        self.assertEqual(sampler.calls, 5)

    def test_pose_lookup_never_uses_future_or_stale_frames(self):
        index = PoseIndex(fixture(), court())
        self.assertEqual(index.at(-1), (None, []))
        self.assertEqual(index.at(1)[0], 0)
        self.assertEqual(index.at(5)[0], 4)
        self.assertEqual(index.at(100), (None, []))

    def test_adjacent_player_cannot_supply_pose_or_object_association(self):
        poses = fixture()
        c = court()
        c.polygon = [[0, 0], [60, 0], [60, 120], [0, 120]]
        index = PoseIndex(poses, c)
        self.assertEqual(index.at(0)[1], [])
        result = object_observations([{"label": "badminton racket", "confidence": 0.9, "box": [75, 40, 85, 50]}], index, 0, 0, "sam")
        self.assertFalse(result[0]["accepted_for_display"])
        self.assertIsNone(result[0]["nearest_player_track_id"])


class GeometryAndTrackTests(unittest.TestCase):
    def test_repeated_stationary_shuttle_remains_raw_counterevidence(self):
        index = PoseIndex(fixture(), court())
        rows = [{"frame_index": frame, "time_s": t, "objects": object_observations([
                 {"label": "shuttlecock", "confidence": 0.8, "box": [75, 75, 78, 78]}], index, frame, t, "sam")}
                for frame, t in [(0, 0), (2, 0.2), (4, 0.4)]]
        self.assertEqual(mark_stationary_sam(rows, 160), 3)
        self.assertTrue(all(row["objects"][0]["stationary_candidate"] for row in rows))
        self.assertTrue(all(not row["objects"][0]["accepted_for_display"] for row in rows))
        self.assertEqual(rows[0]["objects"][0]["confidence"], 0.8)

    def test_flying_racket_above_ground_roi_can_associate_with_in_court_player(self):
        c = court()
        c.polygon = [[0, 80], [160, 80], [160, 120], [0, 120]]
        result = object_observations([{"label": "tennis racket", "confidence": 0.9, "box": [75, 40, 85, 50]}], PoseIndex(fixture(), c), 0, 0, "yolo")
        self.assertTrue(result[0]["accepted_for_display"])
        self.assertEqual(result[0]["association"], "near_in_court_player")
        self.assertEqual(result[0]["native_label"], "tennis racket")

    def test_invalid_boxes_and_floor_racket_are_not_supported(self):
        raw = [{"label": "racket", "confidence": 0.9, "box": [20, 100, 30, 110]},
               {"label": "racket", "confidence": 0.9, "box": [10, 10, 5, 20]},
               {"label": "racket", "confidence": float("nan"), "box": [10, 10, 20, 20]}]
        result = object_observations(raw, PoseIndex(fixture(), court()), 0, 0, "sam")
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["accepted_for_display"])

    def test_tracklets_preserve_label_and_break_after_missing_gap(self):
        tracker = ObjectTracklets(160, 120)
        def obs(label, t):
            return {"accepted_for_display": True, "confidence": 0.9, "native_label": label,
                    "center_normalized": [0.5, 0.5], "time_s": t, "tracklet_id": None}
        a, b, c, d = obs("sports ball", 0), obs("sports ball", 0.5), obs("tennis racket", 0.5), obs("sports ball", 2)
        tracker.update([a], 0)
        tracker.update([b, c], 0.5)
        self.assertEqual(a["tracklet_id"], b["tracklet_id"])
        self.assertNotEqual(b["tracklet_id"], c["tracklet_id"])
        tracker.update([d], 2)
        self.assertNotEqual(b["tracklet_id"], d["tracklet_id"])


class CvAndPacketTests(unittest.TestCase):
    def test_dense_cv_counts_only_roi_motion_and_refreshes_geometry_periodically(self):
        c = court()
        c.polygon = [[80, 0], [160, 0], [160, 120], [80, 120]]
        analyzer = DenseCv(c, PROFILES["badminton"], 160, 2)
        black = np.zeros((120, 160, 3), np.uint8)
        first, _, changed = analyzer.observe(black, 0, 0)
        self.assertEqual(first["roi_motion_fraction"], 0)
        self.assertTrue(changed)
        outside = black.copy()
        outside[20:50, 10:40] = 255
        row, _, changed = analyzer.observe(outside, 1, 0.1)
        self.assertEqual(row["roi_motion_fraction"], 0)
        self.assertFalse(changed)
        inside = outside.copy()
        inside[20:50, 100:130] = 255
        row, _, changed = analyzer.observe(inside, 2, 2)
        self.assertGreater(row["roi_motion_fraction"], 0)
        self.assertTrue(changed)

    def test_football_surface_is_a_diagnostic_not_an_event(self):
        green = np.full((120, 160, 3), [0, 160, 0], np.uint8)
        row, _, _ = DenseCv(court(), PROFILES["football"], 160).observe(green, 0, 0)
        self.assertEqual(row["grass_fraction"], 1)
        self.assertNotIn("goal", row)
        self.assertNotIn("shuttlecock", PROFILES["basketball"].native_labels)
        self.assertNotIn("sports ball", PROFILES["badminton"].native_labels)

    def test_packet_retains_unknowns_and_flags_stationary_observations(self):
        index = PoseIndex(fixture(), court())
        frames = []
        for t in [0, 0.5, 1]:
            obj = object_observations([{"label": "sports ball", "confidence": 0.9, "box": [70, 80, 80, 90]}], index, round(t*10), t, "yolo")[0]
            obj["tracklet_id"] = "ball-1"
            frames.append({"frame_index": round(t*10), "time_s": t, "objects": [obj]})
        packet = evidence_packet({"candidate_id": "C01", "start_s": 0, "end_s": 2}, [], index, frames, [], [],
            {"sport": "basketball", "object_model_status": "loaded", "object_checkpoint_classes": ["sports ball"], "missing_expected_classes": ["basketball"], "unknown_expected_classes": []})
        self.assertTrue(packet["object_tracklets"][0]["possibly_stationary"])
        self.assertTrue(packet["unknowns"])
        self.assertNotIn("goal", packet)
        self.assertEqual(REVIEW_SCHEMA["properties"]["reviews"]["items"]["properties"]["boundary_verified"], {"const": False})


class FusionIntegrationTests(unittest.TestCase):
    def test_full_overlay_and_packets_cover_the_whole_source_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
            for n in range(30):
                image = np.zeros((120, 160, 3), np.uint8)
                image[20:40, 40+n:60+n] = 200
                writer.write(image)
            writer.release()
            archive = root / "poses.npz"
            np.savez_compressed(archive, **fixture(), meta=json.dumps({"video": str(video), "width": 160, "height": 120, "fps": 10, "frames": 30}))
            config = root / "court.json"
            config.write_text(json.dumps({"polygon": court().polygon}))
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps({"video": str(video), "sport": "badminton", "units": [
                {"start_s": 0.2, "end_s": 2.7, "ranking_score": 1, "boundary_status": "estimated", "events": []}]}))
            output = root / "result"
            main([str(video), "--sport", "badminton", "--poses", str(archive), "--court", str(config),
                  "--candidates", str(candidates), "--out-dir", str(output), "--cv-width", "160", "--output-width", "160", "--output-fps", "10"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["coverage"]["cv_frames"], 30)
            self.assertEqual(manifest["coverage"]["cached_pose_frames"], 15)
            self.assertEqual(manifest["timing"]["object_calls"], 0)
            self.assertEqual(manifest["timing"]["sam_calls"], 0)
            self.assertEqual(manifest["coverage"]["object_model_status"], "not_run")
            self.assertEqual(manifest["coverage"]["missing_expected_classes"], [])
            self.assertIn("shuttlecock", manifest["coverage"]["unknown_expected_classes"])
            self.assertEqual(len((output / "opencv.jsonl").read_text().splitlines()), 30)
            packet = json.loads((output / "packets/C01.json").read_text())
            self.assertEqual(len(packet["keyframes"]), 3)
            cap = cv2.VideoCapture(str(output / "full_overlay.mp4"))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 30)
            self.assertTrue(cap.read()[0])
            cap.release()
            reused = root / "reused"
            common = [str(video), "--sport", "badminton", "--poses", str(archive),
                      "--candidates", str(candidates), "--object-evidence", str(output / "manifest.json"), "--no-video"]
            main([*common, "--court", str(config), "--out-dir", str(reused)])
            cached = json.loads((reused / "manifest.json").read_text())
            self.assertEqual(cached["coverage"]["object_model_status"], "not_run")
            self.assertEqual(cached["timing"]["object_calls"], 0)
            config.write_text(json.dumps({"polygon": [[0, 0], [150, 0], [150, 120], [0, 120]]}))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([*common, "--court", str(config), "--out-dir", str(root / "wrong-court")])


class AgentReviewTests(unittest.TestCase):
    def packet(self):
        return {"candidate_id": "C01", "keyframes": [{"path": "frame-000010.jpg"}],
                "player_context": [{"frame_index": 10}], "object_tracklets": [{"samples": [{"observation_id": "sam:10:0"}]}]}

    def review(self):
        return {"candidate_id": "C01", "decision": "highlight_candidate", "reason": "Active movement observed",
                "evidence_refs": ["frame-000010.jpg", "sam:10:0", "pose:10"], "counterevidence": [],
                "unknowns": ["Outcome unobserved"], "boundary_verified": False}

    def test_review_requires_real_candidate_and_observation_references(self):
        review = self.review()
        self.assertEqual(validate_reviews([review], [self.packet()]), [review])
        review["evidence_refs"] = ["invented-hit-10"]
        with self.assertRaises(ValueError):
            validate_reviews([review], [self.packet()])

    def test_agent_interest_cannot_create_boundary_verification_or_duplicate_candidates(self):
        review = self.review()
        review["boundary_verified"] = True
        with self.assertRaises(ValueError):
            validate_reviews([review], [self.packet()])
        review["boundary_verified"] = False
        with self.assertRaises(ValueError):
            validate_reviews([review, review], [self.packet()])


if __name__ == "__main__":
    unittest.main()
