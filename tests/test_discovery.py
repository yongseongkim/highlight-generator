import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from highlight.court import Court
from highlight.discovery import main
from highlight.discovery_candidates import candidate_union, observations, sam_motion_seeds, select_candidates
from highlight.discovery_plan import Probe, context_crops, cv_peaks, detail_crop, refinement_probes, reservations, sampling_coverage, scout_probes, uncertainty_probes
from highlight.discovery_sam import SamExecutor
from highlight.discovery_signals import empty_poses, video_metadata
from highlight.fusion import main as fusion_main
from highlight.fusion_review import validate_reviews
from highlight.fusion_signals import PoseIndex
from highlight.patterns import load_evidence
from highlight.sam import Detection


def court():
    return Court(160, 120, 10, [[0, 60], [160, 60], [160, 120], [0, 120]], [], "manual", "test")


def cv_rows(count=100):
    return [{"frame_index": i, "time_s": i/10, "roi_motion_fraction": 0., "scene_change_candidate": False,
             "camera_unstable": False, "usable_pose": False} for i in range(count)]


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for frame in range(80):
        image = np.zeros((120, 160, 3), np.uint8)
        image[55:95, 25+frame:39+frame] = 255
        writer.write(image)
    writer.release()


class VisibleRectangleDetector:
    """Deterministic test detector follows a real moving rectangle in the fixture."""
    loads = 0

    def __init__(self, *args, **kwargs):
        type(self).loads += 1

    def detect(self, bgr, prompts):
        mask = cv2.inRange(bgr, (200,200,200), (255,255,255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return [Detection("person", .95, np.array([x,y,x+w,y+h]))]


class PlanTests(unittest.TestCase):
    def test_scouting_has_reserved_slots_without_pose_motion_or_candidates(self):
        for budget in [0,1,2,3,9,90]:
            allocation = reservations(budget)
            self.assertEqual(sum(allocation.values()), budget)
            self.assertEqual(allocation["scout"] > 0, budget > 0)
        rows = cv_rows(6000)
        crops = context_crops(court(), 2)
        probes = scout_probes(rows, crops, 27, 5, 600)
        self.assertEqual(len(probes), 27)
        self.assertLess(probes[0].frame_index, 600)
        self.assertGreater(probes[-1].frame_index, 5400)
        self.assertEqual(set(p.crop for p in probes), set(crops))
        self.assertLess(crops[0][1], 60)  # Include air above the ground polygon.

    def test_uncertainty_uses_in_court_missing_pose_and_excludes_camera_changes(self):
        rows = cv_rows()
        for r in rows:
            r["roi_motion_fraction"] = .1
        rows[20]["camera_unstable"] = True
        rows[21]["scene_change_candidate"] = True
        peaks = cv_peaks(rows, .04)
        self.assertTrue(peaks)
        self.assertFalse(any(p["frame_index"] in [20,21,22] for p in peaks))
        probes = uncertainty_probes(peaks, context_crops(court()), 3, 10)
        self.assertEqual(len(probes), 3)
        for p in peaks:
            p["usable_pose"] = True
        self.assertEqual(uncertainty_probes(peaks, context_crops(court()), 3, 10), [])

    def test_sampling_reports_actual_per_crop_gaps_including_video_edges(self):
        coverage = sampling_coverage([{"crop": [0,0,160,120], "time_s": 2},
                                      {"crop": [0,0,160,120], "time_s": 6}], 10)
        self.assertEqual(next(iter(coverage.values()))["max_gap_including_edges_s"], 4)

    def test_followups_diversify_people_and_equipment_and_obey_zero_slots(self):
        people, equipment = [20,55,40,100], [100,80,110,90]
        rows = [{"frame_index": i*20, "time_s": i*2., "objects": [
            {"native_label": "person", "box": people, "confidence": .8},
            {"native_label": "badminton racket", "box": equipment, "confidence": .99}]} for i in range(4)]
        probes = refinement_probes(rows, [], cv_rows(), court(), 8, set())
        crops = {p.crop for p in probes}
        self.assertIn(detail_crop(people,160,120), crops)
        self.assertIn(detail_crop(equipment,160,120), crops)
        self.assertEqual(refinement_probes(rows, [], cv_rows(), court(), 0, set()), [])


class SeedTests(unittest.TestCase):
    def details(self, boxes, frames=None):
        frames = frames or list(range(len(boxes)))
        return [{"frame_index": frame, "decoder_time_s": frame/10, "crop": [0,0,160,120],
                 "objects": [{"label": "badminton racket", "confidence": .9, "box": box}]} for box, frame in zip(boxes, frames)]

    def rows(self, details):
        return observations(details, PoseIndex(empty_poses(), court()), {"frames": 100,"fps": 10,"width": 160,"height": 120})

    def test_sam_can_create_new_candidate_without_pose_or_cv_and_keep_unknown_association(self):
        rows = self.rows(self.details([[40,15,50,25], [48,15,58,25]]))
        self.assertTrue(all(not r["objects"][0]["accepted_for_display"] for r in rows))
        seeds = sam_motion_seeds(rows, court(), {"badminton racket"})
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["court_association"], "unknown")
        units = candidate_union([], seeds, 10)
        self.assertEqual(units[0]["discovery"]["sources"], ["sam"])
        self.assertEqual(units[0]["decision"], "needs_review")
        self.assertFalse(units[0]["boundary_verified"])

    def test_static_same_frame_missing_and_long_gap_observations_are_not_motion(self):
        box = [40,15,50,25]
        cases = [[], self.details([box]), self.details([box,box]),
                 self.details([box,[50,15,60,25]], [0,20])]
        same_frame = self.details([box,[50,15,60,25]], [0,0])
        same_frame[1]["crop"] = [0,0,159,120]
        cases.append(same_frame)
        for details in cases:
            self.assertEqual(sam_motion_seeds(self.rows(details), court(), {"badminton racket"}), [])

    def test_scene_change_between_samples_cannot_be_sam_motion(self):
        rows = self.rows(self.details([[40,15,50,25],[50,15,60,25]], [0,3]))
        self.assertEqual(sam_motion_seeds(rows, court(), {"badminton racket"}, blocked_frames={1}), [])

    def test_small_steps_accumulate_in_observed_history(self):
        rows = self.rows(self.details([[40,15,50,25],[43,15,53,25],[46,15,56,25]], [0,3,6]))
        self.assertEqual(len(sam_motion_seeds(rows, court(), {"badminton racket"})), 1)

    def test_sam_person_outside_player_roi_does_not_supply_action_evidence(self):
        details = self.details([[40,15,50,25],[50,15,60,25]])
        for detail in details:
            detail["objects"][0]["label"] = "person"
        self.assertEqual(sam_motion_seeds(self.rows(details), court(), {"person"}), [])
        restricted = court()
        restricted.polygon = [[0,60],[160,60],[160,90],[0,90]]
        details = self.details([[40,55,50,100],[50,55,60,100]])
        for detail in details:
            detail["objects"][0]["label"] = "person"
        rows = observations(details, PoseIndex(empty_poses(), restricted), {"frames":100,"fps":10,"width":160,"height":120})
        self.assertTrue(all(not row["objects"][0]["accepted_for_display"] for row in rows))
        self.assertEqual(sam_motion_seeds(rows, restricted, {"person"}), [])

    def test_existing_whole_unit_boundaries_survive_new_seeds_and_duration_budget(self):
        existing = {"candidate_id": "original", "start_s": 1., "end_s": 7., "ranking_score": 9.,
                    "boundary_status": "verified", "events": [{"kind": "observed_service"}]}
        seed = {"time_s": 3., "frame_index": 30, "source": "sam", "score": .8,
                "evidence_refs": ["actual:30"], "reason": "test", "court_association": "unknown"}
        original = deepcopy(existing)
        units = candidate_union([existing], [seed], 10)
        self.assertEqual(existing, original)
        for key in ["start_s", "end_s", "boundary_status", "events"]:
            self.assertEqual(units[0][key], original[key])
        self.assertEqual(select_candidates(units, 2, 5), [])

    def test_generated_contexts_stay_bounded_disjoint_and_channels_get_a_turn(self):
        seeds = [{"time_s": t, "frame_index": int(t*10), "source": source, "score": score,
                  "evidence_refs": [], "reason": "test", "court_association": "unknown"}
                 for t,source,score in [(10,"cv",.9),(12,"sam",.8),(40,"sam",.6),(70,"cv",.7)]]
        units = candidate_union([], seeds, 100)
        self.assertTrue(all(u["end_s"]-u["start_s"] <= 20 for u in units))
        self.assertTrue(all(a["end_s"] <= b["start_s"] for a,b in zip(units,units[1:])))
        selected = select_candidates(units, 2, 40)
        sources = {s for u in selected for s in u["discovery"]["sources"]}
        self.assertEqual(sources, {"sam","cv"})


class ExecutorTests(unittest.TestCase):
    def test_global_cap_and_exact_request_cache_without_model_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, model, cache = root/"source.mp4", root/"sam.pt", root/"cache.jsonl"
            make_video(video)
            model.write_bytes(b"test checkpoint identity")
            meta = video_metadata(video)
            probes = [Probe(frame, (0,0,160,120), branch, "test") for frame,branch in [(20,"scout"),(23,"detail"),(26,"detail")]]
            runner = SamExecutor(video, meta, model, ["person"], 2, cache, detector_factory=VisibleRectangleDetector)
            self.assertEqual(len(runner.execute(probes)), 2)
            self.assertEqual(runner.calls, 2)
            self.assertEqual(runner.requests[-1]["status"], "budget_exhausted")
            def forbidden():
                raise AssertionError("Exact cache replay must not load a model")
            replay = SamExecutor(video, meta, model, ["person"], 0, cache, detector_factory=forbidden)
            self.assertEqual(len(replay.execute(probes[:2])), 2)
            self.assertEqual((replay.calls,replay.hits), (0,2))
            changed = SamExecutor(video, meta, model, ["different prompt"], 0, cache, detector_factory=forbidden)
            self.assertEqual(changed.entries, {})
            crop = Probe(20, (0,0,150,120), "scout", "test")
            self.assertEqual(replay.execute([crop]), [])
            source_changed = deepcopy(meta)
            source_changed["source_identity"]["size"] += 1
            self.assertEqual(SamExecutor(video, source_changed, model, ["person"], 0, cache).entries, {})
            model.write_bytes(b"changed checkpoint")
            self.assertEqual(SamExecutor(video, meta, model, ["person"], 0, cache).entries, {})

    def test_corrupted_cache_is_rejected_and_changed_source_evidence_cannot_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, model, cache = root/"source.mp4", root/"sam.pt", root/"cache.jsonl"
            make_video(video)
            model.write_bytes(b"checkpoint")
            meta = video_metadata(video)
            runner = SamExecutor(video, meta, model, ["person"], 1, cache, detector_factory=VisibleRectangleDetector)
            runner.execute([Probe(20,(0,0,160,120),"scout","test")])
            data = json.loads(cache.read_text())
            data["detail"]["frame_index"] = 21
            cache.write_text(json.dumps(data)+"\n")
            with self.assertRaises(ValueError):
                SamExecutor(video, meta, model, ["person"], 1, cache)
            evidence = root/"evidence.json"
            evidence.write_text(json.dumps({"video": str(video), "source_identity": meta["source_identity"]}))
            with video.open("ab") as output:
                output.write(b"changed")
            with self.assertRaises(ValueError):
                load_evidence(evidence, video)


class DiscoveryIntegrationTests(unittest.TestCase):
    def test_no_pose_or_existing_candidates_discovery_cache_fusion_and_agent_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, config, model = root/"source.mp4", root/"court.json", root/"sam.pt"
            make_video(video)
            config.write_text(json.dumps({"polygon": court().polygon}))
            model.write_bytes(b"fixture")
            output, replay, fused = root/"discovered", root/"replayed", root/"fusion"
            common = [str(video), "--sport","badminton","--court",str(config),"--sam-model",str(model),
                      "--max-sam-calls","8","--motion-threshold",".5","--cv-width","160"]
            VisibleRectangleDetector.loads = 0
            with patch("highlight.sam.Sam3", VisibleRectangleDetector), contextlib.redirect_stdout(io.StringIO()):
                main([*common,"--out-dir",str(output)])
                loads = VisibleRectangleDetector.loads
                main([*common,"--out-dir",str(replay),"--sam-cache",str(output/"sam-cache.jsonl")])
            report = json.loads((output/"discovery.json").read_text())
            repeated = json.loads((replay/"discovery.json").read_text())
            self.assertTrue(report["units"])
            self.assertTrue(all(u["discovery"]["sources"] == ["sam"] for u in report["units"]))
            self.assertEqual(report["coverage"]["cv_frames"], 80)
            self.assertEqual(report["timing"]["cached_pose_frames"], 0)
            self.assertLessEqual(report["timing"]["sam_calls"], 8)
            self.assertGreater(report["timing"]["new_calls_by_branch"]["scout"], 0)
            self.assertEqual(repeated["timing"]["sam_calls"], 0)
            self.assertEqual(VisibleRectangleDetector.loads, loads)
            with contextlib.redirect_stdout(io.StringIO()):
                fusion_main([str(video),"--sport","badminton","--court",str(config),
                             "--candidates",str(output/"discovery.json"),"--sam-evidence",str(output/"sam_evidence.json"),
                             "--cv-width","160","--output-width","160","--output-fps","10","--out-dir",str(fused)])
            cap = cv2.VideoCapture(str(fused/"full_overlay.mp4"))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 80)
            cap.release()
            packets = [json.loads(line) for line in (fused/"packets.jsonl").read_text().splitlines()]
            self.assertTrue(packets[0]["discovery_evidence"])
            evidence_id = packets[0]["discovery_evidence"][0]["observation_id"]
            review = {"candidate_id": packets[0]["candidate_id"], "decision": "needs_review", "reason": "객체 움직임 관측",
                      "evidence_refs": [evidence_id], "counterevidence": [], "unknowns": ["코트 관계 미확인"], "boundary_verified": False}
            self.assertEqual(validate_reviews([review],packets), [review])

    def test_plan_only_without_checkpoint_or_pose_and_bad_inputs_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, config = root/"source.mp4", root/"court.json"
            make_video(video)
            config.write_text(json.dumps({"polygon": court().polygon}))
            common = [str(video),"--sport","badminton","--court",str(config),"--plan-only", "--sam-model",str(root/"absent.pt")]
            output = root/"plan"
            with patch("highlight.sam.Sam3", side_effect=AssertionError("No model")), contextlib.redirect_stdout(io.StringIO()):
                main([*common,"--out-dir",str(output)])
            report = json.loads((output/"discovery.json").read_text())
            self.assertEqual(report["timing"]["sam_calls"], 0)
            self.assertTrue(json.loads((output/"sam_requests.json").read_text()))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([*common,"--scout-interval","nan","--out-dir",str(root/"bad")])
            self.assertFalse((root/"bad").exists())


if __name__ == "__main__":
    unittest.main()
