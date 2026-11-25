import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from highlight.court import Court
from highlight.discovery import main as discover
from highlight.discovery_candidates import candidate_union, select_candidates
from highlight.discovery_objects import ObjectScout
from highlight.discovery_signals import empty_poses, video_metadata
from highlight.fusion import main as fuse
from highlight.fusion_review import validate_reviews
from highlight.fusion_signals import PoseIndex
from highlight.model_evidence import document, import_csv, load_streams, main as import_main, validate_stream
from highlight.model_mix import align_streams, annotate_units, motion_seeds, observation_rows
from highlight.sam import Detection


def court():
    return Court(160, 120, 10, [[0,60],[160,60],[160,120],[0,120]], [], "manual", "test")


def fixture(root):
    video, config = root/"source.mp4", root/"court.json"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160,120))
    for frame in range(80):
        image = np.zeros((120,160,3), np.uint8)
        x = 25+round(1.5*frame)
        image[55:95,x:x+14] = 255
        writer.write(image)
    writer.release()
    config.write_text(json.dumps({"polygon": court().polygon}))
    return video, config, video_metadata(video)


def provenance(name="tracknetv3"):
    return {"name": name, "checkpoint_id": "fixture-checkpoint-v1", "backend": "fixture", "run_id": "fixture-run"}


def stream(kind="observed", label="shuttlecock", frames=(0,2,4), track="0"):
    return {"model": provenance(), "rows": [{"frame_index": f, "time_s": f/10,
        "time_basis": "nominal_frame_rate", "crop": [0,0,160,120], "objects": [
            {"label": label, "box": [30+f*3,20,32+f*3,22], "confidence": None,
             "observation_kind": kind, "source_track_id": track}]} for f in frames]}


class FixtureDetector:
    loads = 0

    def __init__(self, *args, **kwargs):
        type(self).loads += 1
        self.device = "fixture"
        self.model = SimpleNamespace(names={0: "person", 1: "sports ball"})

    def detect(self, bgr, prompts):
        mask = cv2.inRange(bgr, (200,200,200), (255,255,255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return [Detection("person", .95, np.array([x,y,x+w,y+h]))]


class ModelContractTests(unittest.TestCase):
    def test_native_tracknet_csv_keeps_invisibility_null_confidence_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, _, meta = fixture(root)
            csv = root/"tracknet.csv"
            csv.write_text("Frame,Visibility,X,Y\n0,1,30,20\n1,0,0,0\n2,1,40,20\n")
            output = root/"model.json"
            with contextlib.redirect_stdout(io.StringIO()):
                import_main([str(video),str(csv),"--sport","badminton","--format","tracknet",
                    "--model-name","TrackNetV3","--checkpoint-id","fixture-v1","--backend","fixture",
                    "--label","shuttlecock","--kind","observed","--out",str(output)])
            streams = load_streams([output, output], meta, "badminton")
            self.assertEqual(len(streams), 1)  # Repeated import never creates independent support.
            self.assertEqual(streams[0]["rows"][1]["objects"], [])
            obj = streams[0]["rows"][0]["objects"][0]
            self.assertEqual(obj["point"], [30,20])
            self.assertIsNone(obj["confidence"])
            self.assertEqual(streams[0]["model"]["provenance"], "externally_declared")
            data = json.loads(output.read_text())
            for key, value in [("sport", "football"), ("width", 320), ("source_identity", {})]:
                changed = deepcopy(data)
                changed[key] = value
                output.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_streams([output], meta, "badminton")

    def test_box_export_preserves_multiple_tracks_and_temporal_clip_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, meta = fixture(root)
            csv = root/"samurai.csv"
            csv.write_text("frame_index,x,y,width,height,track_id\n0,30,55,10,40,A\n0,60,55,10,40,B\n1,35,55,10,40,A\n")
            data = import_csv(csv, meta, "boxes", "person", "tracked", provenance("samurai"), 20)
            self.assertEqual([r["frame_index"] for r in data["rows"]], [20,21])
            self.assertEqual(len(data["rows"][0]["objects"]), 2)
            self.assertEqual(data["rows"][0]["objects"][0]["box"], [30,55,40,95])
            self.assertEqual(data["rows"][0]["objects"][0]["observation_kind"], "tracked")

    def test_reject_invalid_geometry_confidence_frames_and_ambiguous_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, meta = fixture(root)
            changes = [lambda s: s["rows"][0].update(frame_index=-1),
                       lambda s: s["rows"][0].update(time_s=-.1),
                       lambda s: s["rows"][0].update(time_s=float("nan")),
                       lambda s: s["rows"][0]["objects"][0].update(box=[0,0,200,20]),
                       lambda s: s["rows"][0]["objects"][0].update(confidence=2),
                       lambda s: s["rows"][0]["objects"][0].update(point=[200,10]),
                       lambda s: s["model"].update(checkpoint_id="")]
            for change in changes:
                data = stream()
                change(data)
                with self.assertRaises(ValueError):
                    validate_stream(data, meta)
            first, second = root/"first.json", root/"second.json"
            first.write_text(json.dumps(document(meta, "badminton", [stream()])))
            changed = stream()
            changed["rows"][0]["objects"][0]["box"] = [60,20,62,22]
            second.write_text(json.dumps(document(meta, "badminton", [changed])))
            with self.assertRaises(ValueError):
                load_streams([first,second], meta, "badminton")


class MixBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.meta = {"frames": 100, "fps": 10, "width": 160, "height": 120, "source_identity": {"test": 1}}
        self.index = PoseIndex(empty_poses(), court())

    def test_external_only_motion_survives_selection_without_poses_or_sam(self):
        rows = observation_rows([stream()], self.index, self.meta)
        seeds = motion_seeds(rows, court(), "badminton", set())
        self.assertTrue(seeds)
        self.assertEqual(seeds[0]["court_association"], "unknown")
        units = candidate_union([], seeds, 10)
        annotate_units(units)
        self.assertTrue(select_candidates(units, 2, 10))
        self.assertFalse(units[0]["boundary_verified"])
        self.assertFalse(units[0]["model_mix"]["boundary_verification_added"])

    def test_each_sport_accepts_its_native_object_labels(self):
        for sport,label in [("badminton","shuttlecock"),("basketball","basketball"),("football","soccer ball")]:
            data = stream(label=label)
            rows = observation_rows([data],self.index,self.meta)
            self.assertTrue(motion_seeds(rows,court(),sport,set()),sport)

    def test_predicted_interpolated_wrong_sport_static_and_cross_cut_do_not_propose_motion(self):
        for kind in ["predicted", "interpolated"]:
            rows = observation_rows([stream(kind)], self.index, self.meta)
            self.assertFalse(any(o["accepted_for_display"] for row in rows for o in row["objects"]))
            self.assertEqual(motion_seeds(rows, court(), "badminton", set()), [])
        rows = observation_rows([stream()], self.index, self.meta)
        self.assertEqual(motion_seeds(rows, court(), "football", set()), [])
        self.assertEqual(motion_seeds(rows, court(), "badminton", {1,3}), [])
        data = stream()
        for row in data["rows"]:
            row["objects"][0]["box"] = [30,20,32,22]
        self.assertEqual(motion_seeds(observation_rows([data],self.index,self.meta), court(),"badminton",set()), [])

    def test_separate_models_track_ids_and_outside_players_never_form_motion_pairs(self):
        a, b = stream(frames=(0,)), stream(frames=(2,))
        b["model"]["run_id"] = "other"
        self.assertEqual(motion_seeds(observation_rows([a,b],self.index,self.meta),court(),"badminton",set()), [])
        data = stream()
        for i,row in enumerate(data["rows"]):
            row["objects"][0]["source_track_id"] = str(i)
        self.assertEqual(motion_seeds(observation_rows([data],self.index,self.meta),court(),"badminton",set()), [])
        data = stream(label="person")  # feet at y22, outside the supplied ROI
        self.assertEqual(motion_seeds(observation_rows([data],self.index,self.meta),court(),"badminton",set()), [])

    def test_source_decoder_pts_replaces_nominal_external_time_without_losing_it(self):
        data = stream()
        cv_rows = [{"time_s": f*.11} for f in range(100)]
        result = align_streams([data], cv_rows)
        self.assertEqual(result[0]["rows"][1]["time_s"], .22)
        self.assertEqual(result[0]["rows"][1]["reported_time_s"], .2)
        self.assertEqual(data["rows"][1]["time_s"], .2)

    def test_reused_tracker_id_does_not_join_across_camera_epochs(self):
        cv_rows = [{"time_s":f/10,"scene_change_candidate":f==3} for f in range(100)]
        rows = observation_rows(align_streams([stream()],cv_rows),self.index,self.meta)
        first, last = rows[0]["objects"][0], rows[-1]["objects"][0]
        self.assertEqual(first["source_track_id"],last["source_track_id"])
        self.assertNotEqual(first["tracklet_id"],last["tracklet_id"])

    def test_mixing_preserves_verified_whole_units_and_skips_instead_of_trimming(self):
        seeds = motion_seeds(observation_rows([stream()],self.index,self.meta), court(),"badminton",set())
        original = {"candidate_id":"rally", "start_s":0., "end_s":8., "ranking_score":.7,
                    "boundary_status":"verified", "boundary_verified":True, "events":[{"kind":"external_service"}]}
        result = candidate_union([original], seeds, 10)
        annotate_units(result)
        self.assertEqual(result[0]["events"], original["events"])
        self.assertEqual((result[0]["start_s"],result[0]["end_s"],result[0]["boundary_verified"]),(0,8,True))
        self.assertEqual(select_candidates(result, 5, 7), [])

    def test_object_budget_keeps_close_pairs_and_late_coverage_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/"model.pt"
            model.write_bytes(b"fixture")
            meta = {**self.meta, "frames": 36000, "fps":60}
            for budget in [0,1,2,7,600]:
                scout = ObjectScout(model, meta, court(), 2, budget)
                self.assertLessEqual(len(scout.frames), budget)
                self.assertIsNone(scout.detector)
                if budget >= 4:
                    frames = sorted(scout.frames)
                    self.assertEqual(frames[-1],35999)
                    self.assertTrue(any(b-a <= 30 for a,b in zip(frames,frames[1:])))
                if budget == 2:
                    self.assertTrue(all(17000 < f < 19000 for f in scout.frames))


class MixIntegrationTests(unittest.TestCase):
    def test_fresh_yolo_sam_cached_external_mix_replay_and_fusion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, config, meta = fixture(root)
            model, evidence = root/"fixture.pt", root/"external.json"
            model.write_bytes(b"fixture")
            evidence.write_text(json.dumps(document(meta,"badminton",[stream()])))
            out, replay, fused = root/"mixed", root/"replay", root/"fusion"
            common = [str(video),"--sport","badminton","--court",str(config),"--sam-model",str(model),
                      "--max-sam-calls","8","--motion-threshold",".5","--cv-width","160"]
            with patch("highlight.objects.YoloObjects",FixtureDetector), patch("highlight.sam.Sam3",FixtureDetector), contextlib.redirect_stdout(io.StringIO()):
                discover([*common,"--object-model",str(model),"--max-object-calls","8",
                          "--model-evidence",str(evidence),"--out-dir",str(out)])
            report = json.loads((out/"discovery.json").read_text())
            sources = {s for unit in report["all_units"] for s in unit["discovery"]["sources"]}
            self.assertTrue({"yolo_object", "model:tracknetv3", "sam"} <= sources)
            self.assertEqual(report["timing"]["object_calls"], 8)
            self.assertLessEqual(report["timing"]["sam_calls"], 8)
            self.assertEqual(report["coverage"]["cv_frames"],80)
            self.assertEqual(len(report["coverage"]["models"]),2)
            requests = json.loads((out/"sam_requests.json").read_text())
            self.assertTrue(any(r["reason"] == "model_motion_candidate" for r in requests))
            with patch("highlight.objects.YoloObjects",side_effect=AssertionError("replay must not load YOLO")), \
                 patch("highlight.sam.Sam3",side_effect=AssertionError("replay must not load SAM")), contextlib.redirect_stdout(io.StringIO()):
                discover([*common,"--sam-cache",str(out/"sam-cache.jsonl"),"--model-evidence",str(out/"model_evidence.json"),"--out-dir",str(replay)])
                fuse([str(video),"--sport","badminton","--court",str(config),"--candidates",str(out/"discovery.json"),
                      "--sam-evidence",str(out/"sam_evidence.json"),"--cv-width","160","--output-width","160", "--output-fps","10","--out-dir",str(fused)])
            repeated = json.loads((replay/"discovery.json").read_text())
            self.assertEqual(repeated["timing"]["object_calls"],0)
            self.assertEqual(repeated["timing"]["sam_calls"],0)
            self.assertEqual([u["discovery"]["evidence_refs"] for u in report["all_units"]],
                             [u["discovery"]["evidence_refs"] for u in repeated["all_units"]])
            manifest = json.loads((fused/"manifest.json").read_text())
            self.assertEqual(manifest["timing"]["object_calls"],0)
            self.assertEqual(manifest["coverage"]["mixed_model_samples"],11)
            overlay = cv2.VideoCapture(str(fused/manifest["outputs"]["full_overlay"]))
            self.assertEqual(round(overlay.get(cv2.CAP_PROP_FRAME_COUNT)),80)
            overlay.release()
            packets = [json.loads(line) for line in (fused/"packets.jsonl").read_text().splitlines()]
            obj = next(o for p in packets for o in p["discovery_evidence"] if o["observation_id"].startswith("model-"))
            packet = next(p for p in packets if obj in p["discovery_evidence"])
            review = {"candidate_id":packet["candidate_id"],"decision":"needs_review","reason":"fixture evidence",
                      "evidence_refs":[obj["observation_id"]],"counterevidence":[],"unknowns":["outcome"],"boundary_verified":False}
            self.assertEqual(validate_reviews([review],packets),[review])
            review["boundary_verified"] = True
            with self.assertRaises(ValueError):
                validate_reviews([review],packets)
