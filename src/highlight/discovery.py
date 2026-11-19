"""Find review candidates through independent CV, optional poses and budgeted SAM."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import math
import os
from pathlib import Path
import time

from highlight.court import Court, load_net, load_polygon
from highlight.court_geometry import load_calibration
from highlight.discovery_candidates import candidate_union, cv_seeds, observations, sam_motion_seeds, select_candidates, validate_units
from highlight.discovery_plan import context_crops, cv_peaks, refinement_probes, reservations, sampling_coverage, scout_probes, spread, uncertainty_probes
from highlight.discovery_sam import SamExecutor, validate_detail
from highlight.discovery_signals import scan_video, source_poses, video_metadata, pose_discovery_seeds
from highlight.fusion_signals import PROFILES, PoseIndex
from highlight.discovery_objects import ObjectScout
from highlight.model_evidence import document, load_streams, unique_streams
from highlight.model_mix import align_streams, annotate_units, coverage as model_coverage, motion_seeds as model_motion_seeds, observation_rows
from highlight.patterns import _cut_review_clip, load_evidence


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_review(directory: Path, report: dict) -> None:
    cards = []
    for unit in report["units"]:
        media = unit.get("clip", os.path.relpath(report["video"], directory)+f"#t={unit['start_s']:.3f},{unit['end_s']:.3f}")
        cards.append(f'<article><h2>{html.escape(unit["candidate_id"])} · {unit["start_s"]:.2f}–{unit["end_s"]:.2f}초</h2>'
                     f'<p>발견 경로: {html.escape(", ".join(unit["discovery"]["sources"]))} · 경계 상태: '
                     f'{html.escape(str(unit.get("boundary_status", "unknown")))}</p>'
                     f'<video controls preload="none" src="{html.escape(media, quote=True)}"></video>'
                     '<p>관심도 검토 후보입니다. 포즈와 연결되지 않은 객체의 코트·타구·득점은 미확인입니다.</p></article>')
    (directory/"review.html").write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>독립 탐색 후보</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;line-height:1.6}video{width:100%}article{margin:24px 0;padding:16px;background:#f3f5f7}</style>
<h1>포즈에 종속되지 않는 후보 탐색</h1><p>CV·포즈·SAM·선택적 객체/추적 모델 관측을 합친 검토 후보입니다.
샘플링 공백은 미확인이며, 장면의 완결성이나 경기 결과를 검증한 것은 아닙니다.</p>
<p><a href="discovery.json">후보·성능·관측 범위</a> · <a href="sam_requests.json">SAM 요청</a> ·
<a href="sam_observations.json">SAM 관측</a> · <a href="model_observations.json">혼합 모델 관측</a></p>'''
        + ('<p>계획 모드: 신규 SAM 추론을 실행하지 않았습니다.</p>' if report["settings"]["plan_only"] else '')
        + ''.join(cards) + ('<p>현재 근거에서 선택된 후보가 없습니다.</p>' if not cards else '') + '</html>')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--sport", required=True, choices=sorted(PROFILES))
    parser.add_argument("--court", type=Path, required=True)
    parser.add_argument("--court-id", default="main")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--candidates", type=Path, help="optional existing whole candidate intervals")
    parser.add_argument("--sam-evidence", type=Path)
    parser.add_argument("--sam-cache", type=Path, help="exact-input JSONL cache, shared across runs")
    parser.add_argument("--sam-model", type=Path, default=Path("models/sam3.pt"))
    parser.add_argument("--object-model", type=Path, help="optional local YOLO objects, independent of poses; shares CV decoding")
    parser.add_argument("--object-fps", type=float, default=2, help="target scouting rate; finite budgets spread across the source")
    parser.add_argument("--max-object-calls", type=int, default=600)
    parser.add_argument("--model-evidence", type=Path, action="append", default=[], help="repeatable source-bound external/cached model JSON")
    parser.add_argument("--max-sam-calls", type=int, default=90)
    parser.add_argument("--scout-interval", type=float, default=5)
    parser.add_argument("--context-tiles", type=int, choices=[1,2], default=1)
    parser.add_argument("--motion-threshold", type=float, default=.04)
    parser.add_argument("--cv-width", type=int, default=640)
    parser.add_argument("--pre-seconds", type=float, default=12)
    parser.add_argument("--post-seconds", type=float, default=8)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--highlight-budget", type=float, default=120)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--export-clips", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/discovery"))
    args = parser.parse_args(argv)
    if args.max_sam_calls < 0 or args.max_object_calls < 0 or args.max_candidates < 0 or args.cv_width < 32:
        parser.error("Budgets must be nonnegative and CV width >=32")
    if not all(math.isfinite(v) and v >= 0 for v in [args.pre_seconds, args.post_seconds, args.highlight_budget]):
        parser.error("Context/duration budgets must be finite and nonnegative")
    if args.pre_seconds+args.post_seconds <= 0 or not math.isfinite(args.scout_interval) or args.scout_interval <= 0:
        parser.error("Scout interval and total context must be positive")
    if not math.isfinite(args.motion_threshold) or not 0 < args.motion_threshold <= 1:
        parser.error("Motion threshold must be in (0, 1]")
    if not math.isfinite(args.object_fps) or args.object_fps <= 0:
        parser.error("Object FPS must be finite and positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Use a fresh output directory; share --sam-cache to reuse inference")
    started = time.perf_counter()
    if not args.plan_only and args.max_sam_calls:
        from highlight.sam import require_weights
        require_weights(args.sam_model)
    try:
        meta = video_metadata(args.video)
        duration = meta["frames"]/meta["fps"]
        court = Court(meta["width"], meta["height"], meta["fps"], load_polygon(args.court, meta["width"], meta["height"]),
                      [], "manual", "Supplied player ROI, not verified physical court geometry.",
                      load_net(args.court, meta["width"], meta["height"]), load_calibration(args.court, meta["width"], meta["height"]))
        if court.calibration and court.calibration["sport"] != args.sport:
            raise ValueError("Court calibration sport differs")
        index = PoseIndex(source_poses(args.poses, meta), court)
        streams = load_streams(args.model_evidence, meta, args.sport)
        scout = (ObjectScout(args.object_model, meta, court, args.object_fps, args.max_object_calls, args.device)
                 if args.object_model and args.max_object_calls and not args.plan_only else None)
        existing, legacy = [], []
        if args.candidates:
            saved = json.loads(args.candidates.read_text())
            if Path(saved["video"]).resolve() != args.video.resolve() or saved["sport"] != args.sport:
                raise ValueError("Candidate report video/sport differs")
            if saved.get("source_identity", meta["source_identity"]) != meta["source_identity"]:
                raise ValueError("Candidate source video changed")
            if saved.get("court") and saved["court"]["polygon"] != court.polygon:
                raise ValueError("Candidate player ROI differs")
            if saved.get("court_id", args.court_id) != args.court_id:
                raise ValueError("Candidate court ID differs")
            existing = [u for u in saved.get("units", saved.get("selected_units", [])) if u.get("court_id", args.court_id) == args.court_id]
            validate_units(existing, duration)
        if args.sam_evidence:
            saved, legacy = load_evidence(args.sam_evidence, args.video)
            if saved.get("court") and (saved["court"]["width"], saved["court"]["height"]) != (court.width, court.height):
                raise ValueError("SAM source geometry differs")
            for detail in legacy:
                validate_detail(detail, meta)
        cache = args.sam_cache or args.out_dir/"sam-cache.jsonl"
        executor = (SamExecutor(args.video, meta, args.sam_model, ["person", *PROFILES[args.sport].prompts],
                               args.max_sam_calls, cache, args.device)
                    if not args.plan_only and args.max_sam_calls else None)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    timing = {"load_saved_data_s": time.perf_counter()-started, "pose_calls": 0,
              "cached_pose_frames": len(index.frames)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv_rows, timing["dense_pass_wall_s"] = scan_video(args.video, index, args.sport, meta, args.cv_width, observer=scout)
    timing.update(scout.metrics() if scout else {"object_calls": 0, "object_load_s": 0., "object_inference_s": 0.,
                                               "object_observe_s": 0., "object_planned_frames": 0})
    timing["cv_decode_analysis_s"] = max(0., timing["dense_pass_wall_s"]-timing["object_observe_s"])
    if scout is not None:
        streams.append(scout.stream())
        # Sequential stages do not retain both large model instances on the GPU.
        scout.detector = None
    model_analysis_started = time.perf_counter()
    streams = unique_streams(align_streams(streams, cv_rows), meta)
    model_rows = observation_rows(streams, index, meta)
    blocked = {r["frame_index"] for r in cv_rows if r["scene_change_candidate"] or r["camera_unstable"]}
    object_seeds = model_motion_seeds(model_rows, court, args.sport, blocked)
    timing["model_evidence_analysis_s"] = time.perf_counter()-model_analysis_started
    dump(args.out_dir/"model_evidence.json", document(meta, args.sport, streams))
    dump(args.out_dir/"model_observations.json", model_rows)
    (args.out_dir/"opencv.jsonl").write_text(''.join(json.dumps(r, allow_nan=False)+"\n" for r in cv_rows))
    quota = reservations(args.max_sam_calls)
    crops = context_crops(court, args.context_tiles)
    peaks = cv_peaks(cv_rows, args.motion_threshold)
    plans = scout_probes(cv_rows, crops, quota["scout"], args.scout_interval, duration)
    plans += uncertainty_probes(peaks, crops, quota["uncertainty"], duration)
    details = list(legacy)
    if executor:
        details += executor.execute(plans)
    analysis_started = time.perf_counter()
    sam_rows = observations(details, index, meta)
    # Cached hits do not silently expand the search on an otherwise identical rerun.
    remaining = max(0, args.max_sam_calls-len({(p.frame_index, p.crop) for p in plans}))
    seen = {(d["frame_index"], tuple(d["crop"])) for d in details}
    pose_seeds = pose_discovery_seeds(index.poses, court, args.court_id)
    pose_contexts = candidate_union([], pose_seeds, duration, args.pre_seconds, args.post_seconds)
    model_contexts = candidate_union([], object_seeds, duration, args.pre_seconds, args.post_seconds)
    for context in model_contexts:
        context["sam_detail_reason"] = "model_motion_candidate"
    refinement = refinement_probes(sam_rows, existing+pose_contexts+model_contexts, cv_rows, court, remaining, seen)
    cached_analysis_s = time.perf_counter()-analysis_started
    if executor:
        details += executor.execute(refinement)
    analysis_started = time.perf_counter()
    if executor:
        sam_rows = observations(details, index, meta)
    requests = executor.requests if executor else [{**asdict(p), "status": "planned"} for p in plans+refinement]
    seeds = pose_seeds + cv_seeds(peaks) + sam_motion_seeds(sam_rows, court, {"person", *PROFILES[args.sport].native_labels}, blocked_frames=blocked)
    seeds += object_seeds
    all_units = candidate_union(existing, seeds, duration, args.pre_seconds, args.post_seconds)
    annotate_units(all_units)
    units = select_candidates(all_units, args.max_candidates, args.highlight_budget)
    timing["cached_evidence_analysis_s"] = cached_analysis_s+time.perf_counter()-analysis_started
    timing.update(executor.metrics() if executor else {"sam_calls": 0, "sam_cache_hits": 0, "sam_load_s": 0.,
                  "sam_inference_s": 0., "sam_decode_s": 0., "new_calls_by_branch": {}})
    dump(args.out_dir/"sam_requests.json", requests)
    dump(args.out_dir/"sam_observations.json", sam_rows)
    dump(args.out_dir/"sam_evidence.json", {"video": meta["video"], "source_identity": meta["source_identity"],
         "court": asdict(court), "sam_evidence": str(args.sam_evidence.resolve()) if args.sam_evidence else None,
         "fresh_sam_details": executor.details if executor else [], "new_inference_calls": timing["sam_calls"]})
    export_started = time.perf_counter()
    if args.export_clips:
        for unit in units:
            name = unit["candidate_id"]+".mp4"
            _cut_review_clip(args.video, unit["start_s"], unit["end_s"], args.out_dir/name)
            unit["clip"] = name
    timing.update(clip_export_s=time.perf_counter()-export_started, total_s=time.perf_counter()-started)
    report = {"schema_version": 1, "pipeline": "independent_discovery", "video": meta["video"],
              "source_identity": meta["source_identity"], "sport": args.sport, "court_id": args.court_id,
              "court": asdict(court), "duration_s": duration, "units": units, "all_units": all_units,
              "timing": timing, "reservations": quota,
              "coverage": {"cv_frames": len(cv_rows), "legacy_sam_images": len(legacy), "sam_images": len(sam_rows),
                           "models": model_coverage(streams, meta),
                           "by_crop": sampling_coverage(sam_rows, duration), "selected_seconds": sum(u["end_s"]-u["start_s"] for u in units)},
              "outputs": {"model_evidence": "model_evidence.json", "model_observations": "model_observations.json"},
              "settings": {k: [str(p) for p in v] if k == "model_evidence" else str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "limitations": ["Review candidates only; interest does not verify contact, game outcomes or boundaries.",
                              "SAM position links are uncertain; airborne court attribution remains unknown without supporting evidence.",
                              "A finite budget leaves temporal/spatial gaps. Low pose/CV activity never disables independent scouting."]}
    dump(args.out_dir/"discovery.json", report)
    write_review(args.out_dir, report)
    print(json.dumps({"candidates": len(units), "cv_frames": len(cv_rows), "new_sam_calls": timing["sam_calls"],
                      "sam_cache_hits": timing["sam_cache_hits"], "new_object_calls": timing["object_calls"], "total_s": timing["total_s"]}), flush=True)
    print(f"saved {args.out_dir/'review.html'}", flush=True)


if __name__ == "__main__":
    main()
