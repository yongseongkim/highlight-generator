"""Fast badminton workflow: court evidence -> sparse pose -> ranking -> keyframe details."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from highlight import extract_pose
from highlight.badminton_events import Moment, keyframe_plan, rank_moments
from highlight.court import inspect_court
from highlight.pose import load_poses
from highlight.sam import Sam3, require_weights
from highlight.video import resolve_device


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pose-fps", type=float, default=10, help="badminton: maximum pose sampling FPS unless --stride is supplied (default: 10)")
    parser.add_argument("--court", type=Path, help="badminton: JSON with four court polygon corners; otherwise save line evidence without excluding players")
    parser.add_argument("--max-highlights", type=int, default=30, help="badminton: maximum selected moments per video")
    parser.add_argument("--highlight-budget", type=float, default=120, help="badminton: maximum total selected video seconds")
    parser.add_argument("--event-threshold", type=float, default=0.5, help="badminton motion score threshold in [0, 1]")
    parser.add_argument("--pre-seconds", type=float, default=3, help="badminton: context before a pose peak")
    parser.add_argument("--post-seconds", type=float, default=3, help="badminton: context after a pose peak")
    parser.add_argument("--detail-backend", choices=["sam3", "yolo", "none"], default="sam3", help="badminton: optional keyframe object detector (default: sam3)")
    parser.add_argument("--object-model", type=Path, help="badminton: local YOLO object checkpoint for --detail-backend yolo")
    parser.add_argument("--detail-keyframes", "--sam-keyframes", type=int, default=3, help="badminton: 1-9 detail keyframes per selected moment (default: 3)")
    parser.add_argument("--max-detail-calls", "--max-sam-calls", type=int, default=90, help="badminton: hard per-video detail inference budget (default: 90)")
    parser.add_argument("--detail-prompts", nargs="+", default=["badminton racket", "shuttlecock"], help="badminton: SAM text prompts, queried together on each crop")


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    finite = [args.pose_fps, args.highlight_budget, args.event_threshold, args.pre_seconds, args.post_seconds]
    if not all(math.isfinite(value) for value in finite):
        parser.error("Badminton rates, scores and durations must be finite")
    if args.pose_fps <= 0 or args.highlight_budget < 0 or args.max_highlights < 0 or args.max_detail_calls < 0:
        parser.error("Pose FPS must be positive; budgets and counts must be nonnegative")
    if not 0 <= args.event_threshold <= 1 or not 1 <= args.detail_keyframes <= 9:
        parser.error("--event-threshold must be in [0, 1]; --detail-keyframes must be in [1, 9]")
    if args.pre_seconds < 0 or args.post_seconds < 0 or args.pre_seconds + args.post_seconds <= 0:
        parser.error("Context durations must be nonnegative with a positive total")
    if args.no_sam and args.detail_backend == "sam3":
        args.detail_backend = "none"
    # A zero call budget is a useful pose-only run, without loading any detail model.
    details_enabled = args.max_detail_calls > 0 and args.max_highlights > 0 and args.highlight_budget > 0
    if args.detail_backend == "sam3" and details_enabled:
        require_weights(args.sam_model)
    if args.detail_backend == "yolo" and details_enabled and (args.object_model is None or not args.object_model.is_file()):
        parser.error("--detail-backend yolo requires a local --object-model checkpoint")


def _crop_box(box: list[float] | np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 16), max(y2 - y1, 16)
    # Include the hand/racket above and beside the body, while preserving more
    # small-object pixels than resizing an entire 1080p gym image to 644 px.
    return (max(0, math.floor(x1 - w * 0.7)), max(0, math.floor(y1 - h * 0.6)),
            min(width, math.ceil(x2 + w * 0.7)), min(height, math.ceil(y2 + h * 0.25)))


def _box_at(poses: dict[str, np.ndarray], moment: Moment, frame: int) -> np.ndarray:
    """Use the same player's nearest sampled pose; never snap to another court/player."""
    if moment.track_id >= 0:
        rows = np.flatnonzero(poses["track_id"] == moment.track_id)
        if len(rows):
            row = rows[np.argmin(np.abs(poses["frame_index"][rows] - frame))]
            return poses["box"][row]
    return np.asarray(moment.window.peak_box)


def analyze_details(video: Path, poses: dict[str, np.ndarray], moments: list[Moment], plan: list[tuple[int, int]],
                    detector, prompts: list[str], out_dir: Path, save_previews: bool) -> list[list[dict]]:
    """Seek only requested frames; one image encoding/call serves all text prompts."""
    details = [[] for _ in moments]
    if not plan:
        return details
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video for keyframes: {video}")
    try:
        for moment_index, frame_index in plan:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise RuntimeError(f"Cannot seek to detail frame {frame_index}")
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode detail frame {frame_index}")
            actual_frame = round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if actual_frame != frame_index:
                raise RuntimeError(f"Requested detail frame {frame_index}, decoder returned {actual_frame}")
            box = _box_at(poses, moments[moment_index], frame_index)
            x1, y1, x2, y2 = _crop_box(box, bgr.shape[1], bgr.shape[0])
            detections = detector.detect(np.ascontiguousarray(bgr[y1:y2, x1:x2]), prompts)
            objects = []
            for detection in detections:
                full_box = detection.box.astype(float) + [x1, y1, x1, y1]
                objects.append({"label": detection.label, "confidence": detection.conf, "box": full_box.tolist()})
                if save_previews:
                    a, b, c, d = np.rint(full_box).astype(int)
                    cv2.rectangle(bgr, (a, b), (c, d), (0, 255, 255), 2)
                    cv2.putText(bgr, f"{detection.label} {detection.conf:.2f}", (a, max(15, b - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            detail = {"frame_index": frame_index, "decoder_time_s": capture.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                      "crop": [x1, y1, x2, y2], "objects": objects}
            if save_previews:
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                name = f"{video.stem}_highlight{moment_index + 1}_frame{frame_index}.jpg"
                if not cv2.imwrite(str(out_dir / name), bgr):
                    raise OSError(f"Could not save detail preview: {name}")
                detail["preview"] = name
            details[moment_index].append(detail)
            print(f"progress details: {detector.calls}/{len(plan)} calls, source frame {frame_index}", flush=True)
    finally:
        capture.release()
    return details


def run(args: argparse.Namespace, cut_clip, poses_match) -> None:
    started = time.perf_counter()
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    court = inspect_court(args.video, args.out_dir, args.court)
    timings = {"device": device, "court_stage_s": time.perf_counter() - started}
    print(f"court: {len(court.lines)} OpenCV line candidates; {court.note}", flush=True)
    stride = args.stride if args.stride is not None else max(1, math.ceil(court.fps / args.pose_fps))
    input_width = args.input_width if args.input_width is not None else 1280
    pose_path = args.out_dir / f"{args.video.stem}_poses.npz"
    reused = not args.redo_pose and poses_match(pose_path, stride, args.model, args.video, input_width, args.conf)
    t0 = time.perf_counter()
    if not reused:
        print(f"stage YOLO pose on {device}: stride {stride}, width {input_width}", flush=True)
        extract_pose.main([str(args.video), "--out-dir", str(args.out_dir), "--stride", str(stride),
                           "--model", args.model, "--device", device, "--input-width", str(input_width),
                           "--conf", str(args.conf), "--no-overlay"])
    poses, meta = load_poses(pose_path)
    timings.update(pose_reused=reused, pose_stage_s=time.perf_counter() - t0)
    duration = meta["frames"] / meta["fps"]
    t0 = time.perf_counter()
    moments, selection = rank_moments(poses, court, duration, args.max_highlights, args.highlight_budget,
                                       args.event_threshold, args.pre_seconds, args.post_seconds)
    plan = keyframe_plan(moments, meta["fps"], meta["frames"], args.detail_keyframes, args.max_detail_calls)
    if args.detail_backend == "none":
        plan = []
    timings["candidate_stage_s"] = time.perf_counter() - t0
    print(f"selected {len(moments)} moment(s), {selection['selected_video_s']:.1f} video seconds; {len(plan)} detail calls planned", flush=True)
    details = [[] for _ in moments]
    timings.update(detail_calls=0, detail_load_s=0.0, detail_analysis_s=0.0, detail_inference_s=0.0)
    if plan:
        t0 = time.perf_counter()
        if args.detail_backend == "sam3":
            detector = Sam3(args.sam_model, device=device)
        else:
            from highlight.objects import YoloObjects
            detector = YoloObjects(args.object_model, device=device)
        t1 = time.perf_counter()
        details = analyze_details(args.video, poses, moments, plan, detector, args.detail_prompts,
                                  args.out_dir, save_previews=not args.no_overlay)
        timings.update(detail_calls=detector.calls, detail_load_s=t1 - t0,
                       detail_analysis_s=time.perf_counter() - t1, detail_inference_s=detector.inference_s)
        if args.detail_backend == "sam3":
            parameter = next(detector.predictor.model.parameters())
            timings.update(sam_calls=detector.calls, sam_inference_s=detector.inference_s,
                           sam_load_s=timings["detail_load_s"], sam_analysis_s=timings["detail_analysis_s"],
                           sam_actual_device=str(parameter.device), sam_dtype=str(parameter.dtype))
    windows = []
    for i, (moment, evidence) in enumerate(zip(moments, details)):
        window = asdict(moment.window)
        window.update(label="candidate", reason=moment.reason, peak_frame=moment.peak_frame, track_id=moment.track_id,
                      pose_features={"overhead": moment.overhead, "wrist_speed": moment.wrist_speed,
                                     "body_speed": moment.body_speed}, details=evidence)
        if not args.no_clips:
            name = f"{args.video.stem}_highlight{i + 1}_candidate.mp4"
            cut_clip(args.video, moment.window.start_s, moment.window.end_s, args.out_dir / name)
            window["clip"] = name
        windows.append(window)
        print(f"  {i + 1}. {moment.window.start_s:.1f}-{moment.window.end_s:.1f} s: {moment.reason}, score {moment.window.peak_score:.2f}")
    timings["total_before_json_s"] = time.perf_counter() - started
    result = {
        "video": meta["video"], "sport": "badminton", "pipeline": "sparse_pose_keyframe_details",
        "duration_s": duration, "timing": timings, "court": asdict(court),
        "settings": {"model": args.model, "input_width": input_width, "pose_stride": stride,
                     "pose_target_fps": args.pose_fps if args.stride is None else None,
                     "pose_actual_fps": court.fps / stride, "detail_backend": args.detail_backend,
                     "sam_model": str(args.sam_model) if args.detail_backend == "sam3" else None,
                     "object_model": str(args.object_model) if args.object_model is not None else None,
                     "detail_prompts": args.detail_prompts if args.detail_backend == "sam3" else [],
                     "detail_keyframes": args.detail_keyframes, "max_detail_calls": args.max_detail_calls,
                     "max_highlights": args.max_highlights, "highlight_budget_s": args.highlight_budget,
                     "event_threshold": args.event_threshold, "pre_s": args.pre_seconds, "post_s": args.post_seconds},
        "selection": selection, "detail_plan": [{"moment_index": i, "frame_index": frame} for i, frame in plan],
        "limitations": ["Pose motion candidates are not validated smash, winner, hit, or complete-rally labels.",
                        "A missed racket/shuttle detection does not discard a pose candidate.",
                        "No automatic target-court identification; inspect the court preview and supply --court if needed.",
                        "Tiny shuttle detection and the lower pose resolution require accuracy validation."],
        "windows": windows,
    }
    path = args.out_dir / f"{args.video.stem}_highlights.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"saved {path}; total {timings['total_before_json_s']:.2f} s; detail calls {timings['detail_calls']}", flush=True)
