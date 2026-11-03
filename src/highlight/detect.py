"""Find highlight moments in a video and cut them into clips.

    highlight-detect path/to/video.mp4 --sport basketball
    highlight-detect path/to/video.mp4 --sport badminton --device mps

Stage 1 (YOLO pose, whole video): player poses flag candidate windows.
Stage 2 (SAM 3, candidate windows only; basketball): tracks the ball around the hoop and
labels each window made / miss / no shot. Windows without a shot get no clip.
Badminton uses sparse pose sampling, ranked motion peaks and a bounded number of
detail keyframes, with optional court calibration and YOLO object detection.

Runs pose extraction first unless <out-dir>/<video>_poses.npz already matches the
settings. Writes <out-dir>/<video>_highlights.json, the clips and, when SAM 3 ran, an
overlay video per window showing what both models saw.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import imageio_ffmpeg

from highlight import extract_pose
from highlight.candidates import SPORTS, find_windows
from highlight.pose import file_identity, load_poses, poses_by_frame
from highlight.sam import Sam3, require_weights
from highlight.shots import analyze_shots
from highlight.signals import cue_series
from highlight.viz import render_shot_videos
from highlight.video import resolve_device
from highlight.badminton import add_arguments, run as run_badminton, validate_args as validate_badminton


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--sport", required=True, choices=sorted([*SPORTS, "badminton"]))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stride", type=int, help="pose extraction stride (default: 3; badminton uses --pose-fps)")
    parser.add_argument("--input-width", type=int, help="pose model input width, multiple of 32 (default: 1920; badminton: 1280)")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum person confidence (default: 0.25)")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="YOLO pose weights (default: yolo11s-pose.pt)")
    parser.add_argument("--device", help="inference GPU, e.g. mps or cuda:0 (default: auto)")
    parser.add_argument("--sam-model", type=Path, default=extract_pose.MODELS_DIR / "sam3.pt")
    parser.add_argument("--redo-pose", action="store_true", help="re-run pose extraction even if saved poses match")
    parser.add_argument("--no-sam", action="store_true", help="skip the SAM 3 stage and clip every candidate window")
    parser.add_argument("--sam-stride", type=int, default=2, help="basketball: SAM 3 runs every Nth window frame (default: 2)")
    parser.add_argument("--no-clips", action="store_true", help="skip cutting highlight clips")
    parser.add_argument("--no-overlay", action="store_true", help="skip basketball overlay videos / badminton detail JPEGs")
    add_arguments(parser)
    args = parser.parse_args(argv)
    if (args.stride is not None and args.stride < 1) or args.sam_stride < 1:
        parser.error("--stride and --sam-stride must be positive")
    if args.input_width is not None and (args.input_width < 32 or args.input_width % 32):
        parser.error("--input-width must be a positive multiple of 32")
    if not math.isfinite(args.conf) or not 0 < args.conf <= 1:
        parser.error("--conf must be in (0, 1]")
    if args.sport == "badminton":
        validate_badminton(args, parser)
        run_badminton(args, cut_clip, _poses_match)
        return
    args.stride = args.stride if args.stride is not None else 3
    args.input_width = args.input_width if args.input_width is not None else 1920
    device = resolve_device(args.device)
    if args.sport == "basketball" and not args.no_sam:
        # Fail before a long pose pass when the required SAM checkpoint is absent.
        require_weights(args.sam_model)
    started = time.perf_counter()
    timings = {"device": device}

    poses_path = args.out_dir / f"{args.video.stem}_poses.npz"
    reused_pose = not args.redo_pose and _poses_match(poses_path, args.stride, args.model, args.video, args.input_width, args.conf)
    if not reused_pose:
        print(f"stage YOLO pose on {device}", flush=True)
        extract_pose.main(
            [str(args.video), "--out-dir", str(args.out_dir), "--stride", str(args.stride), "--model", args.model,
             "--input-width", str(args.input_width), "--conf", str(args.conf), "--device", device, "--no-overlay"]
        )
    poses, meta = load_poses(poses_path)
    timings.update(pose_reused=reused_pose, pose_stage_s=time.perf_counter() - started)

    sport = SPORTS[args.sport]
    series = cue_series(poses, sport.cue)
    duration_s = meta["frames"] / meta["fps"]
    windows = find_windows(series, sport.threshold, duration_s)
    timings["candidate_stage_s"] = time.perf_counter() - started - timings["pose_stage_s"]
    print(f"{args.video.name}: {len(windows)} candidate window(s) where {sport.cue} >= {sport.threshold}")

    # Stage 2: only basketball has a SAM 3 check so far.
    shots = None
    if args.sport == "basketball" and not args.no_sam and windows:
        print(f"stage SAM 3 on {device}: loading {args.sam_model}", flush=True)
        t0 = time.perf_counter()
        sam = Sam3(args.sam_model, device=device)
        t1 = time.perf_counter()
        shots, hoop = analyze_shots(str(args.video), windows, sam, stride=args.sam_stride)
        t2 = time.perf_counter()
        seconds = sum(w.end_s - w.start_s for w in windows)
        parameter = next(sam.predictor.model.parameters())
        timings.update(
            sam_load_s=t1 - t0, sam_analysis_s=t2 - t1, sam_candidate_video_s=seconds,
            sam_calls=sam.calls, sam_inference_s=sam.inference_s,
            sam_actual_device=str(parameter.device), sam_dtype=str(parameter.dtype),
        )
        print(f"SAM 3: load {t1 - t0:.1f} s, {seconds:.1f} s of video checked in {t2 - t1:.1f} s (every {args.sam_stride} frames)")
        if hoop is None:
            print("  no hoop found; every window is labeled no shot")

    labels = [shot.label for shot in shots] if shots is not None else [""] * len(windows)
    for i, (w, label) in enumerate(zip(windows, labels)):
        line = f"  {i + 1}. {w.start_s:5.1f} s - {w.end_s:5.1f} s   pose peak {w.peak_score:.2f} at {w.peak_s:.1f} s"
        if label == "made":
            line += f"   MADE at {shots[i].made_at_s:.1f} s"
        elif label == "miss":
            line += "   shot, missed"
        elif label == "noshot":
            line += "   no shot at the rim, no clip"
        if not args.no_clips and label != "noshot":
            clip = args.out_dir / f"{_window_name(args.video, i, label)}{args.video.suffix.lower()}"
            cut_clip(args.video, w.start_s, w.end_s, clip)
            line += f"   -> {clip}"
        print(line)

    if shots is not None and not args.no_overlay:
        overlays = [args.out_dir / f"{_window_name(args.video, i, label)}_overlay.mp4" for i, label in enumerate(labels)]
        render_shot_videos(str(args.video), windows, shots, hoop, poses_by_frame(poses), args.sam_stride, overlays, device=device)
        print("overlays (YOLO-pose skeletons + SAM 3 hoop, ball path, verdict):")
        for path in overlays:
            print(f"  {path}")

    json_path = args.out_dir / f"{args.video.stem}_highlights.json"
    timings["total_before_json_s"] = time.perf_counter() - started
    json_path.write_text(
        json.dumps(
            {
                "video": meta["video"],
                "sport": args.sport,
                "cue": sport.cue,
                "threshold": sport.threshold,
                "duration_s": duration_s,
                "timing": timings,
                "settings": {"pose_stride": args.stride, "input_width": args.input_width, "conf": args.conf,
                             "sam_stride": args.sam_stride, "sam_model": str(args.sam_model)},
                "windows": [
                    {**asdict(w), **({"shot": {"label": shots[i].label, **asdict(shots[i])}} if shots is not None else {})}
                    for i, w in enumerate(windows)
                ],
            },
            indent=2,
        )
    )
    print(f"saved {json_path}")


def cut_clip(video: Path, start_s: float, end_s: float, out: Path) -> None:
    """Copy [start_s, end_s] of `video` without re-encoding.

    Stream copy can only start on a keyframe, so a clip may begin up to one keyframe
    interval before `start_s`. Only the first video track and any audio are kept;
    iPhone metadata tracks don't survive the copy.
    """
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-i", str(video), "-t", f"{end_s - start_s:.3f}",
            "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(out),
        ],
        check=True,
    )  # fmt: skip


def _window_name(video: Path, i: int, label: str) -> str:
    return f"{video.stem}_highlight{i + 1}" + (f"_{label}" if label else "")


def _poses_match(path: Path, stride: int, model: str, video: Path, input_width: int, conf: float) -> bool:
    if not path.exists():
        return False
    try:
        _, meta = load_poses(path)
        return (meta.get("stride") == stride and meta.get("input_width") == input_width
                and meta.get("conf") == conf and meta.get("source_identity") == file_identity(video)
                and meta.get("model_identity") == file_identity(extract_pose.model_path(model)))
    except (OSError, ValueError, KeyError):
        return False


if __name__ == "__main__":
    main()
