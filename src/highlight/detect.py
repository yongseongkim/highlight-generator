"""Find highlight moments in a video and cut them into clips.

    highlight-detect path/to/video.mp4 --sport basketball

Stage 1 (YOLO pose, whole video): player poses flag candidate windows.
Stage 2 (SAM 3, candidate windows only; basketball): tracks the ball around the hoop and
labels each window made / miss / no shot. Windows without a shot get no clip.

Runs pose extraction first unless <out-dir>/<video>_poses.npz already matches the
settings. Writes <out-dir>/<video>_highlights.json, the clips and, when SAM 3 ran, an
overlay video per window showing what both models saw.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import imageio_ffmpeg

from highlight import extract_pose
from highlight.candidates import SPORTS, find_windows
from highlight.pose import load_poses, poses_by_frame
from highlight.sam import Sam3
from highlight.shots import analyze_shots
from highlight.signals import cue_series
from highlight.viz import render_shot_videos


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--sport", required=True, choices=sorted(SPORTS))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stride", type=int, default=3, help="pose extraction stride (default: 3)")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="YOLO pose weights (default: yolo11s-pose.pt)")
    parser.add_argument("--redo-pose", action="store_true", help="re-run pose extraction even if saved poses match")
    parser.add_argument("--no-sam", action="store_true", help="skip the SAM 3 stage and clip every candidate window")
    parser.add_argument("--sam-stride", type=int, default=2, help="SAM 3 runs on every Nth frame of a window (default: 2)")
    parser.add_argument("--no-clips", action="store_true", help="only write the JSON, don't cut clips")
    parser.add_argument("--no-overlay", action="store_true", help="skip the per-window pose + SAM 3 overlay videos")
    args = parser.parse_args(argv)

    poses_path = args.out_dir / f"{args.video.stem}_poses.npz"
    if args.redo_pose or not _poses_match(poses_path, args.stride, Path(args.model).name):
        extract_pose.main(
            [str(args.video), "--out-dir", str(args.out_dir), "--stride", str(args.stride), "--model", args.model, "--no-overlay"]
        )
    poses, meta = load_poses(poses_path)

    sport = SPORTS[args.sport]
    series = cue_series(poses, sport.cue)
    duration_s = meta["frames"] / meta["fps"]
    windows = find_windows(series, sport.threshold, duration_s)
    print(f"{args.video.name}: {len(windows)} candidate window(s) where {sport.cue} >= {sport.threshold}")

    # Stage 2: only basketball has a SAM 3 check so far.
    shots = None
    if args.sport == "basketball" and not args.no_sam and windows:
        t0 = time.perf_counter()
        sam = Sam3(extract_pose.MODELS_DIR / "sam3.pt")
        t1 = time.perf_counter()
        shots, hoop = analyze_shots(str(args.video), windows, sam, stride=args.sam_stride)
        t2 = time.perf_counter()
        seconds = sum(w.end_s - w.start_s for w in windows)
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
        render_shot_videos(str(args.video), windows, shots, hoop, poses_by_frame(poses), args.sam_stride, overlays)
        print("overlays (YOLO-pose skeletons + SAM 3 hoop, ball path, verdict):")
        for path in overlays:
            print(f"  {path}")

    json_path = args.out_dir / f"{args.video.stem}_highlights.json"
    json_path.write_text(
        json.dumps(
            {
                "video": meta["video"],
                "sport": args.sport,
                "cue": sport.cue,
                "threshold": sport.threshold,
                "duration_s": duration_s,
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


def _poses_match(path: Path, stride: int, model: str) -> bool:
    if not path.exists():
        return False
    _, meta = load_poses(path)
    return meta["stride"] == stride and meta["model"] == model


if __name__ == "__main__":
    main()
