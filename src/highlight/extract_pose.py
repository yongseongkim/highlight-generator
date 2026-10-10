"""Detect people and their 2D pose in a video, entirely on the GPU.

    highlight-pose path/to/video.mp4 --stride 3

Writes <out-dir>/<video>_poses.npz (see pose.save_poses) and, unless --no-overlay,
<out-dir>/<video>_pose.mp4 with skeletons drawn on the sampled frames.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from highlight.pose import PoseDetector, save_poses
from highlight.video import NvdecReader
from highlight.viz import draw_poses

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stride", type=int, default=3, help="run pose on every Nth frame (default: 3, ~10 fps for 30 fps video)")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="YOLO pose weights; bare names are downloaded into models/")
    parser.add_argument("--input-width", type=int, default=1920, help="model input width, multiple of 32 (default: 1920; smaller misses distant players)")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum person confidence")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-overlay", action="store_true", help="skip writing the skeleton overlay video")
    args = parser.parse_args(argv)

    model = Path(args.model)
    if model.parent == Path("."):
        model = MODELS_DIR / model
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"{args.video.stem}_poses.npz"
    overlay_path = args.out_dir / f"{args.video.stem}_pose.mp4"

    t_start = time.perf_counter()
    reader = NvdecReader(str(args.video), gpu_id=args.gpu)
    detector = PoseDetector(model, input_width=args.input_width, device=f"cuda:{args.gpu}", conf=args.conf)
    info = reader.info
    writer = None
    if not args.no_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, info.fps / args.stride, (info.width, info.height))
    t_loaded = time.perf_counter()

    # The first frame pays for CUDA/model warmup, so per-frame timings skip it.
    sampled = []
    t_decode = t_pose = t_overlay = 0.0
    frames = reader.frames(args.stride)
    while True:
        t0 = time.perf_counter()
        frame = next(frames, None)
        t1 = time.perf_counter()
        if frame is None:
            break
        poses = detector(frame.rgb)
        t2 = time.perf_counter()
        if writer is not None:
            bgr = frame.rgb.cpu().numpy()[..., ::-1].copy()
            writer.write(draw_poses(bgr, poses))
        t3 = time.perf_counter()
        sampled.append((frame.index, frame.time_s, poses))
        if len(sampled) > 1:
            t_decode += t1 - t0
            t_pose += t2 - t1
            t_overlay += t3 - t2
        else:
            t_first = t3 - t0
    if writer is not None:
        writer.release()
    t_end = time.perf_counter()

    save_poses(
        npz_path,
        sampled,
        meta={
            "video": str(args.video.resolve()),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frames": reader.frames_decoded,
            "stride": args.stride,
            "model": model.name,
            "input_width": args.input_width,
            "conf": args.conf,
        },
    )

    n = len(sampled)
    if n == 0:
        print(f"No frames decoded from {args.video}")
        return
    people = [len(p.track_ids) for _, _, p in sampled]
    track_ids = {int(i) for _, _, p in sampled for i in p.track_ids if i >= 0}
    untracked = sum(int((p.track_ids < 0).sum()) for _, _, p in sampled)
    duration = reader.frames_decoded / info.fps
    steady = max(n - 1, 1)
    print(f"video    {info.width}x{info.height} @ {info.fps:.2f} fps, {reader.frames_decoded} frames ({duration:.1f} s)")
    print(
        f"pose     {n} frames (every {args.stride}), {sum(people) / n:.1f} people/frame (min {min(people)}, max {max(people)}),"
        f" {len(track_ids)} track IDs, {untracked}/{sum(people)} detections without an ID"
    )
    print(
        f"per sampled frame  decode {t_decode / steady * 1000:.1f} ms (incl. {args.stride - 1} skipped)"
        f" | pose {t_pose / steady * 1000:.1f} ms | overlay {t_overlay / steady * 1000:.1f} ms"
    )
    steady_speed = duration * (n - 1) / n / max(t_decode + t_pose + t_overlay, 1e-9)
    print(
        f"total    {t_end - t_start:.2f} s = load {t_loaded - t_start:.2f} s + first frame {t_first:.2f} s"
        f" + rest {t_end - t_loaded - t_first:.2f} s ({steady_speed:.1f}x realtime after warmup)"
    )
    print(f"saved    {npz_path}" + ("" if writer is None else f"\n         {overlay_path}"))


if __name__ == "__main__":
    main()
