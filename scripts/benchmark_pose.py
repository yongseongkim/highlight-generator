"""Benchmark fresh pose, basketball SAM, or sparse badminton extraction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import psutil
import torch
import ultralytics

from highlight.pose import load_poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pipeline", choices=["pose", "combined", "badminton"], default="pose")
    parser.add_argument("--sport", choices=["basketball", "football", "badminton"], default="basketball")
    parser.add_argument("--sam-model", type=Path, default=Path("models/sam3.pt"))
    parser.add_argument("--monitor-mac", action="store_true", help="record whole-Mac resources for the full run")
    parser.add_argument("--court", type=Path, help="badminton: calibration JSON for these videos")
    parser.add_argument("--detail-backend", choices=["sam3", "yolo", "none"], default="sam3")
    parser.add_argument("--object-model", type=Path)
    parser.add_argument("--max-detail-calls", type=int, default=90)
    args = parser.parse_args()
    if args.pipeline == "combined" and args.sport != "basketball":
        parser.error("The combined SAM 3 pipeline currently only supports basketball")
    if args.pipeline == "badminton":
        args.sport = "badminton"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "benchmark.json"
    if result_path.exists():
        parser.error(f"Use a fresh output directory; {result_path} already exists")
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pr_head": "057ee4f2a34304433177e14ac17287c2470c8b31",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "device": args.device,
        "pipeline": args.pipeline,
        "title": ("Sparse YOLO-pose + badminton keyframe details" if args.pipeline == "badminton" else
                  "YOLO11s-pose + SAM 3" if args.pipeline == "combined" else "YOLO11s-pose"),
        "expected_runs": len(args.videos),
        "settings": {"model": "yolo11s-pose.pt", "input_width": 1920, "stride": 3, "quantize": 16, "overlay": False},
        "scope": ("Fresh sparse pose + bounded keyframe details. Pose-based badminton candidates; selection accuracy is not validated."
                  if args.pipeline == "badminton" else
                  "Fresh YOLO pose + basketball SAM 3 stage. Running this on badminton tests performance, not badminton highlight accuracy."
                  if args.pipeline == "combined" else "Full-video pose stage only. CPU decode/upload, GPU model, CPU tracking. No cached poses."),
        "runs": [],
    }
    if args.pipeline == "combined":
        report["settings"].update(sport=args.sport, sam_model=str(args.sam_model), sam_stride=2, clips=False)
    elif args.pipeline == "badminton":
        report["settings"].update(sport="badminton", input_width=1280, stride=None, pose_target_fps=10,
                                   detail_backend=args.detail_backend, detail_keyframes=3,
                                   max_detail_calls=args.max_detail_calls, max_highlights=30, highlight_budget_s=120,
                                   sam_model=str(args.sam_model), object_model=str(args.object_model) if args.object_model else None,
                                   court=str(args.court) if args.court else None, clips=False)

    def save_report():
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(result_path)

    save_report()
    monitor = monitor_log = None
    if args.monitor_mac:
        monitor_log = (args.out_dir / "mac_monitor.log").open("w")
        monitor = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name("monitor_mac.py")), str(args.out_dir), "--max-seconds", "43200"],
            stdout=monitor_log, stderr=subprocess.STDOUT,
        )
    failed = False
    for video in args.videos:
        output = args.out_dir / video.stem
        if (output / f"{video.stem}_poses.npz").exists():
            parser.error(f"Cached pose data exists: {output}")
        command = [sys.executable, "-u", "-m", "highlight.extract_pose", str(video),
                   "--device", args.device, "--out-dir", str(output), "--no-overlay"]
        if args.pipeline in {"combined", "badminton"}:
            command = [sys.executable, "-u", "-m", "highlight.detect", str(video),
                       "--device", args.device, "--out-dir", str(output), "--sport", args.sport,
                       "--sam-model", str(args.sam_model), "--redo-pose", "--no-overlay", "--no-clips"]
        if args.pipeline == "badminton":
            command += ["--detail-backend", args.detail_backend, "--max-detail-calls", str(args.max_detail_calls)]
            if args.court is not None:
                command += ["--court", str(args.court)]
            if args.object_model is not None:
                command += ["--object-model", str(args.object_model)]
        print(f"START {video.name}", flush=True)
        log_path = args.out_dir / f"{video.stem}.log"
        environment = dict(os.environ)
        environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        peak_rss = cpu_seconds = 0.0
        samples = 0
        started = time.perf_counter()
        last_update = started
        with log_path.open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            process = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    cpu = process.cpu_times()
                    cpu_seconds = cpu.user + cpu.system
                    samples += 1
                except psutil.NoSuchProcess:
                    pass
                if time.perf_counter() - last_update >= 30:
                    lines = log_path.read_text(errors="replace").splitlines()
                    progress = next((line for line in reversed(lines) if line.startswith("progress ")), "initializing")
                    print(f"{video.stem}: {progress}; peak RSS {peak_rss / 2**30:.2f} GiB", flush=True)
                    last_update = time.perf_counter()
                time.sleep(0.25)
        wall = time.perf_counter() - started
        run = {"video": str(video), "command": command, "exit_code": child.returncode,
               "wall_s": wall, "peak_rss_gib": peak_rss / 2**30,
               "cpu_seconds_sampled": cpu_seconds, "resource_samples": samples, "log": str(log_path)}
        if child.returncode == 0:
            arrays, meta = load_poses(output / f"{video.stem}_poses.npz")
            duration = meta["frames"] / meta["fps"]
            run.update({"meta": meta, "duration_s": duration,
                        "sampled_frames": len(arrays["sampled_frame_index"]),
                        "detections": len(arrays["track_id"]),
                        "unique_tracks": len(np.unique(arrays["track_id"][arrays["track_id"] >= 0])),
                        "wall_realtime_multiple": duration / wall,
                        "sampled_fps": len(arrays["sampled_frame_index"]) / wall})
            if args.pipeline in {"combined", "badminton"}:
                highlights = json.loads((output / f"{video.stem}_highlights.json").read_text())
                run["pipeline_timing"] = highlights["timing"]
                run["candidate_windows"] = len(highlights["windows"])
                run["sam_ran"] = highlights["timing"].get("sam_calls", 0) > 0
                run["pipeline_settings"] = highlights["settings"]
        else:
            run["error_tail"] = log_path.read_text(errors="replace").splitlines()[-30:]
            failed = True
        report["runs"].append(run)
        save_report()
        print(f"DONE {video.name}: exit={child.returncode}, wall={wall:.2f}s, peak RSS={peak_rss / 2**30:.2f}GiB", flush=True)
        if child.returncode:
            print("\n".join(run["error_tail"]), flush=True)
            break
    if monitor is not None:
        try:
            monitor.wait(timeout=30)
        except subprocess.TimeoutExpired:
            monitor.terminate()
            monitor.wait(timeout=10)
        monitor_log.close()
        if monitor.returncode:
            print(f"Mac monitor exited with {monitor.returncode}; see {args.out_dir / 'mac_monitor.log'}", flush=True)
            failed = True
        report["monitor_exit_code"] = monitor.returncode
    report["status"] = "failed" if failed else "complete"
    save_report()
    print(f"Results: {result_path}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
