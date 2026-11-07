"""Measure extraction from existing pose/SAM archives without re-running YOLO."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--court-root", type=Path, help="optional per-video <id>.json court/foreground ROI proposals")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--refine-sam", action="store_true")
    parser.add_argument("--motion-seeds", action="store_true", help="benchmark peak expansion via highlight-plays instead of action clips")
    parser.add_argument("--reel", action="store_true")
    parser.add_argument("--monitor-mac", action="store_true")
    parser.add_argument("--max-highlights", type=int, default=5)
    parser.add_argument("--highlight-budget", type=float, default=40)
    parser.add_argument("--max-sam-calls", type=int, default=15)
    args = parser.parse_args()
    if args.motion_seeds and args.refine_sam:
        parser.error("Motion benchmarks reuse cached SAM evidence; omit --refine-sam")
    if args.motion_seeds and args.court_root is None:
        parser.error("--motion-seeds requires --court-root")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / "benchmark.json"
    if destination.exists():
        parser.error("Use a fresh output directory")
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "expected_runs": len(args.videos),
              "title": "Cached poses + motion seed context" if args.motion_seeds else "Cached YOLO pose + temporal patterns + SAM3",
              "pipeline": "plays" if args.motion_seeds else "patterns", "status": "running",
              "scope": ("Reuses pose archives and saved SAM evidence for motion-seed context proposals. Includes clips and optional H.264 reels. No new model inference."
                        if args.motion_seeds else "Reuses high-resolution pose archives and saved SAM evidence. Optional fresh SAM refinement; includes clip, JPEG, and optional H.264 reel export. No YOLO inference."),
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "videos"},
              "runs": []}

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(destination)

    save()
    monitor = monitor_log = None
    if args.monitor_mac:
        monitor_log = (args.out_dir / "mac_monitor.log").open("w")
        monitor = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("monitor_mac.py")), str(args.out_dir),
                                    "--max-seconds", "3600"], stdout=monitor_log, stderr=subprocess.STDOUT)
    failed = False
    for video in args.videos:
        name = video.stem
        output = args.out_dir / name
        evidence = args.sam_root / name / f"{name}_patterns.json"
        if not evidence.exists():
            evidence = args.sam_root / name / f"{name}_highlights.json"
        command = [sys.executable, "-u", "-m", "highlight.patterns", str(video),
                   "--poses", str(args.pose_root / name / f"{name}_poses.npz"),
                   "--sam-evidence", str(evidence),
                   "--out-dir", str(output), "--device", args.device, "--max-highlights", str(args.max_highlights),
                   "--highlight-budget", str(args.highlight_budget), "--max-sam-calls", str(args.max_sam_calls)]
        if args.motion_seeds:
            command = [sys.executable, "-u", "-m", "highlight.plays", str(video), "--sport", "badminton", "--motion-seeds",
                       "--poses", str(args.pose_root / name / f"{name}_poses.npz"), "--sam-evidence", str(evidence),
                       "--out-dir", str(output), "--max-highlights", str(args.max_highlights),
                       "--highlight-budget", str(args.highlight_budget)]
        if args.refine_sam:
            command.append("--refine-sam")
        if args.reel:
            command.append("--reel")
        if args.court_root:
            command += ["--court", str(args.court_root / f"{name}.json")]
        environment = dict(os.environ)
        environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        log_path = args.out_dir / f"{name}.log"
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        peak_rss = 0
        with log_path.open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            process = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.1)
        run = {"video": str(video), "exit_code": child.returncode, "wall_s": time.perf_counter() - started,
               "peak_process_rss_gib": peak_rss / 2**30, "command": command, "log": str(log_path)}
        if child.returncode == 0:
            if args.motion_seeds:
                result = json.loads((output / f"{name}_plays.json").read_text())
                run.update(diagnostics=result["diagnostics"], selected_windows=len(result["selected_units"]),
                           selected_s=sum(u["end_s"] - u["start_s"] for u in result["selected_units"]),
                           verified_boundaries=sum(u["boundary_status"] == "verified" for u in result["selected_units"]))
            else:
                result = json.loads((output / f"{name}_patterns.json").read_text())
                run.update(pattern_counts=result["pattern_counts"], selected_windows=len(result["windows"]), selected_s=result["selected_s"])
            run.update(duration_s=result["duration_s"], pipeline_timing=result["timing"], reel=result["reel"])
        else:
            run["error_tail"] = log_path.read_text(errors="replace").splitlines()[-20:]
            failed = True
        report["runs"].append(run)
        save()
        print(f"DONE {name}: exit={child.returncode}, {run['wall_s']:.2f}s, peak RSS {peak_rss / 2**30:.2f} GiB", flush=True)
        if failed:
            print("\n".join(run["error_tail"]), flush=True)
            break
    if monitor:
        try:
            monitor.wait(timeout=30)
        except subprocess.TimeoutExpired:
            monitor.terminate()
            monitor.wait(timeout=10)
        monitor_log.close()
        report["monitor_exit_code"] = monitor.returncode
        failed = failed or monitor.returncode != 0
    report["status"] = "failed" if failed else "complete"
    save()
    print(destination, flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
