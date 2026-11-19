"""Measure one multi-rate fusion run; pass other flags through to highlight-fusion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--monitor-mac", action="store_true")
    args, extra = parser.parse_known_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / "benchmark.json"
    if destination.exists() or (args.out_dir / "manifest.json").exists():
        parser.error("Use a fresh output directory")
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "title": "Dense OpenCV + cached poses + YOLO objects + sparse SAM",
              "pipeline": "fusion", "expected_runs": 1, "runs": [], "status": "running",
              "scope": "Full-video CV and overlays; reused poses/SAM are separate from new bounded object/SAM calls."}

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(destination)

    save()
    monitor = monitor_log = None
    if args.monitor_mac:
        monitor_log = (args.out_dir / "mac_monitor.log").open("w")
        monitor = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("monitor_mac.py")), str(args.out_dir),
                                    "--max-seconds", "3600"], stdout=monitor_log, stderr=subprocess.STDOUT)
    command = [sys.executable, "-u", "-m", "highlight.fusion", str(args.video), "--out-dir", str(args.out_dir), *extra]
    started = time.perf_counter()
    peak = 0
    with (args.out_dir / "fusion.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                peak = max(peak, process.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.1)
    row = {"video": str(args.video), "exit_code": child.returncode, "wall_s": time.perf_counter()-started,
           "peak_parent_rss_gib": peak / 2**30, "command": command}
    if child.returncode == 0:
        manifest = json.loads((args.out_dir / "manifest.json").read_text())
        row.update(pipeline_timing=manifest["timing"], coverage=manifest["coverage"])
    else:
        row["error_tail"] = (args.out_dir / "fusion.log").read_text(errors="replace").splitlines()[-30:]
    report["runs"].append(row)
    save()
    if monitor:
        try:
            monitor.wait(timeout=30)
        except subprocess.TimeoutExpired:
            monitor.terminate()
            monitor.wait(timeout=10)
        report["monitor_exit_code"] = monitor.returncode
        monitor_log.close()
    failed = child.returncode != 0 or (monitor is not None and monitor.returncode != 0)
    report["status"] = "failed" if failed else "complete"
    save()
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
