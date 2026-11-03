"""Sample Mac CPU, RAM, swap, GPU counters, and thermal pressure during a benchmark."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import psutil


def thermal_reader():
    """Use NSProcessInfo's public thermalState API; no root-only sensor access."""
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    msg_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    msg_int = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    info = msg_ptr(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
    selector = objc.sel_registerName(b"thermalState")
    labels = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
    return lambda: labels.get(msg_int(info, selector), "unknown")


def gpu_stats() -> dict:
    output = subprocess.check_output(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1", "-l"], text=True, timeout=5)
    line = next((line for line in output.splitlines() if '"PerformanceStatistics" =' in line), "")
    values = {key: int(value) for key, value in re.findall(r'"([^"]+)"=(\d+)', line)}
    return {
        "gpu_util_percent": values.get("Device Utilization %"),
        "gpu_memory_in_use_gib": values.get("In use system memory", 0) / 2**30,
        "gpu_memory_allocated_gib": values.get("Alloc system memory", 0) / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    summary_path = args.benchmark_dir / "mac_monitor_summary.json"
    raw_path = args.benchmark_dir / "mac_metrics.jsonl"
    if raw_path.exists():
        parser.error(f"Monitor log already exists: {raw_path}")
    report_at_start = json.loads((args.benchmark_dir / "benchmark.json").read_text())
    thermal = thermal_reader()
    psutil.cpu_percent()
    active_processes = {}
    samples = []
    started = time.monotonic()
    last_print = started
    completed_at = None
    swap_start = psutil.swap_memory()

    def summarize():
        groups = {}
        for name in sorted({s["active_video"] or "idle" for s in samples}):
            rows = [s for s in samples if (s["active_video"] or "idle") == name]
            metrics = {}
            for key in ("cpu_percent", "process_cpu_percent", "process_rss_gib", "memory_used_gib",
                        "memory_available_gib", "swap_used_gib", "gpu_util_percent", "gpu_memory_in_use_gib"):
                values = [r[key] for r in rows if r.get(key) is not None]
                if values:
                    metrics[key] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
            groups[name] = {"samples": len(rows), "metrics": metrics,
                            "thermal_states": sorted({r["thermal_state"] for r in rows})}
        return {"started_at": samples[0]["time"] if samples else None, "interval_s": args.interval,
                "completed_runs_at_monitor_start": len(report_at_start["runs"]),
                "active_video_at_monitor_start": samples[0]["active_video"] if samples else None,
                "coverage": "From monitor startup through benchmark completion plus 15 seconds of idle time.",
                "notes": ["CPU percent is normalized across all logical CPUs; process CPU uses 100% per logical CPU.",
                          "GPU and memory counters are whole-system values, including other apps; GPU memory is shared RAM.",
                          "Thermal pressure is NSProcessInfo.thermalState, not a temperature in Celsius.",
                          "GPU utilization and memory are sampled IOAccelerator driver counters."],
                "samples": len(samples), "groups": groups,
                "swap_in_bytes_during_monitoring": samples[-1]["swap_in_bytes"] - swap_start.sin if samples else 0,
                "swap_out_bytes_during_monitoring": samples[-1]["swap_out_bytes"] - swap_start.sout if samples else 0}

    with raw_path.open("w") as raw:
        while time.monotonic() - started < args.max_seconds:
            cycle = time.monotonic()
            active = None
            process_cpu = process_rss = None
            for process in psutil.process_iter(["pid", "cmdline"]):
                try:
                    command = process.info["cmdline"] or []
                    module = next((name for name in ("highlight.extract_pose", "highlight.detect", "highlight.patterns") if name in command), None)
                    if module is None:
                        continue
                    pos = command.index(module)
                    active = Path(command[pos + 1]).stem
                    process = active_processes.setdefault(process.pid, process)
                    process_cpu = process.cpu_percent()
                    process_rss = process.memory_info().rss / 2**30
                    break
                except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
                    continue
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": time.monotonic() - started,
                   "active_video": active, "cpu_percent": psutil.cpu_percent(),
                   "cpu_per_core_percent": psutil.cpu_percent(percpu=True),
                   "process_cpu_percent": process_cpu, "process_rss_gib": process_rss,
                   "memory_used_gib": memory.used / 2**30, "memory_available_gib": memory.available / 2**30,
                   "memory_percent": memory.percent, "swap_used_gib": swap.used / 2**30,
                   "swap_in_bytes": swap.sin, "swap_out_bytes": swap.sout, "thermal_state": thermal()}
            try:
                row.update(gpu_stats())
            except (subprocess.SubprocessError, OSError) as error:
                row["gpu_error"] = str(error)
            samples.append(row)
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            summary_path.write_text(json.dumps(summarize(), indent=2))
            if len(samples) == 1 or time.monotonic() - last_print >= 30:
                print(f"{active or 'idle'}: CPU {row['cpu_percent']:.1f}%, GPU {row.get('gpu_util_percent')}%, "
                      f"RAM {row['memory_used_gib']:.1f} GiB, swap {row['swap_used_gib']:.2f} GiB, "
                      f"thermal {row['thermal_state']}", flush=True)
                last_print = time.monotonic()
            report = json.loads((args.benchmark_dir / "benchmark.json").read_text())
            if len(report["runs"]) == report.get("expected_runs", 3) or any(run["exit_code"] for run in report["runs"]):
                completed_at = completed_at or time.monotonic()
                if time.monotonic() - completed_at >= 15:
                    break
            time.sleep(max(0, args.interval - (time.monotonic() - cycle)))
    print(f"Monitoring complete: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
