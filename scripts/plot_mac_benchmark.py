"""Export the Mac monitoring log as a PNG chart and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.benchmark_dir / "mac_metrics.jsonl").read_text().splitlines()]
    summary = json.loads((args.benchmark_dir / "mac_monitor_summary.json").read_text())
    report = json.loads((args.benchmark_dir / "benchmark.json").read_text())
    times = [r["elapsed_s"] / 60 for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, height_ratios=[2, 2, 1])
    title = report.get("title", "YOLO11s-pose")
    backend = "MPS FP16"
    if report.get("pipeline") in {"patterns", "plays"} and not any(run.get("pipeline_timing", {}).get("sam_calls", 0) for run in report["runs"]):
        backend = "CPU · cached model outputs"
    fig.suptitle(f"MacBook Pro · M4 Pro (20 GPU cores) · 48 GB\n{title} · {backend}", fontsize=15, y=0.98)

    for key, label, color in (("cpu_percent", "Whole-Mac CPU", "#e58017"), ("gpu_util_percent", "Whole-Mac GPU", "#1266ad")):
        axes[0].plot(times, [r.get(key) for r in rows], label=label, color=color, linewidth=1.5)
    axes[0].set(ylabel="Utilization (%)", ylim=(0, 105))
    axes[0].legend(loc="center right")

    for key, label, color in (("memory_available_gib", "Available system RAM", "#22804e"),
                              ("gpu_memory_in_use_gib", "Whole-Mac GPU memory in use", "#1266ad"),
                              ("process_rss_gib", "Pipeline process RSS", "#a8409b")):
        axes[1].plot(times, [r.get(key) for r in rows], label=label, color=color, linewidth=1.5)
    axes[1].set(ylabel="Memory (GiB)", ylim=(0, None))
    axes[1].legend(loc="center right")

    thermal = {"nominal": 0, "fair": 1, "serious": 2, "critical": 3}
    axes[2].step(times, [thermal.get(r["thermal_state"]) for r in rows], where="post", color="#a8409b", linewidth=2)
    axes[2].set(yticks=list(thermal.values()), yticklabels=list(thermal), ylim=(-0.15, 3.15),
                ylabel="Thermal pressure", xlabel=f"Minutes since monitoring began ({rows[0]['time']})")

    previous = None
    for row, minute in zip(rows, times):
        name = row["active_video"] or "Idle"
        if name != previous:
            for ax in axes:
                ax.axvline(minute, color="#8c949f", linewidth=0.8, linestyle="--")
            axes[0].text(minute + 0.03, 101, name, fontsize=9, va="top")
            previous = name
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, times[-1])
    fig.text(0.08, 0.035,
             textwrap.fill(summary["coverage"], width=115) + "\n"
             "CPU/GPU include other apps. GPU memory is shared RAM; RSS excludes GPU allocations.\n"
             "Thermal pressure is not temperature. "
             f"Swap-outs during monitoring: {summary['swap_out_bytes_during_monitoring'] / 2**20:.1f} MiB.", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.10, right=0.97, hspace=0.18)
    chart = args.benchmark_dir / "mac-performance.png"
    fig.savefig(chart, dpi=160, facecolor="white")
    flat_keys = [key for key in rows[0] if key != "cpu_per_core_percent"]
    with (args.benchmark_dir / "mac-performance.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(chart)


if __name__ == "__main__":
    main()
