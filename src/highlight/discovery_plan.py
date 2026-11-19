"""Pure, deterministic SAM request scheduling with independent discovery slots."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Probe:
    frame_index: int
    crop: tuple[int, int, int, int]
    branch: str
    reason: str


def reservations(total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("SAM budget must be nonnegative")
    scout = max(1, int(total * .3)) if total else 0
    uncertain = int(total * .3)
    return {"scout": scout, "uncertainty": uncertain, "detail": total - scout - uncertain}


def context_crops(court, tiles: int = 1) -> list[tuple[int, int, int, int]]:
    if tiles not in (1, 2):
        raise ValueError("Context tiles must be 1 or 2")
    polygon = np.array(court.polygon)
    left = max(0, math.floor(polygon[:, 0].min() - .02*court.width))
    right = min(court.width, math.ceil(polygon[:, 0].max() + .02*court.width))
    top = max(0, math.floor(polygon[:, 1].min() - .35*court.height))
    bottom = min(court.height, math.ceil(polygon[:, 1].max()))
    if tiles == 1:
        return [(left, top, right, bottom)]
    middle, overlap = (left+right)//2, max(1, round((right-left)*.08))
    return [(left, top, min(right, middle+overlap), bottom), (max(left, middle-overlap), top, right, bottom)]


def nearest_frame(rows: list[dict], t: float) -> int:
    times = np.array([r["time_s"] for r in rows])
    return int(rows[int(np.argmin(abs(times-t)))]["frame_index"])


def spread(records: list[dict], limit: int, duration: float, score: str = "score") -> list[dict]:
    """One strongest record per time bin, then fill unused slots; never first-N only."""
    if limit <= 0:
        return []
    bins = {}
    for record in records:
        bucket = min(limit-1, int(record["time_s"] / max(duration, .001) * limit))
        if bucket not in bins or record[score] > bins[bucket][score]:
            bins[bucket] = record
    chosen = list(bins.values())
    used = {id(r) for r in chosen}
    chosen += sorted((r for r in records if id(r) not in used), key=lambda r: -r[score])[:limit-len(chosen)]
    return sorted(chosen, key=lambda r: r["time_s"])


def scout_probes(rows: list[dict], crops: list[tuple], slots: int, interval: float, duration: float) -> list[Probe]:
    if slots <= 0 or not rows:
        return []
    count = min(slots, max(1, math.ceil(duration / interval)))
    return list(dict.fromkeys(Probe(nearest_frame(rows, t), crops[i % len(crops)], "scout", "pose_independent_time_coverage")
                              for i, t in enumerate((np.arange(count)+.5)*duration/count)))


def cv_peaks(rows: list[dict], threshold: float, cooldown: float = 2) -> list[dict]:
    buckets = {}
    for previous, row in zip(rows, rows[1:]):
        if (row["scene_change_candidate"] or row.get("camera_unstable") or previous["scene_change_candidate"]
                or previous.get("camera_unstable") or min(previous["roi_motion_fraction"], row["roi_motion_fraction"]) < threshold):
            continue
        bucket = int(row["time_s"] / cooldown)
        if bucket not in buckets or row["roi_motion_fraction"] > buckets[bucket]["score"]:
            buckets[bucket] = {"frame_index": row["frame_index"], "time_s": row["time_s"],
                               "score": row["roi_motion_fraction"], "usable_pose": row.get("usable_pose", False)}
    return list(buckets.values())


def uncertainty_probes(peaks: list[dict], crops: list[tuple], slots: int, duration: float) -> list[Probe]:
    candidates = [p for p in peaks if not p["usable_pose"]]
    return [Probe(r["frame_index"], crops[i % len(crops)], "uncertainty", "roi_motion_without_usable_pose")
            for i, r in enumerate(spread(candidates, slots, duration))]


def detail_crop(box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = max((x2-x1)*2.4, width*.25), max((y2-y1)*2.4, height*.25)
    return (max(0, math.floor(cx-w/2)), max(0, math.floor(cy-h/2)),
            min(width, math.ceil(cx+w/2)), min(height, math.ceil(cy+h/2)))


def refinement_probes(observations: list[dict], units: list[dict], rows: list[dict], court,
                      slots: int, seen: set[tuple]) -> list[Probe]:
    if slots <= 0 or not rows:
        return []
    parents = []
    for i, row in enumerate(observations):
        objects = [o for o in row["objects"] if o["confidence"] >= .4]
        objects = [o for o in objects if o["native_label"].lower() != "person" or
                   court.contains(np.array([(o["box"][0]+o["box"][2])/2, o["box"][3]]), margin=0)]
        if not objects:
            continue
        # Diversify followups: static spare equipment must not always outrank people.
        preferred = [o for o in objects if (o["native_label"].lower() == "person") == bool(i % 2)]
        obj = max(preferred or objects, key=lambda o: o["confidence"])
        parents.append({"frame_index": row["frame_index"], "time_s": row["time_s"], "score": obj["confidence"],
                        "crop": detail_crop(obj["box"], court.width, court.height), "reason": "independent_object_followup"})
    for unit in units:
        t = unit.get("peak_time_s", (unit["start_s"]+unit["end_s"])/2)
        parents.append({"frame_index": nearest_frame(rows, t), "time_s": t, "score": .5,
                        "crop": context_crops(court)[0], "reason": unit.get("sam_detail_reason", "existing_candidate_detail")})
    duration = rows[-1]["time_s"] + 1/court.fps
    parents = spread(parents, max(1, math.ceil(slots/2)), duration)
    result = []
    keys = set(seen)
    # Visit every parent before adding its second/third neighboring frame.
    for offset in [0, .3, -.3]:
        for parent in parents:
            frame = nearest_frame(rows, parent["time_s"]+offset)
            key = (frame, parent["crop"])
            if key not in keys:
                result.append(Probe(frame, parent["crop"], "detail", parent["reason"]))
                keys.add(key)
                if len(result) >= slots:
                    return result
    return result


def sampling_coverage(observations: list[dict], duration: float) -> dict:
    groups = {}
    for row in observations:
        groups.setdefault(str(row["crop"]), []).append(row["time_s"])
    return {crop: {"images": len(times), "times_s": sorted(set(times)),
                   "max_gap_including_edges_s": max(np.diff([0, *sorted(set(times)), duration])).item()}
            for crop, times in groups.items()}
