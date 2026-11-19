"""Independent review seeds and bounded context union; never verify game events."""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
from copy import deepcopy
import math

import numpy as np

from highlight.discovery_plan import spread
from highlight.discovery_sam import detail_source, validate_detail
from highlight.fusion_signals import object_observations


def observations(details: list[dict], index, meta: dict) -> list[dict]:
    result = []
    by_input = {(r["frame_index"], tuple(r["crop"])): r for r in details}
    for detail in sorted(by_input.values(), key=lambda r: (r["frame_index"], tuple(r["crop"]))):
        validate_detail(detail, meta)
        frame = detail["frame_index"]
        t = float(detail.get("decoder_time_s", frame/meta["fps"]))
        objects = object_observations(detail["objects"], index, frame, t, detail_source(detail))
        result.append({"frame_index": frame, "time_s": t, "crop": detail["crop"],
                       "branch": detail.get("branch", "legacy"), "objects": objects})
    return result


def sam_motion_seeds(rows: list[dict], court, labels: set[str], max_gap_s: float = 1.,
                     blocked_frames: set[int] | None = None, source: str = "sam") -> list[dict]:
    """Nearest observed same-label pairs; no interpolation, identity or hit claim."""
    tracks = defaultdict(list)
    seeds = []
    blocked = sorted(blocked_frames or set())
    def interrupted(start, end):
        i = bisect_right(blocked, start)
        return i < len(blocked) and blocked[i] <= end
    for row in rows:
        if row["frame_index"] in (blocked_frames or set()):
            tracks.clear()
            continue
        for obj in row["objects"]:
            label = obj["native_label"].lower()
            if (label not in labels or obj.get("observation_kind", "observed") not in {"observed", "tracked"}
                    or obj.get("stationary_candidate") or (obj["confidence"] is not None and obj["confidence"] < .4)):
                continue
            key = (label, obj.get("source_track_id"))
            box = np.array(obj["box"])
            if label == "person" and not court.contains(np.array([(box[0]+box[2])/2, box[3]]), margin=0):
                continue
            center = (box[:2]+box[2:])/2
            active = [history for history in tracks[key] if obj["time_s"]-history[-1]["time_s"] <= max_gap_s
                      and not interrupted(history[-1]["frame_index"], obj["frame_index"])]
            # Duplicate crops at one source frame do not create temporal evidence.
            if any(h[-1]["frame_index"] == obj["frame_index"] and
                   np.linalg.norm(center-(np.array(h[-1]["box"][:2])+h[-1]["box"][2:])/2) < 4 for h in active):
                continue
            matches = [(np.linalg.norm(center-(np.array(h[-1]["box"][:2])+h[-1]["box"][2:])/2), h)
                       for h in active if h[-1]["frame_index"] < obj["frame_index"]]
            history = None
            if matches:
                link_distance, history = min(matches, key=lambda pair: pair[0])
                if link_distance > .2*math.hypot(court.width, court.height):
                    history = None
            if history is not None:
                history[:] = [p for p in history if obj["time_s"]-p["time_s"] <= max_gap_s]
                matched = history[0]
                distance = np.linalg.norm(center-(np.array(matched["box"][:2])+matched["box"][2:])/2)
                if max(4, court.width*.004) <= distance <= .2*math.hypot(court.width, court.height):
                    # Unscored visual tracks retain null confidence. This fallback
                    # weights review interest only, never model certainty.
                    strength = min(p["confidence"] if p["confidence"] is not None else .4 for p in [obj, matched])
                    seeds.append({"frame_index": obj["frame_index"], "time_s": obj["time_s"], "source": source,
                        "score": float(min(1., .5*strength+.5*distance/(.08*court.width))),
                        "evidence_refs": [matched["observation_id"], obj["observation_id"]],
                        "reason": f"{source}_position_change",
                        "court_association": "near_in_court_player" if all(p["association"] == "near_in_court_player" for p in [obj, matched]) else "unknown"})
                history.append(obj)
            else:
                active.append([obj])
            tracks[key] = active[-64:]
    return seeds


def cv_seeds(peaks: list[dict]) -> list[dict]:
    return [{"frame_index": p["frame_index"], "time_s": p["time_s"], "source": "cv",
             "score": min(.9, p["score"]*5), "evidence_refs": [f"cv:{p['frame_index']}"],
             "reason": "roi_motion_peak", "court_association": "ground_roi_motion"} for p in peaks]


def validate_units(units: list[dict], duration: float) -> None:
    for unit in units:
        if (not all(isinstance(unit[k], (int, float)) and math.isfinite(unit[k]) for k in ["start_s", "end_s", "ranking_score"])
                or not 0 <= unit["start_s"] < unit["end_s"] <= duration+.001):
            raise ValueError("Existing candidate has invalid times/score")


def candidate_union(existing: list[dict], seeds: list[dict], duration: float, pre: float = 12, post: float = 8) -> list[dict]:
    validate_units(existing, duration)
    units = deepcopy(existing)
    for unit in units:
        if "candidate_id" in unit:
            unit.setdefault("origin_candidate_id", unit["candidate_id"])
        if "clip" in unit:
            unit["origin_clip"] = unit.pop("clip")
        unit.setdefault("discovery", {"sources": ["pose"], "evidence_refs": [], "seeds": []})
    # Bound seed volume while retaining temporal coverage; all observations remain on disk.
    seeds = [seed for source in source_order({s["source"] for s in seeds})
             for seed in spread([s for s in seeds if s["source"] == source], 120, duration)]
    pending = []
    for seed in seeds:
        found = next((u for u in units if u["start_s"] <= seed["time_s"] <= u["end_s"]), None)
        if found is None:
            pending.append(seed)
        else:
            add_seed(found, seed)
    for seed in sorted(pending, key=lambda s: (-s["score"], s["time_s"])):
        found = next((u for u in units if u["start_s"] <= seed["time_s"] <= u["end_s"]), None)
        if found is not None:
            add_seed(found, seed)
            continue
        t = seed["time_s"]
        before = max([0., *[u["end_s"] for u in units if u["end_s"] < t]])
        after = min([duration, *[u["start_s"] for u in units if u["start_s"] > t]])
        unit = {"start_s": max(before, t-pre), "end_s": min(after, t+post), "peak_time_s": t,
                "ranking_score": seed["score"], "boundary_status": "estimated", "boundary_verified": False,
                "start_reason": "bounded_discovery_context", "end_reason": "bounded_discovery_context",
                "events": [], "decision": "needs_review", "discovery": {"sources": [], "evidence_refs": [], "seeds": []}}
        if unit["end_s"] <= unit["start_s"]:
            continue
        add_seed(unit, seed)
        units.append(unit)
    units.sort(key=lambda u: (u["start_s"], u["end_s"]))
    for i, unit in enumerate(units, 1):
        unit["candidate_id"] = f"D{i:03}"
    return units


def add_seed(unit: dict, seed: dict) -> None:
    data = unit["discovery"]
    if seed["source"] not in data["sources"]:
        data["sources"].append(seed["source"])
    data["evidence_refs"] = sorted(set(data["evidence_refs"]) | set(seed["evidence_refs"]))
    data["seeds"].append(seed)


def select_candidates(units: list[dict], limit: int, seconds: float) -> list[dict]:
    pools = {source: sorted([u for u in units if source in u["discovery"]["sources"]],
                            key=lambda u: -u["ranking_score"]) for source in
             source_order({s for u in units for s in u["discovery"]["sources"]})}
    selected, seen, used = [], set(), 0.
    while any(pools.values()) and len(selected) < limit:
        for pool in pools.values():
            while pool:
                unit = pool.pop(0)
                key = unit["candidate_id"]
                if key in seen:
                    continue
                seen.add(key)
                length = unit["end_s"]-unit["start_s"]
                if used+length > seconds+1e-9:
                    continue
                selected.append(unit)
                used += length
                break
            if len(selected) == limit:
                break
    return sorted(selected, key=lambda u: u["start_s"])


def source_order(sources: set[str]) -> list[str]:
    return [s for s in ["sam", "yolo_object", *sorted(sources-{"sam", "yolo_object", "cv", "pose"}), "cv", "pose"]
            if s in sources]
