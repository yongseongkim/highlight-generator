"""Mix model evidence by source time and role, without averaging confidences."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from highlight.discovery_candidates import sam_motion_seeds
from highlight.fusion_signals import PROFILES, object_observations
from highlight.model_evidence import stream_id, validate_stream


def align_streams(streams: list[dict], cv_rows: list[dict]) -> list[dict]:
    """The dense source decoding pass, not external FPS arithmetic, sets time."""
    result = deepcopy(streams)
    epochs, epoch, unstable = [], 0, False
    for row in cv_rows:
        camera = row.get("camera_unstable", False)
        if row.get("scene_change_candidate", False) or (camera and not unstable):
            epoch += 1
        epochs.append(epoch)
        unstable = camera
    for stream in result:
        for row in stream["rows"]:
            row.setdefault("reported_time_s", row["time_s"])
            row["time_s"] = cv_rows[row["frame_index"]]["time_s"]
            row["time_basis"] = "source_decoder"
            row["camera_epoch"] = epochs[row["frame_index"]]
    return result


def observation_rows(streams: list[dict], index, meta: dict) -> list[dict]:
    rows = []
    for stream in streams:
        validate_stream(stream, meta)
        identity = stream_id(stream, meta)
        for row in stream["rows"]:
            # Validated input preserves positional correspondence with its objects.
            objects = object_observations(row["objects"], index, row["frame_index"], row["time_s"], identity)
            for obj, raw in zip(objects, row["objects"]):
                obj.update(observation_kind=raw["observation_kind"], source_track_id=raw.get("source_track_id"),
                           model_name=stream["model"]["name"], time_basis=row["time_basis"],
                           crop=row["crop"], reported_time_s=row.get("reported_time_s", row["time_s"]))
                obj["camera_epoch"] = row.get("camera_epoch", 0)
                if "point" in raw:
                    obj.update(point=raw["point"], geometry_kind=raw.get("geometry_kind"))
                if raw.get("source_track_id") is not None:
                    obj["tracklet_id"] = f"{identity}:epoch:{obj['camera_epoch']}:track:{raw['source_track_id']}:{raw['label']}"
                if raw["observation_kind"] not in {"observed", "tracked"}:
                    obj.update(accepted_for_display=False, display_rejection_reason="interpolated_or_predicted")
            rows.append({"frame_index": row["frame_index"], "time_s": row["time_s"], "crop": row["crop"],
                         "stream_id": identity, "model_name": stream["model"]["name"], "objects": objects})
    return sorted(rows, key=lambda row: (row["frame_index"], row["stream_id"]))


def motion_seeds(rows: list[dict], court, sport: str, blocked: set[int]) -> list[dict]:
    seeds = []
    # No temporal links between two models, two runs, or different source tracks.
    for identity in sorted({row["stream_id"] for row in rows}):
        samples = [row for row in rows if row["stream_id"] == identity]
        name = samples[0]["model_name"]
        channel = "yolo_object" if name == "yolo_object" else f"model:{name}"
        seeds += sam_motion_seeds(samples, court, {"person", *PROFILES[sport].native_labels},
                                  blocked_frames=blocked, source=channel)
    return seeds


def coverage(streams: list[dict], meta: dict) -> list[dict]:
    duration = meta["frames"]/meta["fps"]
    result = []
    for stream in streams:
        times = [row["time_s"] for row in stream["rows"]]
        edges = [0., *times, duration]
        result.append({"stream_id": stream_id(stream, meta), "model": stream["model"],
                       "sampled_frames": len(times), "observations_by_kind": dict(Counter(
                           obj["observation_kind"] for row in stream["rows"] for obj in row["objects"])),
                       "max_sample_gap_including_edges_s": max((b-a for a,b in zip(edges, edges[1:])), default=duration),
                       "external_inference_cost_s": None,
                       "note": "External runtime is unknown; reusing observations does not rerun the model."})
    return result


def annotate_units(units: list[dict]) -> None:
    for unit in units:
        counts = Counter(seed["source"] for seed in unit["discovery"]["seeds"])
        unit["model_mix"] = {"seed_counts_by_source": dict(counts),
                             "score_semantics": "Review-interest heuristic; model confidences are not averaged or calibrated.",
                             "boundary_verification_added": False}
