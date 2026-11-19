"""Source-bound interchange for external model observations, without inference."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from highlight.discovery_sam import digest
from highlight.discovery_signals import video_metadata
from highlight.fusion_signals import PROFILES
from highlight.pose import file_identity

KINDS = {"observed", "tracked", "interpolated", "predicted"}


def stream_id(stream: dict, meta: dict) -> str:
    return "model-" + digest({"source_identity": meta["source_identity"], "model": stream["model"]})[:20]


def document(meta: dict, sport: str, streams: list[dict]) -> dict:
    return {"schema_version": 1, "video": meta["video"], "source_identity": meta["source_identity"],
            "sport": sport, "width": meta["width"], "height": meta["height"], "streams": streams}


def validate_stream(stream: dict, meta: dict) -> None:
    model = stream["model"]
    for key in ("name", "checkpoint_id", "backend", "run_id"):
        if not isinstance(model[key], str) or not model[key].strip():
            raise ValueError(f"Model provenance requires {key}")
    previous, previous_time = -1, -1.
    for row in stream["rows"]:
        frame, t = row["frame_index"], row["time_s"]
        if type(frame) is not int or not previous < frame < meta["frames"]:
            raise ValueError("Model frames must be unique, increasing source frame indices")
        previous = frame
        if (type(t) not in (int, float) or not math.isfinite(t) or not max(0., previous_time) <= t <= meta["frames"]/meta["fps"]+1):
            raise ValueError("Invalid model timestamp")
        previous_time = t
        crop = row["crop"]
        if (not isinstance(crop, list) or len(crop) != 4 or any(type(x) is not int for x in crop)
                or not 0 <= crop[0] < crop[2] <= meta["width"] or not 0 <= crop[1] < crop[3] <= meta["height"]):
            raise ValueError("Model crop must use original image coordinates")
        if row["time_basis"] not in {"nominal_frame_rate", "source_decoder"}:
            raise ValueError("Unknown model time basis")
        track_ids = set()
        for obj in row["objects"]:
            box, confidence = obj["box"], obj["confidence"]
            if (not isinstance(obj["label"], str) or not obj["label"].strip() or obj["observation_kind"] not in KINDS
                    or not isinstance(box, list) or len(box) != 4 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                    or not crop[0] <= box[0] < box[2] <= crop[2] or not crop[1] <= box[1] < box[3] <= crop[3]):
                raise ValueError("Invalid native model label, kind or global box")
            if confidence is not None and (type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("Model confidence must be null or in [0, 1]")
            if "point" in obj:
                point = obj["point"]
                if (not isinstance(point, list) or len(point) != 2 or
                        any(type(x) not in (int, float) or not math.isfinite(x) for x in point) or
                        not crop[0] <= point[0] < crop[2] or not crop[1] <= point[1] < crop[3]):
                    raise ValueError("Invalid source point")
            track = obj.get("source_track_id")
            if track is not None:
                if type(track) not in (str, int) or str(track) in track_ids:
                    raise ValueError("Invalid or repeated source track ID within one frame")
                track_ids.add(str(track))


def load_streams(paths: list[Path], meta: dict, sport: str) -> list[dict]:
    streams = []
    for path in paths:
        data = json.loads(path.read_text())
        if (data["schema_version"] != 1 or Path(data["video"]).resolve() != Path(meta["video"]).resolve()
                or data["source_identity"] != meta["source_identity"] or data["sport"] != sport
                or (data["width"], data["height"]) != (meta["width"], meta["height"])):
            raise ValueError("Model evidence source, geometry or sport differs")
        for stream in data["streams"]:
            validate_stream(stream, meta)
            streams.append(stream)
    return unique_streams(streams, meta)


def unique_streams(streams: list[dict], meta: dict) -> list[dict]:
    result = {}
    for stream in streams:
        key = stream_id(stream, meta)
        if key in result and result[key] != stream:
            raise ValueError("Different model runs share the same provenance; supply distinct run_id values")
        result[key] = stream
    return list(result.values())


def import_csv(path: Path, meta: dict, format: str, label: str, kind: str,
               model: dict, frame_offset: int = 0) -> dict:
    """TrackNet native CSV or explicit frame_index,x,y,width,height box export.

    Inputs are full-resolution, consecutive frames (possibly a temporal clip),
    never resized/cropped/strided frame numbers guessed into the source video.
    """
    rows = []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = {"Frame", "Visibility", "X", "Y"} if format == "tracknet" else {"frame_index", "x", "y", "width", "height"}
        if not fields <= set(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(fields-set(reader.fieldnames or []))}")
        for item in reader:
            frame = int(item["Frame" if format == "tracknet" else "frame_index"]) + frame_offset
            visible = int(item["Visibility"]) if format == "tracknet" else 1
            if visible not in (0, 1):
                raise ValueError("TrackNet Visibility must be 0 or 1")
            objects = []
            x, y = float(item["X" if format == "tracknet" else "x"]), float(item["Y" if format == "tracknet" else "y"])
            w, h = (2., 2.) if format == "tracknet" else (float(item["width"]), float(item["height"]))
            if not all(math.isfinite(v) for v in [x, y, w, h]):
                raise ValueError("Nonfinite CSV coordinate")
            if w < 0 or h < 0:
                raise ValueError("Negative CSV box extent")
            if visible and (format == "tracknet" or (w > 0 and h > 0)):
                if format == "tracknet":
                    if not 0 <= x < meta["width"] or not 0 <= y < meta["height"]:
                        raise ValueError("TrackNet point outside source; resized coordinates need explicit conversion first")
                    box = [max(0., x-1), max(0., y-1), min(meta["width"], x+1), min(meta["height"], y+1)]
                else:
                    box = [x, y, x+w, y+h]
                objects = [{"label": label, "box": box,
                            "confidence": float(item["confidence"]) if item.get("confidence") else None,
                            "observation_kind": kind, "source_track_id": item.get("track_id") or "0"}]
                if format == "tracknet":
                    objects[0].update(point=[x, y], geometry_kind="point_with_display_box")
            if rows and frame == rows[-1]["frame_index"]:
                rows[-1]["objects"].extend(objects)
            else:
                rows.append({"frame_index": frame, "time_s": frame/meta["fps"], "time_basis": "nominal_frame_rate",
                             "crop": [0, 0, meta["width"], meta["height"]], "objects": objects})
    stream = {"model": model, "rows": rows}
    validate_stream(stream, meta)
    return stream


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--sport", choices=sorted(PROFILES), required=True)
    parser.add_argument("--format", choices=["tracknet", "boxes"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-id", required=True, help="external checkpoint/version/hash declaration")
    parser.add_argument("--backend", required=True, help="backend used by external inference; no inference is run here")
    parser.add_argument("--label", required=True, help="native label, never infer a species from a generic ball label")
    parser.add_argument("--kind", choices=sorted(KINDS), required=True, help="inpainted CSV must not be declared observed")
    parser.add_argument("--frame-offset", type=int, default=0, help="source start frame for a full-resolution temporal clip")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.frame_offset < 0:
        parser.error("Use a new output file and a nonnegative frame offset")
    try:
        meta = video_metadata(args.video)
        model = {"name": args.model_name, "checkpoint_id": args.checkpoint_id, "backend": args.backend,
                 "run_id": digest(file_identity(args.csv))[:20], "provenance": "externally_declared"}
        stream = import_csv(args.csv, meta, args.format, args.label, args.kind, model, args.frame_offset)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document(meta, args.sport, [stream]), indent=2, allow_nan=False)+"\n")
    print(f"saved {args.out}; {len(stream['rows'])} samples; no model inference", flush=True)


if __name__ == "__main__":
    main()
