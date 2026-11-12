"""Extract temporal action highlights from saved poses and SAM3 evidence.

No YOLO inference is performed. SAM3 runs only when --refine-sam is requested.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import html
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np

from highlight.badminton import analyze_details
from highlight.badminton_events import Moment, keyframe_plan
from highlight.candidates import Window
from highlight.court import Court, load_net, load_polygon
from highlight.extract_pose import MODELS_DIR
from highlight.pattern_signals import Action, action_bursts, highlight_candidates, mine_actions, sam_support, select_actions
from highlight.pose import FramePoses, file_identity, load_poses
from highlight.sam import Sam3, require_weights
from highlight.video import resolve_device
from highlight.viz import draw_poses


def load_evidence(path: Path, video: Path, seen: set[Path] | None = None) -> tuple[dict, list[dict]]:
    """Follow saved evidence provenance so changing selection need not repeat SAM."""
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        raise ValueError("SAM evidence provenance contains a cycle")
    seen.add(path)
    report = json.loads(path.read_text())
    if Path(report["video"]).resolve() != video.resolve():
        raise ValueError("SAM evidence belongs to a different source video")
    details = []
    if report.get("sam_evidence"):
        parent = Path(report["sam_evidence"])
        if not parent.is_absolute():
            parent = path.parent / parent
        _, details = load_evidence(parent, video, seen)
    details += [detail for window in report.get("windows", []) for detail in window.get("details", [])]
    details += report.get("fresh_sam_details", [])
    unique = {(detail["frame_index"], tuple(detail.get("crop", []))): detail for detail in details}
    return report, list(unique.values())


def already_sampled(frame: int, box: np.ndarray, details: list[dict]) -> bool:
    """An existing image call is reusable only if its crop covered this player."""
    for detail in details:
        if detail["frame_index"] != frame:
            continue
        if detail.get("crop") is None:
            return True
        crop = np.asarray(detail["crop"])
        intersection = np.maximum(np.minimum(crop[2:], box[2:]) - np.maximum(crop[:2], box[:2]), 0)
        if np.prod(intersection) >= 0.8 * np.prod(box[2:] - box[:2]):
            return True
    return False


def _preview(video: Path, action: Action, poses: dict, destination: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, action.frame_index):
            raise RuntimeError(f"Cannot seek to {action.frame_index}")
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode preview frame {action.frame_index}")
    finally:
        capture.release()
    rows = np.flatnonzero((poses["frame_index"] == action.frame_index) & (poses["track_id"] == action.track_id))
    if len(rows):
        image = draw_poses(image, FramePoses(poses["track_id"][rows], poses["box"][rows], poses["score"][rows], poses["keypoints"][rows]))
    # Only accepted nearby evidence; raw spare-shuttle/wall boxes stay in JSON.
    for obj in action.sam.get("accepted_objects", []):
        if abs(obj["frame_index"] - action.frame_index) > 3:
            continue
        x1, y1, x2, y2 = np.rint(obj["box"]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 95), (25, 25, 25), -1)
    cv2.putText(image, f"{action.pattern} | {action.time_s:.2f}s | ranking {action.ranking_score:.3f}",
                (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(image, f"SAM: {action.sam.get('status', 'not_sampled')} | candidate; outcome unknown",
                (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Cannot save {destination}")
    return image


def _cut_review_clip(video: Path, start_s: float, end_s: float, destination: Path) -> None:
    """Transcoding discards pre-seek keyframe context; make review clips 720p30."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start_s:.6f}",
                    "-i", str(video), "-t", f"{end_s - start_s:.6f}", "-map", "0:v:0", "-map", "0:a?",
                    "-vf", "scale=1280:-2,fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    "-c:a", "aac", "-movflags", "+faststart", str(destination)], check=True)


def _reel(clips: list[Path], destination: Path) -> None:
    """A small H.264/AAC review reel, compatible with ordinary video players."""
    if not clips:
        return
    listing = destination.with_suffix(".concat.txt")
    listing.write_text("".join("file '" + str(clip.resolve()).replace("'", "'\\''") + "'\n" for clip in clips))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(destination)], check=True)


def _review(path: Path, result: dict) -> None:
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for window in result["windows"]:
        media = (f'<video controls preload="none" poster="{esc(window.get("preview", ""))}" src="{esc(window["clip"])}"></video>'
                 if "clip" in window else f'<img src="{esc(window["preview"])}">' if "preview" in window else "")
        phases = " → ".join(f"{esc(name)} {value:.2f}s" for name, value in window["phases"].items())
        cards.append(f'<article><h2>{esc(window["pattern"])} · {window["time_s"]:.2f}s</h2>{media}'
                     f'<p>{phases}</p><p>SAM evidence: {esc(window["sam"]["status"])}. Ranking score: {window["ranking_score"]:.3f}.</p>'
                     f'<details><summary>Pose measurements</summary><pre>{esc(json.dumps(window["features"], indent=2))}</pre></details></article>')
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pattern review: {esc(result['video'])}</title><style>body{{font:16px system-ui;background:#f3f5f7;color:#18202a;max-width:1100px;margin:30px auto;padding:0 20px}}article{{background:white;padding:18px;margin:20px 0;border-radius:12px}}video,img{{width:100%;max-height:600px}}pre{{white-space:pre-wrap}}h1{{font-size:27px}}p{{line-height:1.5}}</style>
<h1>Temporal action candidates</h1><p>{esc(Path(result['video']).name)} · {len(result['windows'])} selected clips.
No YOLO rerun. These are pose sequence matches, not verified smashes, winners, or complete rallies.</p>
<p>{esc(result['court']['note'])}</p><p>Pattern counts within the selected court: {esc(result['pattern_counts'])}.</p>
<p><a href="{esc(Path(result['video']).stem + '_patterns.json')}">Full data</a> · <a href="patterns.csv">Measurements CSV</a></p>
{''.join(cards) if cards else '<p>No complete action sequence matched the current thresholds.</p>'}</html>'''
    path.write_text(body)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--sam-evidence", type=Path, help="saved badminton _highlights.json with detail frames")
    parser.add_argument("--court", type=Path, help="four-corner court JSON; overrides saved court calibration")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/patterns"))
    parser.add_argument("--max-highlights", type=int, default=5)
    parser.add_argument("--highlight-budget", type=float, default=40)
    parser.add_argument("--min-pose-score", type=float, default=0.7, help="heuristic ranking score, not a probability")
    parser.add_argument("--no-bursts", action="store_true", help="skip repeated multi-track action groups")
    parser.add_argument("--include-low-reach", action="store_true", help="also select exploratory low reaches; these can be shuttle pickups")
    parser.add_argument("--refine-sam", action="store_true", help="inspect selected pattern peaks with a bounded SAM3 pass")
    parser.add_argument("--sam-model", type=Path, default=MODELS_DIR / "sam3.pt")
    parser.add_argument("--max-sam-calls", type=int, default=15)
    parser.add_argument("--device", help="GPU for optional SAM refinement")
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--reel", action="store_true", help="also create a 720p H.264/AAC review reel")
    args = parser.parse_args(argv)
    if args.max_highlights < 0 or args.max_sam_calls < 0 or not math.isfinite(args.highlight_budget) or args.highlight_budget < 0:
        parser.error("Counts and highlight duration must be nonnegative")
    if not math.isfinite(args.min_pose_score) or not 0 <= args.min_pose_score <= 1:
        parser.error("--min-pose-score must be in [0, 1]")
    if args.reel and args.no_clips:
        parser.error("--reel requires clip export")
    started = time.perf_counter()
    poses, meta = load_poses(args.poses)
    if Path(meta["video"]).resolve() != args.video.resolve():
        parser.error("Pose archive belongs to a different source video")
    if meta.get("source_identity") is not None and meta["source_identity"] != file_identity(args.video):
        parser.error("Source video changed since pose extraction")
    if not meta["fps"] > 0 or meta["frames"] <= 0:
        parser.error("Pose archive must describe a nonempty video with a positive frame rate")
    try:
        evidence, details = load_evidence(args.sam_evidence, args.video) if args.sam_evidence else ({}, [])
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    court = Court(**evidence["court"]) if evidence.get("court") else Court(meta["width"], meta["height"], meta["fps"], None, [], "unconfigured", "No target court supplied; visible players from adjacent courts can contribute.")
    if (court.width, court.height) != (meta["width"], meta["height"]):
        parser.error("SAM evidence and pose archive use different image geometry")
    if args.court:
        court.polygon = load_polygon(args.court, court.width, court.height)
        court.source = "manual"
        court.net = load_net(args.court, court.width, court.height)
        court.note = json.loads(args.court.read_text()).get("note", "Player feet are filtered to the supplied court boundary.")
    if court.polygon is None:
        parser.error("Provide --court or saved court calibration so neighboring courts cannot contribute to highlights")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timing = {"load_saved_data_s": time.perf_counter() - started, "pose_reused": True, "yolo_calls": 0}
    t0 = time.perf_counter()
    actions, diagnostics = mine_actions(poses, court, meta["frames"] / meta["fps"])
    eligible = highlight_candidates(actions, args.min_pose_score, args.include_low_reach)
    bursts = [] if args.no_bursts else action_bursts(eligible)
    all_actions = actions + bursts
    eligible += bursts
    timing["pattern_mining_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for action in all_actions:
        action.sam = sam_support(action, poses, details, court)
    selected = select_actions(eligible, args.max_highlights, args.highlight_budget)
    timing.update(saved_sam_analysis_s=time.perf_counter() - t0, sam_calls=0, sam_load_s=0.0, sam_analysis_s=0.0, sam_inference_s=0.0)
    print(f"patterns: {dict(Counter(a.pattern for a in all_actions))}; selected {len(selected)}; YOLO reruns: 0", flush=True)
    fresh_details = []
    if args.refine_sam and selected and args.max_sam_calls:
        moments = [Moment(Window(a.start_s, a.end_s, a.time_s, a.pose_score, a.box), a.frame_index,
                          a.track_id, a.pattern, 0, a.features.get("wrist_speed", 0), 0) for a in selected]
        plan = keyframe_plan(moments, meta["fps"], meta["frames"], per_moment=3, max_calls=args.max_sam_calls)
        plan = [(i, frame) for i, frame in plan if not already_sampled(frame, np.asarray(selected[i].box), details)]
        if plan:
            require_weights(args.sam_model)
            device = resolve_device(args.device)
            t0 = time.perf_counter()
            sam = Sam3(args.sam_model, device=device)
            t1 = time.perf_counter()
            results = analyze_details(args.video, poses, moments, plan, sam, ["badminton racket", "shuttlecock"], args.out_dir, False)
            fresh_details = [detail for group in results for detail in group]
            parameter = next(sam.predictor.model.parameters())
            timing.update(sam_calls=sam.calls, sam_load_s=t1 - t0, sam_analysis_s=time.perf_counter() - t1,
                          sam_inference_s=sam.inference_s, sam_actual_device=str(parameter.device), sam_dtype=str(parameter.dtype))
        for action in selected:
            action.sam = sam_support(action, poses, details + fresh_details, court)
    t0 = time.perf_counter()
    windows, clips, thumbnails = [], [], []
    for i, action in enumerate(selected, 1):
        name = f"{args.video.stem}_pattern{i}_{action.pattern}"
        window = {**asdict(action), "ranking_score": action.ranking_score, "label": "candidate"}
        if not args.no_clips:
            clip = args.out_dir / f"{name}.mp4"
            _cut_review_clip(args.video, action.start_s, action.end_s, clip)
            window["clip"] = clip.name
            clips.append(clip)
        if not args.no_previews:
            preview = args.out_dir / f"{name}.jpg"
            thumbnails.append(cv2.resize(_preview(args.video, action, poses, preview), (640, 360)))
            window["preview"] = preview.name
        windows.append(window)
    if args.reel and clips:
        _reel(clips, args.out_dir / f"{args.video.stem}_pattern_highlights.mp4")
    if thumbnails:
        columns = min(3, len(thumbnails))
        while len(thumbnails) % columns:
            thumbnails.append(np.zeros_like(thumbnails[0]))
        sheet = np.vstack([np.hstack(thumbnails[i:i + columns]) for i in range(0, len(thumbnails), columns)])
        if not cv2.imwrite(str(args.out_dir / "patterns-contact-sheet.jpg"), sheet):
            raise OSError("Cannot save pattern contact sheet")
    timing["export_s"] = time.perf_counter() - t0
    summaries = {}
    for pattern in sorted({a.pattern for a in all_actions}):
        matches = [a for a in all_actions if a.pattern == pattern]
        summaries[pattern] = {"matches": len(matches), "median_pose_score": float(np.median([a.pose_score for a in matches])),
                              "matched_track_ids": len({a.track_id for a in matches}),
                              "representative_times_s": [a.time_s for a in sorted(matches, key=lambda a: -a.ranking_score)[:5]]}
    timing["total_before_json_s"] = time.perf_counter() - started
    result = {"video": str(args.video.resolve()), "poses": str(args.poses.resolve()),
              "sam_evidence": str(args.sam_evidence.resolve()) if args.sam_evidence else None,
              "pipeline": "cached_pose_temporal_patterns", "duration_s": meta["frames"] / meta["fps"],
              "scope": {"court_only": True, "foot_boundary_margin_px": 0, "net_ground_segment": court.net},
              "court": asdict(court), "timing": timing, "diagnostics": diagnostics,
              "pattern_counts": dict(Counter(a.pattern for a in all_actions)), "pattern_summaries": summaries,
              "settings": {"max_highlights": args.max_highlights, "highlight_budget_s": args.highlight_budget,
                           "min_pose_score": args.min_pose_score, "max_sam_calls": args.max_sam_calls,
                           "refine_sam": args.refine_sam, "bursts_enabled": not args.no_bursts,
                           "include_low_reach": args.include_low_reach},
              "sam_evidence_frames_reused": len(details), "fresh_sam_details": fresh_details,
              "selected_s": sum(a.end_s - a.start_s for a in selected),
              "actions": [{**asdict(a), "ranking_score": a.ranking_score} for a in all_actions], "windows": windows,
              "reel": f"{args.video.stem}_pattern_highlights.mp4" if args.reel and clips else None,
              "limitations": ["Rule matches are not verified smashes, winners, shuttle contacts, or highlight preferences.",
                              "Court exclusion is only as accurate as the supplied court boundary.",
                              "Net-touch candidates do not confirm hairpins without shuttle trajectory evidence.",
                              "Sparse SAM evidence cannot establish shuttle trajectory; absent evidence is unknown.",
                              "Heuristic ranking scores are not calibrated probabilities."]}
    path = args.out_dir / f"{args.video.stem}_patterns.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    with (args.out_dir / "patterns.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pattern", "time_s", "track_id", "pose_score", "ranking_score", "sam_status", "selected"])
        writer.writeheader()
        for action in all_actions:
            writer.writerow({"pattern": action.pattern, "time_s": action.time_s, "track_id": action.track_id,
                             "pose_score": action.pose_score, "ranking_score": action.ranking_score,
                             "sam_status": action.sam["status"], "selected": any(action is a for a in selected)})
    _review(args.out_dir / "review.html", result)
    print(f"saved {path}; {len(windows)} selected patterns, {result['selected_s']:.1f} selected seconds, "
          f"{timing['sam_calls']} fresh SAM calls, {timing['total_before_json_s']:.2f}s elapsed", flush=True)


if __name__ == "__main__":
    main()
