"""Select full rallies or scoring plays from sport-event observations.

Boundary observations come from a detector or annotation file. Pose-only activity
estimates are an explicit diagnostic mode and never count as observed landings.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import math
from pathlib import Path
import time

from highlight.activity import activity_events, court_activity
from highlight.court import Court, load_net, load_polygon
from highlight.game_units import SportEvent, badminton_units, rank_units, scoring_units
from highlight.pattern_signals import highlight_candidates, mine_actions, sam_support
from highlight.patterns import _cut_review_clip, _reel, load_evidence
from highlight.pose import file_identity, load_poses


def read_events(path: Path, video: Path, sport: str) -> list[SportEvent]:
    data = json.loads(path.read_text())
    if data.get("sport") != sport:
        raise ValueError("Event file sport does not match --sport")
    source = Path(data["video"])
    if not source.is_absolute():
        source = path.parent / source
    if source.resolve() != video.resolve():
        raise ValueError("Event observations belong to a different video")
    result = []
    for item in data["events"]:
        values = dict(item)
        values.setdefault("court_id", data.get("court_id", "main"))
        values.setdefault("source", data.get("source", "external"))
        result.append(SportEvent(**values))
    return result


def write_review(path: Path, report: dict) -> None:
    cards = []
    esc = html.escape
    for i, unit in enumerate(report["selected_units"], 1):
        media = f'<video controls preload="none" src="{esc(unit["clip"], quote=True)}"></video>' if unit.get("clip") else ""
        cards.append(f'<article><h2>{i}. {esc(unit["unit_type"])} · {unit["play_start_s"]:.2f}–{unit["play_end_s"]:.2f}s</h2>'
                     f'{media}<p>Boundary status: {esc(unit["boundary_status"])}. '
                     f'Start: {esc(unit["start_reason"])}. End: {esc(unit["end_reason"])}.</p>'
                     f'<p>Interest score: {unit["ranking_score"]:.3f}; action pattern matches: {unit["features"]["action_match_count"]}.</p></article>')
    note = ("Pose activity alone cannot establish service or ground contact. These are activity-boundary proposals."
            if report["mode"] == "pose_activity_estimates" else
            "Units follow the supplied sport events. Verification flags come from the event source, independently of highlight ranking.")
    path.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Game-unit highlight review</title><style>body{font:17px system-ui;max-width:1050px;margin:30px auto;padding:0 20px;background:#f4f6f8;color:#19232d}article{background:white;padding:20px;margin:20px 0;border-radius:12px}video{width:100%}p{line-height:1.5}</style>
<h1>Rallies and scoring plays</h1><p>''' + esc(note) + '</p>' + ''.join(cards) +
                    ('<p>No whole units fit the current selection criteria. No rally was shortened to fit the budget.</p>' if not cards else '') + '</html>')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--sport", required=True, choices=["badminton", "basketball", "football"])
    boundary = parser.add_mutually_exclusive_group(required=True)
    boundary.add_argument("--events", type=Path, help="timestamped serve/end or goal observations")
    boundary.add_argument("--estimate-from-poses", action="store_true", help="diagnostic activity proposals; not detected serves/landings")
    parser.add_argument("--poses", type=Path, help="optional cached poses for activity or within-unit interest ranking")
    parser.add_argument("--sam-evidence", type=Path, help="reuse saved badminton SAM evidence for within-unit ranking")
    parser.add_argument("--court", type=Path, help="required whenever poses contribute")
    parser.add_argument("--court-id", default="main", help="ignore observations assigned to other courts")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/plays"))
    parser.add_argument("--max-highlights", type=int, default=5)
    parser.add_argument("--highlight-budget", type=float, default=120)
    parser.add_argument("--verified-only", action="store_true", help="only select units with both boundary observations verified")
    parser.add_argument("--idle-seconds", type=float, default=2.5)
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument("--reel", action="store_true")
    args = parser.parse_args(argv)
    if args.max_highlights < 0 or not math.isfinite(args.highlight_budget) or args.highlight_budget < 0:
        parser.error("Counts and duration budget must be nonnegative")
    if not math.isfinite(args.idle_seconds) or args.idle_seconds <= 0:
        parser.error("--idle-seconds must be finite and positive")
    if args.reel and args.no_clips:
        parser.error("--reel requires clip export")
    if args.estimate_from_poses and (args.sport != "badminton" or args.poses is None):
        parser.error("Activity estimates require badminton and --poses")
    if args.poses is not None and args.court is None:
        parser.error("--poses requires --court so other courts cannot contribute")
    if args.sam_evidence and (args.sport != "badminton" or args.poses is None):
        parser.error("--sam-evidence requires badminton and --poses")
    started = time.perf_counter()
    timings = {"yolo_calls": 0, "sam_calls": 0}
    poses = meta = court = None
    if args.poses is not None:
        poses, meta = load_poses(args.poses)
        if Path(meta["video"]).resolve() != args.video.resolve():
            parser.error("Pose archive belongs to another video")
        if meta.get("source_identity") is not None and meta["source_identity"] != file_identity(args.video):
            parser.error("Source video changed since pose extraction")
        if not math.isfinite(meta["fps"]) or meta["fps"] <= 0 or meta["frames"] <= 0:
            parser.error("Pose archive must have a positive frame rate and frame count")
        duration = meta["frames"] / meta["fps"]
        court = Court(meta["width"], meta["height"], meta["fps"],
                      load_polygon(args.court, meta["width"], meta["height"]), [], "manual",
                      "Only players inside the selected court contribute.", load_net(args.court, meta["width"], meta["height"]))
    else:
        import cv2
        capture = cv2.VideoCapture(str(args.video))
        try:
            if not capture.isOpened() or capture.get(cv2.CAP_PROP_FPS) <= 0:
                parser.error("Cannot read source video duration")
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
    samples = []
    t0 = time.perf_counter()
    if args.estimate_from_poses:
        samples = court_activity(poses, court)
        events = activity_events(samples, args.court_id, idle_s=args.idle_seconds)
        mode = "pose_activity_estimates"
    else:
        try:
            events = read_events(args.events, args.video, args.sport)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(str(exc))
        mode = "sport_events"
    if args.sport == "badminton":
        units = badminton_units(events, duration, args.court_id, allow_activity_estimates=args.estimate_from_poses)
    else:
        units = scoring_units(events, args.sport, duration, args.court_id)
    timings["unit_detection_s"] = time.perf_counter() - t0
    # Pose patterns are auxiliary ranking features; they cannot create/shorten an
    # event-defined rally or generate a basketball/football goal.
    t0 = time.perf_counter()
    actions = []
    details = []
    if poses is not None and args.sport == "badminton":
        if args.sam_evidence:
            try:
                evidence, details = load_evidence(args.sam_evidence, args.video)
                if evidence.get("court") and (evidence["court"]["width"], evidence["court"]["height"]) != (meta["width"], meta["height"]):
                    parser.error("SAM evidence and pose archive use different image geometry")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                parser.error(str(exc))
        matches, _ = mine_actions(poses, court, duration)
        actions = highlight_candidates(matches)
        for action in actions:
            action.sam = sam_support(action, poses, details, court)
    selected = rank_units(units, actions, args.max_highlights, args.highlight_budget, args.verified_only)
    timings["interest_ranking_s"] = time.perf_counter() - t0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    clips, windows = [], []
    for i, unit in enumerate(selected, 1):
        window = {**asdict(unit), "boundary_status": unit.boundary_status}
        if not args.no_clips:
            clip = args.out_dir / f"{args.video.stem}_play{i}_{unit.unit_type}.mp4"
            _cut_review_clip(args.video, unit.start_s, unit.end_s, clip)
            window["clip"] = clip.name
            clips.append(clip)
        windows.append(window)
    reel = args.out_dir / f"{args.video.stem}_plays.mp4"
    if args.reel and clips:
        _reel(clips, reel)
    timings["export_s"] = time.perf_counter() - t0
    timings["total_before_json_s"] = time.perf_counter() - started
    report = {"video": str(args.video.resolve()), "sport": args.sport, "mode": mode,
              "court_id": args.court_id, "duration_s": duration, "court": asdict(court) if court else None,
              "events_source": str(args.events.resolve()) if args.events else None,
              "poses_source": str(args.poses.resolve()) if args.poses else None,
              "sam_evidence": str(args.sam_evidence.resolve()) if args.sam_evidence else None,
              "sam_evidence_frames_reused": len(details),
              "timing": timings, "events": [asdict(e) for e in events], "activity_samples": samples,
              "units": [{**asdict(u), "boundary_status": u.boundary_status} for u in units],
              "selected_units": windows, "reel": reel.name if args.reel and clips else None,
              "settings": {"max_highlights": args.max_highlights, "highlight_budget_s": args.highlight_budget,
                           "verified_only": args.verified_only, "idle_seconds": args.idle_seconds},
              "limitations": ["This command consumes boundary observations; it does not train or run a serve, landing, or football-goal detector.",
                              "Pose-only pauses do not establish the shuttle landing or a complete rally.",
                              "Event verification and interest ranking are independent."]}
    path = args.out_dir / f"{args.video.stem}_plays.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    write_review(args.out_dir / "review.html", report)
    print(f"{args.sport}: {len(units)} unit(s), {len(windows)} selected; "
          f"{sum(u.boundary_status == 'verified' for u in units)} with verified boundaries; {timings['total_before_json_s']:.2f}s", flush=True)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
