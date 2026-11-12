"""Apply an agent's evidence-citing review without re-running vision models."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from highlight.patterns import _reel


def validate_reviews(reviews: list[dict], packets: list[dict]) -> list[dict]:
    by_id = {packet["candidate_id"]: packet for packet in packets}
    seen = set()
    fields = {"candidate_id", "decision", "reason", "evidence_refs", "counterevidence", "unknowns", "boundary_verified"}
    for review in reviews:
        if not isinstance(review, dict) or set(review) != fields:
            raise ValueError("Review fields must match review_schema.json")
        cid = review["candidate_id"]
        if cid not in by_id or cid in seen:
            raise ValueError(f"Unknown or duplicate candidate ID: {cid}")
        seen.add(cid)
        if review["decision"] not in {"highlight_candidate", "reject", "needs_review"}:
            raise ValueError(f"Invalid decision for {cid}")
        if review["boundary_verified"] is not False:
            raise ValueError("An LLM interest judgment cannot newly verify a game boundary")
        if not isinstance(review["reason"], str) or not review["reason"].strip():
            raise ValueError("A review needs a reason")
        for name in ("evidence_refs", "counterevidence", "unknowns"):
            if not isinstance(review[name], list) or not all(isinstance(value, str) for value in review[name]):
                raise ValueError(f"{name} must be an array of strings")
        packet = by_id[cid]
        known = {image["path"] for image in packet["keyframes"]}
        known.update(f"pose:{sample['frame_index']}" for sample in packet["player_context"])
        known.update(sample["observation_id"] for track in packet["object_tracklets"] for sample in track["samples"])
        known.update(obj["observation_id"] for obj in packet.get("rejected_object_observations", []))
        if not review["evidence_refs"] or not set(review["evidence_refs"]) <= known:
            raise ValueError(f"{cid} must cite existing keyframes, pose frames or object observation IDs")
    return reviews


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--review", type=Path, required=True, help="JSON matching review_schema.json")
    parser.add_argument("--reviewer", default="external_agent", help="provenance label, not a model invocation")
    args = parser.parse_args(argv)
    packets = [json.loads(line) for line in (args.analysis_dir / "packets.jsonl").read_text().splitlines()]
    data = json.loads(args.review.read_text())
    try:
        if set(data) != {"reviews"} or not isinstance(data["reviews"], list):
            raise ValueError("Expected an object containing only a reviews array")
        reviews = validate_reviews(data["reviews"], packets)
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    accepted = {r["candidate_id"] for r in reviews if r["decision"] == "highlight_candidate"}
    selected = sorted((p for p in packets if p["candidate_id"] in accepted), key=lambda p: p["interval_s"][0])
    clips = [args.analysis_dir / f"{p['candidate_id']}_overlay.mp4" for p in selected]
    if any(not path.is_file() for path in clips):
        parser.error("Selected overlay clips do not exist; run fusion with video export")
    destination = args.analysis_dir / "reviewed_highlights.mp4"
    if clips:
        _reel(clips, destination)
    result = {"reviewer": args.reviewer, "review_source": str(args.review.resolve()), "reviews": reviews,
              "selected_candidates": [{"candidate_id": p["candidate_id"], "interval_s": p["interval_s"],
                                       "boundary_status": p["boundary_status"]} for p in selected],
              "reel": destination.name if clips else None, "new_model_calls": 0,
              "note": "Agent interest judgments do not add boundary/outcome verification; omitted candidates remain unreviewed."}
    (args.analysis_dir / "reviewed_selection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    cards = []
    for review in reviews:
        esc = html.escape
        cid = review["candidate_id"]
        cards.append(f'<article><h2>{esc(cid)} · {esc(review["decision"])}</h2><p>{esc(review["reason"])}</p>'
                     f'<p>반대 근거: {esc(" / ".join(review["counterevidence"]))}</p>'
                     f'<p>미확인: {esc(" / ".join(review["unknowns"]))}</p>'
                     f'<p>참조: {esc(", ".join(review["evidence_refs"]))}</p>'
                     f'<a href="packets/{esc(cid, quote=True)}.json">근거 묶음</a></article>')
    media = '<video controls preload="metadata" src="reviewed_highlights.mp4"></video>' if clips else '<p>선택된 후보가 없습니다.</p>'
    (args.analysis_dir / "reviewed.html").write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>근거 기반 에이전트 검토</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#f3f5f7;color:#17252d}article{background:white;padding:20px;margin:20px 0;border-radius:12px}video{width:100%}p{line-height:1.6}</style>
<h1>근거 기반 에이전트 검토</h1><p>관심도 후보 선정이며, 랠리 경계나 득점을 새로 검증한 결과는 아닙니다.</p>
<p><a href="review.html">전체 시간축·레이어 보기</a> · <a href="reviewed_selection.json">검토 결과 JSON</a></p>'''
        + media + ''.join(cards) + '</html>')
    print(f"selected {len(clips)} / {len(packets)} candidates; no new vision inference", flush=True)


if __name__ == "__main__":
    main()
