# Agent guide

Keep this file concise and English. Keep user-facing documentation in Korean in
`README.md`. Read relevant modules or README sections; prefer benchmark summaries
over generated logs and media.

## Commands

- Install: `uv sync --frozen`
- Behavior tests (no model inference): `uv run python -m unittest discover -s tests -v`
- Whitespace: `git diff --check`
- CLI details: `uv run highlight-plays --help` (also pose, detect, court, patterns, fusion).
- Docs-only edits: check links, fences, and command flags; skip model runs.

## Code map

Python >=3.10; uv, PyTorch, Ultralytics, OpenCV. Current code is in `src/highlight/`.
Legacy `detector/` and `videos/` have separate dependencies.

| Files under `src/highlight/` | Responsibility |
| --- | --- |
| `video.py`, `pose.py`, `extract_pose.py` | CUDA/MPS decoding, YOLO poses, CPU ByteTrack, caches |
| `court.py` | Line diagnostics, explicit court polygon and ground-level net |
| `motion.py` | Strong motion seeds and bounded context expansion |
| `game_units.py`, `plays.py` | Event-defined plays, motion mode, ranking, exports |
| `pattern_signals.py`, `patterns.py` | Temporal action rules, cached SAM support, optional refinement |
| `badminton.py`, `badminton_events.py` | Sparse pose candidates and budgeted detail keyframes |
| `detect.py`, `signals.py`, `candidates.py`, `shots.py` | Sport dispatch, original cues, inferred basket outcomes |
| `sam.py`, `objects.py`, `viz.py` | Detail inference and visualization |
| `fusion.py`, `fusion_signals.py`, `fusion_review.py` | Multi-rate overlays, evidence packets, validated agent reviews |

`scripts/benchmark_pose.py` measures fresh inference; `benchmark_patterns.py`
measures cached extraction (`--motion-seeds` for contexts). `monitor_mac.py` and
`plot_mac_benchmark.py` record/export resources. Behavior tests are in `tests/`.
`benchmark_fusion.py` measures dense CV, fresh object/SAM calls, and overlay export.

## Invariants

- Filter player feet to the supplied court with zero margin before motion/action
  analysis and SAM support. OpenCV lines do not identify the target court.
- Motion cuts are estimates: defaults are 12 s before/8 s after; merged contexts
  stay bounded/disjoint. Tracking loss or a lull never verifies service/landing.
- Event mode preserves whole units; skip rather than trim to fit budgets. Keep
  unknown boundaries and verification separate from interest scores.
- Compact/net actions need no overhead. A pose, net touch, or sparse SAM sample
  cannot confirm a hairpin, hit, winner, or complete rally. Football goals need
  external events; the basketball rim rule is inferred evidence.
- `highlight-plays` uses cached poses/optional SAM only. Fresh SAM is bounded in
  detect/pattern refinement; require its gated checkpoint before expensive work.
- Pose cache keys include source/checkpoint identity, stride, width, confidence.
  Associate saved SAM by source frame/geometry/crop, not cross-cache IDs.
- Preserve CUDA/NVDEC and MPS/OpenCV paths; never silently use CPU inference.
- Fusion keeps native object labels, source frame/time, sampling gaps and bounded
  display ages. Flying objects need more than ground-ROI membership. LLM reviews
  cite existing evidence IDs; interest judgments never add boundary verification.
- Label fresh inference, cached analysis, and export costs separately. Parent RSS
  excludes FFmpeg and does not measure total GPU memory; Mac counters include other apps; thermal
  pressure is not temperature. Speed comparisons do not establish accuracy.

## Changes

Use feature branches and PRs; do not push directly to `master` unless requested.
Keep models, input videos, caches, generated media, and secrets out of commits;
track compact benchmark summaries instead. Reuse existing caches for iteration.
Test changed behavior at the appropriate scope; docs edits need no new tests.
