# highlight-generator

Finds the highlights in your own sports videos and cuts them into clips.

Most of a game is nothing happening, so the expensive model shouldn't look at all of it.
A cheap pose model runs over the whole video and flags the moments where a player does
something (arms up for a shot, a leg swung for a kick). SAM 3 then looks only at those
moments and, for basketball, decides whether the shot went in.

Everything runs on the GPU: frames are decoded with NVDEC and stay in GPU memory through
preprocessing and inference. There is no CPU fallback.

## Requirements

- An NVIDIA GPU with a driver that supports CUDA 12.8. Developed on an RTX 4060 Ti (8 GB)
  under WSL2. SAM 3 peaks at about 3.4 GB of VRAM.
- [uv](https://docs.astral.sh/uv/)

## Setup

```sh
uv sync

# SAM 3 weights are gated: request access at https://huggingface.co/facebook/sam3 first.
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf download facebook/sam3 sam3.pt --local-dir models
```

The YOLO11-pose weights are downloaded into `models/` on the first run.

## Usage

```sh
uv run highlight-detect path/to/video.mp4 --sport basketball   # or --sport football
uv run highlight-pose path/to/video.mp4                        # pose only
```

`highlight-detect` writes to `outputs/`:

| File | What it is |
| --- | --- |
| `<video>_highlight<N>_made.mp4`, `_miss.mp4` | Clips cut from the original, without re-encoding |
| `<video>_highlight<N>_<label>_overlay.mp4` | What both models saw: skeletons, hoop, ball path, rim zoom, verdict |
| `<video>_highlights.json` | Windows, pose peaks, shot labels and the ball track |
| `<video>_poses.npz` | Every person's box, track ID and 17 keypoints per sampled frame (reused on later runs) |

`highlight-pose` also writes `<video>_pose.mp4`, the skeletons on every sampled frame.
Run either command with `--help` for the options.

## How it works

1. **Pose, whole video.** YOLO11s-pose with ByteTrack runs on every 3rd frame at full
   1920 px width (smaller inputs miss distant players). A per-sport cue, measured in
   torso lengths so it doesn't depend on distance to the camera, marks active frames:
   - basketball `arms_up`: a wrist at least 0.3 torso lengths above the head
   - football `foot_up`: the ankles at least 0.6 torso lengths apart vertically

   Active frames close together become one window, padded 1.5 s before and 2 s after.
   About 5x realtime on the RTX 4060 Ti.
2. **SAM 3, candidate windows only (basketball).** The text prompt "basketball hoop"
   finds the rim once. "basketball" is then searched for on every 2nd frame in a
   644 px crop around the rim, and the ball's position is measured from the rim center
   in rim widths. A shot is **made** when the ball crosses rim height near the middle
   of the rim and keeps dropping through the net, a **miss** when it went above the rim
   but didn't, and **no shot** (no clip) when it never came near the rim.

## Limitations

- Assumes a fixed camera and one hoop in view.
- Football has no SAM 3 stage yet, so every candidate window becomes a clip.
- Clips are stream-copied, so one can start up to a keyframe interval (about 1 s on
  iPhone video) before its window.
- The thresholds were tuned on three short clips.

## Earlier code

`detector/` and `videos/` are an earlier (2019) experiment: a CNN classifier trained on
game-broadcast highlights. They have their own `requirements.txt` and don't share code
with the pipeline above.
