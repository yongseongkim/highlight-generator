"""NVDEC on CUDA; OpenCV decoding with explicit upload on Apple MPS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import torch


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class Frame:
    index: int  # display-order frame number, counting every decoded frame
    time_s: float  # presentation time relative to the first frame
    rgb: torch.Tensor  # (H, W, 3) uint8 RGB on the GPU


def resolve_device(device: str | None = None, gpu_id: int = 0) -> str:
    device = device or (f"cuda:{gpu_id}" if torch.cuda.is_available() else "mps")
    if device.startswith("cuda") and torch.cuda.is_available():
        return device
    if device == "mps" and torch.backends.mps.is_available():
        return device
    raise RuntimeError(f"Requested GPU {device!r} is unavailable; no CPU inference fallback")


def open_video(path: str, device: str) -> NvdecReader | OpenCvReader:
    if device.startswith("cuda"):
        return NvdecReader(path, gpu_id=torch.device(device).index or 0)
    return OpenCvReader(path, device)


class OpenCvReader:
    """Decode on the CPU and upload sampled RGB frames to MPS (not zero-copy NVDEC)."""

    def __init__(self, path: str, device: str = "mps"):
        self.path = path
        self.device = device
        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            self.info = VideoInfo(
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                capture.get(cv2.CAP_PROP_FPS),
            )
            if self.info.fps <= 0:
                raise RuntimeError(f"Video has no valid frame rate: {path}")
        finally:
            capture.release()
        self.frames_decoded = 0

    def frames(self, stride: int = 1) -> Iterator[Frame]:
        if stride < 1:
            raise ValueError("stride must be positive")
        capture = cv2.VideoCapture(self.path)
        self.frames_decoded = 0
        first_ms = None
        try:
            while capture.grab():
                index = self.frames_decoded
                self.frames_decoded += 1
                if index % stride:
                    continue
                ok, bgr = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot retrieve frame {index}: {self.path}")
                pts_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                if first_ms is None:
                    first_ms = pts_ms
                rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).to(self.device)
                yield Frame(index, (pts_ms - first_ms) / 1000, rgb)
        finally:
            capture.release()


class NvdecReader:
    """Decodes a video with NVDEC and yields every `stride`-th frame as a GPU tensor.

    Uses PyNvVideoCodec's demuxer + decoder directly: its SimpleDecoder duplicates and
    drops frames on iPhone HEVC files, which would shift every timestamp after it.
    """

    def __init__(self, path: str, gpu_id: int = 0):
        import PyNvVideoCodec as nvc

        self.path = path
        self.gpu_id = gpu_id
        demuxer = nvc.CreateDemuxer(filename=path)
        self.info = VideoInfo(width=demuxer.Width(), height=demuxer.Height(), fps=demuxer.FrameRate())
        self.frames_decoded = 0

    def frames(self, stride: int = 1) -> Iterator[Frame]:
        import PyNvVideoCodec as nvc

        demuxer = nvc.CreateDemuxer(filename=self.path)
        decoder = nvc.CreateDecoder(
            gpuid=self.gpu_id,
            codec=demuxer.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.RGB,
        )
        time_base = demuxer.GetTimebaseNum() / demuxer.GetTimebaseDen()
        first_pts = None
        self.frames_decoded = 0
        # Skipped frames still have to be decoded: later frames are predicted from them.
        for packet in demuxer:
            for decoded in decoder.Decode(packet):
                pts = decoded.getPTS()
                if first_pts is None:
                    first_pts = pts
                index = self.frames_decoded
                self.frames_decoded += 1
                if index % stride == 0:
                    # Copy out of the decoder's surface pool, which is reused for later frames
                    # and freed with the decoder. A GPU-to-GPU copy, ~0.1 ms at 1080p.
                    rgb = torch.from_dlpack(decoded).clone()
                    yield Frame(index, (pts - first_pts) * time_base, rgb)
