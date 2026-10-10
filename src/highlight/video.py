"""Video decoding on the GPU with NVDEC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import PyNvVideoCodec as nvc
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


class NvdecReader:
    """Decodes a video with NVDEC and yields every `stride`-th frame as a GPU tensor.

    Uses PyNvVideoCodec's demuxer + decoder directly: its SimpleDecoder duplicates and
    drops frames on iPhone HEVC files, which would shift every timestamp after it.
    """

    def __init__(self, path: str, gpu_id: int = 0):
        self.path = path
        self.gpu_id = gpu_id
        demuxer = nvc.CreateDemuxer(filename=path)
        self.info = VideoInfo(width=demuxer.Width(), height=demuxer.Height(), fps=demuxer.FrameRate())
        self.frames_decoded = 0

    def frames(self, stride: int = 1) -> Iterator[Frame]:
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
