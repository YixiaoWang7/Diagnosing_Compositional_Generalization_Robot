"""Lazy RGB decoders with one bounded handle pool and no decoded-frame cache."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from robopolicy.datasets.config import VideoConfig


def require_decoder(config: VideoConfig) -> None:
    """Validate the explicitly selected backend without silently switching it."""
    try:
        if config.decoder == "opencv":
            import cv2  # noqa: F401
        else:
            from torchcodec.decoders import VideoDecoder  # noqa: F401
    except (ImportError, RuntimeError, OSError) as error:
        extra = "opencv" if config.decoder == "opencv" else "torchcodec"
        raise ImportError(
            f"Decoder {config.decoder!r} is unavailable. Install robopolicy[{extra}] "
            "and matching PyTorch/FFmpeg libraries; see docs/installation.md. "
            f"Original error: {error}"
        ) from error


class DecoderHandle:
    """Own a single native video decoder and produce RGB uint8 NCHW frames."""

    def __init__(self, path: Path, config: VideoConfig, expected_frames: int) -> None:
        """Open one decoder and verify the sidecar's total frame count."""
        self.config = config
        self.path = path
        self.frame_count = expected_frames
        self.native: Any = None
        if config.decoder == "opencv":
            import cv2

            self.native = cv2.VideoCapture(str(path))
            if not self.native.isOpened():
                self.close()
                raise RuntimeError(f"OpenCV could not open video {path}")
            actual = int(self.native.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            from torchcodec.decoders import VideoDecoder

            self.native = VideoDecoder(
                str(path), device="cpu", dimension_order="NCHW", num_ffmpeg_threads=config.threads
            )
            actual = len(self.native)
        if actual != expected_frames:
            self.close()
            raise ValueError(
                f"Video length mismatch at {path}: decoded={actual}, sidecar={expected_frames}"
            )

    def read(self, indices: list[int]) -> Tensor:
        """Decode arbitrary, possibly repeated frame indices into ``(N, 3, H, W)``."""
        if self.native is None:
            raise RuntimeError(f"Decoder is closed: {self.path}")
        if not indices:
            raise ValueError("A decode request must contain at least one frame")
        if min(indices) < 0 or max(indices) >= self.frame_count:
            raise ValueError("Requested video frame is outside the source bounds")
        if self.config.decoder == "torchcodec":
            frames: Tensor = self.native.get_frames_at(indices=indices).data
            return frames
        import cv2

        unique = sorted(set(indices))
        decoded: dict[int, Tensor] = {}
        for index in unique:
            position = int(self.native.get(cv2.CAP_PROP_POS_FRAMES))
            if position > index or index - position > 32:
                if not self.native.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise RuntimeError(f"Video seek failed: {self.path}, frame {index}")
                position = index
            for _ in range(index - position):
                if not self.native.grab():
                    raise RuntimeError(f"Video frame missing: {self.path}, frame {index}")
            ok, frame = self.native.read()
            if not ok:
                raise RuntimeError(f"Video decode failed: {self.path}, frame {index}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded[index] = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            # Shape: (3, H, W), with RGB order for both backends.
        return torch.stack([decoded[index] for index in indices])  # (N, 3, H, W)

    def close(self) -> None:
        """Release the native capture or the reference-counted TorchCodec handle."""
        if self.native is not None and self.config.decoder == "opencv":
            self.native.release()
        self.native = None


class DecoderPool:
    """LRU pool used only by one decode thread; cleanup follows thread shutdown."""

    def __init__(self, config: VideoConfig, counters: Any) -> None:
        """Initialize an empty pool and shared aggregate resource counters."""
        self.config = config
        self.counters = counters
        self.handles: OrderedDict[Path, DecoderHandle] = OrderedDict()

    def _count(self, change: int) -> None:
        """Update aggregate live and peak native handle counts under a lock."""
        with self.counters.get_lock():
            self.counters[0] += change
            self.counters[1] = max(self.counters[0], self.counters[1])

    def read(self, path: Path, indices: list[int], total_frames: int) -> Tensor:
        """Return RGB uint8 ``(N, 3, H, W)`` using at most max_handles decoders."""
        if path not in self.handles:
            if len(self.handles) >= self.config.max_handles:
                _, handle = self.handles.popitem(last=False)
                handle.close()
                self._count(-1)
            self.handles[path] = DecoderHandle(path, self.config, total_frames)
            self._count(1)
        self.handles.move_to_end(path)
        frames = self.handles[path].read(indices)
        if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Decoder must return RGB uint8 (N, 3, H, W): {path}")
        with self.counters.get_lock():
            self.counters[2] += 1
        return frames

    def close(self) -> None:
        """Close every handle on normal exit, early exit, or decoding failure."""
        for handle in self.handles.values():
            handle.close()
            self._count(-1)
        self.handles.clear()
