"""Bounded video streams and attention overlays aligned to processed images."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from corl2026_cg.instructions import ActiveInstructionProfile, annotate_instruction


class EpisodeVideos:
    """Open writers lazily and close them on both rollout success and failure."""

    def __init__(
        self,
        out_dir: Path,
        policy: Any,
        *,
        offset: int,
        max_videos: int,
        max_attention_videos: int,
        fps: int,
        profile: ActiveInstructionProfile | None = None,
    ) -> None:
        self.out_dir, self.policy, self.offset = out_dir, policy, offset
        self.max_videos, self.max_attention_videos, self.fps = max_videos, max_attention_videos, fps
        self.profile = profile
        self.writers: dict[tuple[int, str], Any] = {}
        self.last_batch: dict[str, torch.Tensor] | None = None
        self.attention_frames: dict[int, np.ndarray] = {}

    def _write(self, index: int, kind: str, pixels: np.ndarray) -> None:
        key = (index, kind)
        if key not in self.writers:
            import imageio.v2 as imageio

            directory = self.out_dir / kind
            directory.mkdir(parents=True, exist_ok=True)
            self.writers[key] = imageio.get_writer(
                directory / f"episode-{index:06d}.mp4",
                fps=self.fps,
                codec="libx264",
                macro_block_size=1,
            )
        self.writers[key].append_data(pixels)

    def __call__(
        self, index: int, observation: dict[str, np.ndarray], batch: dict[str, torch.Tensor] | None
    ) -> None:
        """Record RGB cameras and reuse attention until the next policy replan."""
        episode = self.offset + index
        if "images" not in observation:
            return
        instruction_id = int(observation["task_indices"][0]) if self.profile else None
        if episode < self.max_videos:
            pixels = np.concatenate(observation["images"].transpose(0, 2, 3, 1), axis=1)
            if self.profile is not None and instruction_id is not None:
                pixels = annotate_instruction(pixels, self.profile, instruction_id)
            self._write(episode, "videos", pixels)
        if episode >= self.max_attention_videos:
            return
        if batch is not None and batch is not self.last_batch:
            self.last_batch = batch
            with torch.no_grad():
                inputs = self.policy.packer(batch, device=self.policy.device)
                maps = self.policy.policy.attention_maps(inputs)
            weights = maps.get("query_attn")
            if weights is None:
                raise ValueError("The checkpoint fusion does not expose query attention maps")
            assert inputs.images is not None
            processing = self.policy.packer.config
            mean = torch.tensor(processing.image_mean).reshape(1, 3, 1, 1)
            std = torch.tensor(processing.image_std).reshape(1, 3, 1, 1)
            for sample in range(weights.shape[0]):
                images = (inputs.images[sample, -1].cpu() * std + mean).clamp(0, 1)
                panels = []
                for camera in range(images.shape[0]):
                    rgb = images[camera].permute(1, 2, 0).numpy()
                    camera_panels = [rgb]
                    for query in weights[sample, -1, camera]:
                        side = math.isqrt(query.numel())
                        if side * side != query.numel():
                            raise ValueError("Attention visualization requires a square patch grid")
                        heat = functional.interpolate(
                            query.cpu().reshape(1, 1, side, side),
                            size=rgb.shape[:2],
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0].numpy()
                        heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-8)
                        color = np.stack([heat, np.zeros_like(heat), 1 - heat], axis=-1)
                        camera_panels.append(0.6 * rgb + 0.4 * color)
                    panels.append(np.concatenate(camera_panels, axis=1))
                self.attention_frames[sample] = (255 * np.concatenate(panels, axis=0)).astype(
                    np.uint8
                )
        if index in self.attention_frames:
            pixels = self.attention_frames[index]
            if self.profile is not None and instruction_id is not None:
                pixels = annotate_instruction(pixels, self.profile, instruction_id)
            self._write(episode, "attention", pixels)

    def close(self) -> None:
        """Flush all streams."""
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()
