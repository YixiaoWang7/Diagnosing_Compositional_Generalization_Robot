#!/usr/bin/env python

"""
Projector / pooling modules used by vision encoders.

Kept separate from `robopolicy.modules.vision` so `vision/` stays focused on image encoders.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax operation for extracting keypoints from feature maps."""

    def __init__(self, input_shape: tuple[int, int, int], num_kp: int | None = None) -> None:
        """Configure SpatialSoftmax and allocate its owned state.

        Args:
            input_shape: Per-sample feature map shape (C, H, W).
            num_kp: Num kp setting for this component.
        """
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        self.nets: nn.Conv2d | None
        self.pos_grid: Tensor
        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        grid_x, grid_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(grid_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(grid_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Spatial-softmax forward.

        Args:
            features (Tensor): Shape [B, C, H, W]. Feature map to pool.

        Returns:
            Tensor: Keypoints of shape [B, C_out, 2], where C_out = num_kp if provided,
            else C_out = C. The last dimension is (x, y) in normalized [-1, 1] coordinates.

        Details:
            1. (Optional) 1x1 conv reduces/expands channels to `num_kp`.
            2. Softmax over spatial positions.
            3. Expected position computed via sum of p(x,y) * [x,y].
        """
        if self.nets is not None:
            features = self.nets(features)

        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints
