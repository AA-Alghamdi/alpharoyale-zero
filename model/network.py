"""ResNet + policy/value heads for ClashRoyale-Zero.

Architecture
------------
Spatial input  → 3×3 conv (256 filters) → 20 SE-ResBlocks
Scalar input   → Linear(256) → ReLU → Linear(256)
Merge          → concat channel-wise (broadcast scalars to spatial grid)
Policy head    → 1×1 conv(2) → flatten → Linear(2305)
Value head     → 1×1 conv(1) → flatten → Linear(256) → ReLU → Linear(1) → tanh
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f_nn  # noqa: N812

from crsim.constants import (
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))  # (b, c)
        s = f_nn.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (b, c)
        return x * s.view(b, c, 1, 1)


class ResBlock(nn.Module):
    """Pre-activation residual block with optional SE."""

    def __init__(self, channels: int, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = f_nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return f_nn.relu(out + residual)


class CRZeroNet(nn.Module):
    """Full ClashRoyale-Zero network: spatial encoder + scalar encoder → policy + value."""

    def __init__(
        self,
        spatial_channels: int = SPATIAL_CHANNELS,
        scalar_dim: int = SCALAR_FEATURES,
        n_res_blocks: int = 20,
        n_filters: int = 256,
        action_size: int = ACTION_SPACE_SIZE,
        use_se: bool = True,
    ) -> None:
        super().__init__()

        self.action_size = action_size

        # --- Spatial encoder ---
        self.spatial_input = nn.Sequential(
            nn.Conv2d(spatial_channels, n_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(n_filters, use_se=use_se) for _ in range(n_res_blocks)]
        )

        # --- Scalar encoder ---
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, n_filters),
            nn.ReLU(),
            nn.Linear(n_filters, n_filters),
            nn.ReLU(),
        )

        # Merge: add broadcast scalar features to spatial features
        # (scalar gets tiled to H×W, then added channel-wise)

        # --- Policy head ---
        self.policy_conv = nn.Sequential(
            nn.Conv2d(n_filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(32 * ARENA_H * ARENA_W, action_size)

        # --- Value head ---
        self.value_conv = nn.Sequential(
            nn.Conv2d(n_filters, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(4 * ARENA_H * ARENA_W, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        spatial : (B, C, H, W) float — spatial feature planes
        scalar  : (B, S) float — scalar features
        action_mask : (B, A) bool, optional — True = valid action

        Returns
        -------
        policy_logits : (B, A) float — masked log-probabilities
        value         : (B,) float — win probability in [-1, 1]
        """
        bsz = spatial.shape[0]

        # Spatial tower
        s = self.spatial_input(spatial)      # (bsz, F, H, W)
        s = self.res_blocks(s)               # (bsz, F, H, W)

        # Scalar tower → broadcast-add
        sc = self.scalar_encoder(scalar)     # (bsz, F)
        s = s + sc.view(bsz, -1, 1, 1)      # broadcast add

        # Policy head
        p = self.policy_conv(s)              # (bsz, 32, H, W)
        p = p.view(bsz, -1)                 # (bsz, 32*H*W)
        p = self.policy_fc(p)               # (bsz, A)

        if action_mask is not None:
            p = p.masked_fill(~action_mask, -1e9)

        # Value head
        v = self.value_conv(s)               # (bsz, 4, H, W)
        v = v.view(bsz, -1)                 # (bsz, 4*H*W)
        v = self.value_fc(v).squeeze(-1)    # (bsz,)

        return p, v

    def predict(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper: returns policy probabilities (softmax) and value."""
        with torch.no_grad():
            logits, value = self.forward(spatial, scalar, action_mask)
            policy = f_nn.softmax(logits, dim=-1)
        return policy, value
