"""Ring replay buffer for storing self-play experience.

Each entry stores:
  - spatial features  (C, H, W) float32
  - scalar features   (S,)      float32
  - MCTS policy       (A,)      float32
  - game outcome      scalar    float32  (+1 win, -1 loss, 0 draw)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from crsim.constants import (
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
)


@dataclass
class ReplayEntry:
    spatial: np.ndarray   # (C, H, W)
    scalar: np.ndarray    # (S,)
    policy: np.ndarray    # (A,)
    value: float          # terminal outcome from this player's perspective


class ReplayBuffer:
    """Thread-safe ring buffer for training data."""

    def __init__(self, capacity: int = 500_000) -> None:
        self.capacity = capacity

        # Pre-allocate arrays for zero-copy storage
        self._spatial = np.zeros(
            (capacity, SPATIAL_CHANNELS, ARENA_H, ARENA_W), dtype=np.float32
        )
        self._scalar = np.zeros(
            (capacity, SCALAR_FEATURES), dtype=np.float32
        )
        self._policy = np.zeros(
            (capacity, ACTION_SPACE_SIZE), dtype=np.float32
        )
        self._value = np.zeros(capacity, dtype=np.float32)

        self._size: int = 0
        self._write_idx: int = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._size

    def push(self, entry: ReplayEntry) -> None:
        """Add a single experience entry."""
        with self._lock:
            idx = self._write_idx
            self._spatial[idx] = entry.spatial
            self._scalar[idx] = entry.scalar
            self._policy[idx] = entry.policy
            self._value[idx] = entry.value
            self._write_idx = (idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def push_batch(
        self,
        spatials: np.ndarray,
        scalars: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Add a batch of entries efficiently."""
        bsz = spatials.shape[0]
        with self._lock:
            for i in range(bsz):
                idx = self._write_idx
                self._spatial[idx] = spatials[i]
                self._scalar[idx] = scalars[i]
                self._policy[idx] = policies[i]
                self._value[idx] = values[i]
                self._write_idx = (idx + 1) % self.capacity
            self._size = min(self._size + bsz, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch.

        Returns
        -------
        spatial : (B, C, H, W)
        scalar  : (B, S)
        policy  : (B, A)
        value   : (B,)
        """
        with self._lock:
            indices = np.random.randint(0, self._size, size=batch_size)
            return (
                self._spatial[indices].copy(),
                self._scalar[indices].copy(),
                self._policy[indices].copy(),
                self._value[indices].copy(),
            )
