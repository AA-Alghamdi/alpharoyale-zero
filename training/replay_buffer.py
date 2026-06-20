"""Ring replay buffer for storing self-play experience.

Each entry stores:
  - spatial features  (C, H, W) float32
  - scalar features   (S,)      float32
  - MCTS policy       (A,)      float32
  - game outcome      scalar    float32  (+1 win, -1 loss, 0 draw)
  - entity features   (E, D)    float32  (optional, for EntityTransformer)
  - entity mask       (E,)      bool     (optional)
  - auxiliary targets: crown, tower HP, game length (optional, for KataGo heads)
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

ENTITY_SLOTS = 64
ENTITY_DIM = 40


@dataclass
class ReplayEntry:
    spatial: np.ndarray   # (C, H, W)
    scalar: np.ndarray    # (S,)
    policy: np.ndarray    # (A,)
    value: float          # terminal outcome from this player's perspective
    # Entity features for Transformer
    entity_features: np.ndarray | None = None  # (E, D)
    entity_mask: np.ndarray | None = None      # (E,)
    # Auxiliary targets
    crown_target: int = 3  # 0-6 index (3 = no crown diff)
    tower_hp_target: np.ndarray | None = None  # (6,)
    game_length_target: float = 0.5
    # Priority for PER
    priority: float = 1.0


class ReplayBuffer:
    """Thread-safe ring buffer with entity features and aux targets."""

    # Large per-position tensors are stored in float16 to halve memory. A single
    # spatial observation is SPATIAL_CHANNELS * ARENA_H * ARENA_W floats, which
    # for 125 cards is 254*32*18 = ~146K values. At 500K positions that is ~292GB
    # in float32 — far beyond any machine — so the default capacity is modest and
    # the per-position memory cost is logged. Reduce SPATIAL_CHANNELS (e.g. with
    # semantic unit planes instead of one plane per card type) for larger buffers.
    def __init__(
        self,
        capacity: int = 50_000,
        max_size: int | None = None,
    ) -> None:
        self.capacity = max_size or capacity

        # Per-position memory estimate (float16 for the big arrays).
        bytes_per_pos = (
            SPATIAL_CHANNELS * ARENA_H * ARENA_W * 2          # spatial f16
            + ENTITY_SLOTS * ENTITY_DIM * 2                    # entity f16
            + SCALAR_FEATURES * 4 + ACTION_SPACE_SIZE * 4      # scalar/policy f32
        )
        est_gb = bytes_per_pos * self.capacity / 1e9
        logger.info(
            "ReplayBuffer capacity=%d, ~%.1f KB/position, ~%.1f GB total",
            self.capacity, bytes_per_pos / 1024, est_gb,
        )

        # Arrays are allocated lazily on first push so that constructing a buffer
        # never blocks/OOMs and tests can use tiny capacities cheaply.
        self._allocated = False
        self._spatial: np.ndarray | None = None
        self._scalar: np.ndarray | None = None
        self._policy: np.ndarray | None = None
        self._value: np.ndarray | None = None
        self._entity_features: np.ndarray | None = None
        self._entity_mask: np.ndarray | None = None
        self._crown_target: np.ndarray | None = None
        self._tower_hp_target: np.ndarray | None = None
        self._game_length_target: np.ndarray | None = None
        self._priority: np.ndarray | None = None

        self._size: int = 0
        self._write_idx: int = 0
        self._lock = threading.Lock()

    def _ensure_allocated(self) -> None:
        if self._allocated:
            return
        cap = self.capacity
        self._spatial = np.zeros(
            (cap, SPATIAL_CHANNELS, ARENA_H, ARENA_W), dtype=np.float16
        )
        self._scalar = np.zeros((cap, SCALAR_FEATURES), dtype=np.float32)
        self._policy = np.zeros((cap, ACTION_SPACE_SIZE), dtype=np.float32)
        self._value = np.zeros(cap, dtype=np.float32)
        self._entity_features = np.zeros(
            (cap, ENTITY_SLOTS, ENTITY_DIM), dtype=np.float16
        )
        self._entity_mask = np.zeros((cap, ENTITY_SLOTS), dtype=bool)
        self._crown_target = np.full(cap, 3, dtype=np.int64)
        self._tower_hp_target = np.zeros((cap, 6), dtype=np.float32)
        self._game_length_target = np.full(cap, 0.5, dtype=np.float32)
        self._priority = np.ones(cap, dtype=np.float32)
        self._allocated = True

    def __len__(self) -> int:
        return self._size

    def push(self, entry: ReplayEntry) -> None:
        """Add a single experience entry."""
        with self._lock:
            self._ensure_allocated()
            idx = self._write_idx
            self._spatial[idx] = entry.spatial
            self._scalar[idx] = entry.scalar
            self._policy[idx] = entry.policy
            self._value[idx] = entry.value

            if entry.entity_features is not None:
                self._entity_features[idx] = entry.entity_features
            if entry.entity_mask is not None:
                self._entity_mask[idx] = entry.entity_mask

            self._crown_target[idx] = entry.crown_target
            if entry.tower_hp_target is not None:
                self._tower_hp_target[idx] = entry.tower_hp_target
            self._game_length_target[idx] = entry.game_length_target
            self._priority[idx] = entry.priority

            self._write_idx = (idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def push_batch(
        self,
        spatials: np.ndarray,
        scalars: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Add a batch of entries efficiently (legacy API)."""
        bsz = spatials.shape[0]
        with self._lock:
            self._ensure_allocated()
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
        """Sample a random mini-batch (legacy API).

        Returns (spatial, scalar, policy, value).
        """
        with self._lock:
            indices = np.random.randint(0, self._size, size=batch_size)
            return (
                self._spatial[indices].astype(np.float32),
                self._scalar[indices].copy(),
                self._policy[indices].copy(),
                self._value[indices].copy(),
            )

    def sample_full(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample with all fields including entity features and aux targets.

        Returns dict with keys: spatial, scalar, policy, value,
        entity_features, entity_mask, crown_target, tower_hp_target,
        game_length_target.
        """
        with self._lock:
            indices = np.random.randint(0, self._size, size=batch_size)
            return {
                "spatial": self._spatial[indices].astype(np.float32),
                "scalar": self._scalar[indices].copy(),
                "policy": self._policy[indices].copy(),
                "value": self._value[indices].copy(),
                "entity_features": self._entity_features[indices].astype(np.float32),
                "entity_mask": self._entity_mask[indices].copy(),
                "crown_target": self._crown_target[indices].copy(),
                "tower_hp_target": self._tower_hp_target[indices].copy(),
                "game_length_target": self._game_length_target[indices].copy(),
            }

    def sample_prioritized(
        self, batch_size: int, alpha: float = 0.6,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Prioritized experience replay sampling.

        Returns (batch_dict, indices, weights) for importance sampling.
        """
        with self._lock:
            priorities = self._priority[:self._size] ** alpha
            probs = priorities / priorities.sum()
            indices = np.random.choice(self._size, size=batch_size, p=probs)
            # Importance sampling weights
            weights = (self._size * probs[indices]) ** (-0.4)
            weights /= weights.max()
            batch = {
                "spatial": self._spatial[indices].astype(np.float32),
                "scalar": self._scalar[indices].copy(),
                "policy": self._policy[indices].copy(),
                "value": self._value[indices].copy(),
                "entity_features": self._entity_features[indices].astype(np.float32),
                "entity_mask": self._entity_mask[indices].copy(),
                "crown_target": self._crown_target[indices].copy(),
                "tower_hp_target": self._tower_hp_target[indices].copy(),
                "game_length_target": self._game_length_target[indices].copy(),
            }
            return batch, indices, weights

    def update_priorities(
        self, indices: np.ndarray, priorities: np.ndarray,
    ) -> None:
        """Update priorities for PER."""
        with self._lock:
            self._priority[indices] = priorities
