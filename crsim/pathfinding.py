"""Flow-field pathfinding for ground troops.

Air troops fly in straight lines.  Ground troops must cross the river via
bridges.  We precompute flow fields so each entity lookup is O(1) per tick.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from crsim.constants import (
    ARENA_H,
    ARENA_W,
    BRIDGE_LEFT_COLS,
    BRIDGE_RIGHT_COLS,
    RIVER_ROW_HI,
    RIVER_ROW_LO,
)

# Passability grid (True = walkable).  Built once, shared by all games.
_PASSABLE: np.ndarray | None = None


def _build_passable() -> np.ndarray:
    """Build the static passability grid (18×32)."""
    grid = np.ones((ARENA_W, ARENA_H), dtype=bool)
    # River is impassable except at bridges
    for x in range(ARENA_W):
        for y in (RIVER_ROW_LO, RIVER_ROW_HI):
            is_bridge = (
                (BRIDGE_LEFT_COLS[0] <= x <= BRIDGE_LEFT_COLS[1])
                or (BRIDGE_RIGHT_COLS[0] <= x <= BRIDGE_RIGHT_COLS[1])
            )
            if not is_bridge:
                grid[x, y] = False
    return grid


def get_passable() -> np.ndarray:
    global _PASSABLE
    if _PASSABLE is None:
        _PASSABLE = _build_passable()
    return _PASSABLE


# 8-connected neighbours (dx, dy, cost)
_NEIGHBOURS_8 = [
    (-1, -1, math.sqrt(2)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2)),
]


def compute_flow_field(
    targets: list[tuple[int, int]],
    passable: np.ndarray | None = None,
) -> np.ndarray:
    """BFS from *targets* outward, returning a direction field.

    Returns
    -------
    flow : ndarray, shape (ARENA_W, ARENA_H, 2), dtype float32
        For each cell, a unit (dx, dy) vector pointing toward the nearest
        target.  Cells that are unreachable have (0, 0).
    """
    if passable is None:
        passable = get_passable()

    _inf = float("inf")
    cost = np.full((ARENA_W, ARENA_H), _inf, dtype=np.float64)
    flow = np.zeros((ARENA_W, ARENA_H, 2), dtype=np.float32)

    q: deque[tuple[int, int]] = deque()
    for tx, ty in targets:
        if 0 <= tx < ARENA_W and 0 <= ty < ARENA_H:
            cost[tx, ty] = 0.0
            q.append((tx, ty))

    # Dijkstra-like BFS (uniform cost for cardinal, √2 for diag)
    while q:
        cx, cy = q.popleft()
        for dx, dy, step_cost in _NEIGHBOURS_8:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < ARENA_W and 0 <= ny < ARENA_H and passable[nx, ny]:
                new_cost = cost[cx, cy] + step_cost
                if new_cost < cost[nx, ny]:
                    cost[nx, ny] = new_cost
                    q.append((nx, ny))

    # Build direction vectors: each cell points toward the neighbour with
    # lowest cost (i.e., toward the nearest target).
    for x in range(ARENA_W):
        for y in range(ARENA_H):
            if cost[x, y] >= _inf:
                continue
            best_cost = cost[x, y]
            best_dx, best_dy = 0.0, 0.0
            for dx, dy, _ in _NEIGHBOURS_8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < ARENA_W and 0 <= ny < ARENA_H:
                    if cost[nx, ny] < best_cost:
                        best_cost = cost[nx, ny]
                        best_dx, best_dy = float(dx), float(dy)
            # Normalize
            mag = math.sqrt(best_dx * best_dx + best_dy * best_dy)
            if mag > 0:
                flow[x, y, 0] = best_dx / mag
                flow[x, y, 1] = best_dy / mag

    return flow


class FlowFieldCache:
    """Caches flow fields for common target sets.

    In practice the targets rarely change (only when a building is built or
    destroyed), so we cache aggressively.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[tuple[int, int], ...], np.ndarray] = {}

    def get(self, targets: list[tuple[int, int]]) -> np.ndarray:
        key = tuple(sorted(targets))
        if key not in self._cache:
            self._cache[key] = compute_flow_field(targets)
        return self._cache[key]

    def invalidate(self) -> None:
        self._cache.clear()


def direction_to_target(
    src_x: float,
    src_y: float,
    tgt_x: float,
    tgt_y: float,
) -> tuple[float, float]:
    """Unit direction vector from src to tgt (for air / straight-line moves)."""
    dx = tgt_x - src_x
    dy = tgt_y - src_y
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)
