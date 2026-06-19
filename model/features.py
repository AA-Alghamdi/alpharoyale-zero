"""State encoding: convert a CRGame into tensors for the neural network.

Spatial tensor : (C, H, W) = (44, 32, 18)   — channel-first for PyTorch Conv2d
Scalar tensor  : (116,)
"""

from __future__ import annotations

import math

import numpy as np

from crsim.constants import (
    ARENA_H,
    ARENA_W,
    KING_TOWER_HP,
    MAX_ELIXIR,
    NUM_CARD_TYPES,
    NUM_HAND_SLOTS,
    PRINCESS_TOWER_HP,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
    TOTAL_MAX_TICKS,
)
from crsim.game import CRGame

# Gaussian splat sigma for unit density channels
_SIGMA = 0.5
_GAUSSIAN_RANGE = 2  # only splat within ±2 tiles


def _gaussian(dx: float, dy: float) -> float:
    return math.exp(-(dx * dx + dy * dy) / (2 * _SIGMA * _SIGMA))


def encode_state(
    game: CRGame,
    player: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the game from `player`'s perspective.

    Returns
    -------
    spatial : ndarray, shape (SPATIAL_CHANNELS, ARENA_H, ARENA_W) float32
    scalar  : ndarray, shape (SCALAR_FEATURES,) float32
    """
    spatial = np.zeros((SPATIAL_CHANNELS, ARENA_H, ARENA_W), dtype=np.float32)
    scalar = np.zeros(SCALAR_FEATURES, dtype=np.float32)

    flip = player == 1  # flip board so current player is always at bottom

    # ------------------------------------------------------------------
    # Spatial channels
    # ------------------------------------------------------------------
    for entity in game.entities:
        if not entity.alive:
            continue

        # Determine channel offset
        is_friendly = entity.owner == player
        if entity.is_tower:
            # Towers go into channels 40–41
            chan = 40 if is_friendly else 41
            ex, ey = entity.x, entity.y
            if flip:
                ey = (ARENA_H - 1) - ey
            ix, iy = int(round(ex)), int(round(ey))
            ix = max(0, min(ARENA_W - 1, ix))
            iy = max(0, min(ARENA_H - 1, iy))
            hp_norm = entity.hp / entity.max_hp if entity.max_hp > 0 else 0
            spatial[chan, iy, ix] = max(spatial[chan, iy, ix], hp_norm)
            continue

        # Unit density channels
        ct = entity.card_type
        if ct < 0 or ct >= NUM_CARD_TYPES:
            continue
        chan_base = 0 if is_friendly else NUM_CARD_TYPES
        chan = chan_base + int(ct)

        ex, ey = entity.x, entity.y
        if flip:
            ey = (ARENA_H - 1) - ey

        hp_norm = entity.hp / entity.max_hp if entity.max_hp > 0 else 0

        # Gaussian splat
        cx, cy = int(round(ex)), int(round(ey))
        for dx in range(-_GAUSSIAN_RANGE, _GAUSSIAN_RANGE + 1):
            for dy in range(-_GAUSSIAN_RANGE, _GAUSSIAN_RANGE + 1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < ARENA_W and 0 <= gy < ARENA_H:
                    w = _gaussian(ex - gx, ey - gy)
                    spatial[chan, gy, gx] += hp_norm * w

    # Channel 42: valid placement mask
    mask = game.get_valid_actions_mask(player)
    for slot in range(NUM_HAND_SLOTS):
        for x in range(ARENA_W):
            for y in range(ARENA_H):
                action_id = slot * ARENA_W * ARENA_H + x * ARENA_H + y
                if mask[action_id]:
                    vy = (ARENA_H - 1 - y) if flip else y
                    spatial[42, vy, x] = 1.0

    # Channel 43: static map (river = 1, bridges = 0.5)
    from crsim.constants import (
        BRIDGE_LEFT_COLS,
        BRIDGE_RIGHT_COLS,
        RIVER_ROW_HI,
        RIVER_ROW_LO,
    )
    for y in (RIVER_ROW_LO, RIVER_ROW_HI):
        vy = (ARENA_H - 1 - y) if flip else y
        for x in range(ARENA_W):
            is_bridge = (
                BRIDGE_LEFT_COLS[0] <= x <= BRIDGE_LEFT_COLS[1]
                or BRIDGE_RIGHT_COLS[0] <= x <= BRIDGE_RIGHT_COLS[1]
            )
            spatial[43, vy, x] = 0.5 if is_bridge else 1.0

    # ------------------------------------------------------------------
    # Scalar features
    # ------------------------------------------------------------------
    ps = game.players[player]
    idx = 0

    # Elixir
    scalar[idx] = ps.elixir / MAX_ELIXIR
    idx += 1

    # Elixir regen rate (phase encoding)
    scalar[idx] = float(game.phase) / 3.0
    idx += 1

    # Cards in hand: 4 one-hot vectors of length NUM_CARD_TYPES
    for slot in range(NUM_HAND_SLOTS):
        card_idx = ps.hand[slot]
        card_type = ps.deck[card_idx]
        scalar[idx + int(card_type)] = 1.0
        idx += NUM_CARD_TYPES

    # Next card
    next_ct = ps.deck[ps.next_card_idx]
    scalar[idx + int(next_ct)] = 1.0
    idx += NUM_CARD_TYPES

    # Time remaining
    time_frac = game.tick_count / TOTAL_MAX_TICKS
    scalar[idx] = time_frac
    idx += 1

    # Tower status: alive/dead — 3 per side (king + 2 princess) × 2 sides = 6
    for p in (player, 1 - player):
        kt = game.king_towers[p]
        scalar[idx] = 1.0 if (kt is not None and kt.alive) else 0.0
        idx += 1
        pts = game.princess_towers[p]
        for i in range(2):
            if i < len(pts):
                scalar[idx] = 1.0 if pts[i].alive else 0.0
            idx += 1

    # Tower HP (normalized) — same layout: 3 per side × 2 = 6
    for p in (player, 1 - player):
        kt = game.king_towers[p]
        scalar[idx] = (kt.hp / KING_TOWER_HP) if (kt and kt.alive) else 0.0
        idx += 1
        pts = game.princess_towers[p]
        for i in range(2):
            if i < len(pts):
                scalar[idx] = (pts[i].hp / PRINCESS_TOWER_HP) if pts[i].alive else 0.0
            idx += 1

    # Score difference (crowns)
    our_crowns = game._count_crowns(player)
    their_crowns = game._count_crowns(1 - player)
    scalar[idx] = (our_crowns - their_crowns) / 3.0
    idx += 1

    return spatial, scalar


def encode_batch(
    games: list[CRGame],
    players: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a batch of game states.

    Returns
    -------
    spatial : ndarray, shape (B, SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    scalar  : ndarray, shape (B, SCALAR_FEATURES)
    """
    batch = len(games)
    spatials = np.zeros((batch, SPATIAL_CHANNELS, ARENA_H, ARENA_W), dtype=np.float32)
    scalars = np.zeros((batch, SCALAR_FEATURES), dtype=np.float32)

    for i, (g, p) in enumerate(zip(games, players)):
        spatials[i], scalars[i] = encode_state(g, p)

    return spatials, scalars
