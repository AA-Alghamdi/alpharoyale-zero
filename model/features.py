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
            # Towers go into channels 2*NUM_CARD_TYPES and 2*NUM_CARD_TYPES+1
            chan = 2 * NUM_CARD_TYPES if is_friendly else 2 * NUM_CARD_TYPES + 1
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
                    spatial[2 * NUM_CARD_TYPES + 2, vy, x] = 1.0

    # Channel 2*N+3: static map (river = 1, bridges = 0.5)
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
            spatial[2 * NUM_CARD_TYPES + 3, vy, x] = 0.5 if is_bridge else 1.0

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


# ---------------------------------------------------------------------------
# Entity feature extraction (for EntityTransformer)
# ---------------------------------------------------------------------------
ENTITY_FEATURE_DIM = 40  # must match model/transformer_net.py
MAX_ENTITY_SLOTS = 64  # max entities we encode

# Feature layout per entity (40-d):
#   [0]      type_id (int, used for type embedding lookup)
#   [1]      owner (0=friendly, 1=enemy)
#   [2:4]    position (x/W, y/H) normalized
#   [4]      hp / max_hp
#   [5]      max_hp / 5000 (absolute scale)
#   [6]      dps / 500
#   [7]      speed / 4.0
#   [8]      attack_range / 10.0
#   [9]      is_flying
#   [10]     is_splash
#   [11]     is_building
#   [12]     is_tower
#   [13]     is_king_tower
#   [14]     target_mode (one-hot 0-3) → [14:18]
#   [18]     attack_timer / attack_interval (readiness)
#   [19]     building_timer / 70.0
#   [20]     elixir_cost / 10.0
#   [21:23]  velocity direction estimate (dx, dy) — from target
#   [23:40]  reserved / zero-padded


def extract_entity_features(
    game: CRGame,
    player: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-entity feature vectors for the EntityTransformer.

    Returns
    -------
    entity_features : (MAX_ENTITY_SLOTS, ENTITY_FEATURE_DIM) float32
    entity_mask : (MAX_ENTITY_SLOTS,) bool — True for real entities
    """
    from crsim.cards import CARD_DEFS
    from crsim.pathfinding import direction_to_target

    flip = player == 1
    features = np.zeros((MAX_ENTITY_SLOTS, ENTITY_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(MAX_ENTITY_SLOTS, dtype=bool)

    alive_entities = [e for e in game.entities if e.alive]
    n = min(len(alive_entities), MAX_ENTITY_SLOTS)

    for i in range(n):
        e = alive_entities[i]
        is_friendly = e.owner == player
        f = features[i]

        # Type ID (for embedding lookup)
        f[0] = float(e.card_type)

        # Owner
        f[1] = 0.0 if is_friendly else 1.0

        # Position (flipped if player 1)
        ex, ey = e.x, e.y
        if flip:
            ey = (ARENA_H - 1) - ey
        f[2] = ex / ARENA_W
        f[3] = ey / ARENA_H

        # HP
        f[4] = e.hp / max(e.max_hp, 1.0)
        f[5] = e.max_hp / 5000.0

        # Combat stats
        f[6] = e.dps / 500.0
        f[7] = e.speed / 4.0
        f[8] = e.attack_range / 10.0

        # Flags
        f[9] = 1.0 if e.is_flying else 0.0
        f[10] = 1.0 if e.is_splash else 0.0
        f[11] = 1.0 if e.is_building else 0.0
        f[12] = 1.0 if e.is_tower else 0.0
        f[13] = 1.0 if e.is_king_tower else 0.0

        # Target mode (one-hot)
        tm = int(e.target_mode)
        if 0 <= tm <= 3:
            f[14 + tm] = 1.0

        # Attack readiness
        if e.attack_interval > 0:
            f[18] = e.attack_timer / e.attack_interval
        else:
            f[18] = 0.0

        # Building timer
        f[19] = e.building_timer / 70.0 if e.is_building else 0.0

        # Elixir cost (from card def if available)
        if e.card_type in CARD_DEFS:
            f[20] = CARD_DEFS[e.card_type].cost / 10.0

        # Velocity estimate (direction toward target)
        target = game._get_entity(e.target_eid) if e.target_eid >= 0 else None
        if target is not None and e.speed > 0:
            dx, dy = direction_to_target(e.x, e.y, target.x, target.y)
            if flip:
                dy = -dy
            f[21] = dx
            f[22] = dy

        mask[i] = True

    return features, mask


# ---------------------------------------------------------------------------
# Auxiliary target extraction (real values, not approximations)
# ---------------------------------------------------------------------------

def extract_auxiliary_targets(
    game: CRGame,
    player: int,
) -> dict[str, np.ndarray]:
    """Extract real auxiliary targets from game state for KataGo-style training.

    Returns dict with:
      crown_diff: int in [-3, 3] → index in [0, 6]
      tower_hp: (6,) float — normalized HP of all 6 towers from player's perspective
      game_length_remaining: float in [0, 1]
    """
    our_crowns = game._count_crowns(player)
    their_crowns = game._count_crowns(1 - player)
    crown_diff = our_crowns - their_crowns  # [-3, 3]
    crown_idx = crown_diff + 3  # [0, 6]

    # Tower HP: [our_king, our_L_princess, our_R_princess,
    #            their_king, their_L_princess, their_R_princess]
    tower_hp = np.zeros(6, dtype=np.float32)
    for side_idx, p in enumerate([player, 1 - player]):
        offset = side_idx * 3
        kt = game.king_towers[p]
        tower_hp[offset] = (kt.hp / kt.max_hp) if (kt and kt.alive) else 0.0
        pts = game.princess_towers[p]
        for i in range(2):
            if i < len(pts):
                tower_hp[offset + 1 + i] = (
                    pts[i].hp / pts[i].max_hp if pts[i].alive else 0.0
                )

    # Game length remaining
    game_length_remaining = 1.0 - (game.tick_count / TOTAL_MAX_TICKS)
    game_length_remaining = max(0.0, min(1.0, game_length_remaining))

    return {
        "crown_target": np.array(crown_idx, dtype=np.int64),
        "tower_hp_target": tower_hp,
        "game_length_target": np.array(game_length_remaining, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Full state encoding (upgraded)
# ---------------------------------------------------------------------------

def encode_state_full(
    game: CRGame,
    player: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Full state encoding including entity features and auxiliary targets.

    Returns
    -------
    spatial : (SPATIAL_CHANNELS, ARENA_H, ARENA_W) float32
    scalar : (SCALAR_FEATURES,) float32
    entity_features : (MAX_ENTITY_SLOTS, ENTITY_FEATURE_DIM) float32
    entity_mask : (MAX_ENTITY_SLOTS,) bool
    aux_targets : dict with crown_target, tower_hp_target, game_length_target
    """
    spatial, scalar = encode_state(game, player)
    entity_feats, entity_mask = extract_entity_features(game, player)
    aux_targets = extract_auxiliary_targets(game, player)
    return spatial, scalar, entity_feats, entity_mask, aux_targets


def encode_batch(
    games: list[CRGame],
    players: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a batch of game states (legacy API — spatial + scalar only).

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


def encode_batch_full(
    games: list[CRGame],
    players: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode a batch with entity features.

    Returns
    -------
    spatial : (B, SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    scalar : (B, SCALAR_FEATURES)
    entity_features : (B, MAX_ENTITY_SLOTS, ENTITY_FEATURE_DIM)
    entity_mask : (B, MAX_ENTITY_SLOTS) bool
    """
    batch = len(games)
    spatials = np.zeros((batch, SPATIAL_CHANNELS, ARENA_H, ARENA_W), dtype=np.float32)
    scalars = np.zeros((batch, SCALAR_FEATURES), dtype=np.float32)
    ent_feats = np.zeros((batch, MAX_ENTITY_SLOTS, ENTITY_FEATURE_DIM), dtype=np.float32)
    ent_masks = np.zeros((batch, MAX_ENTITY_SLOTS), dtype=bool)

    for i, (g, p) in enumerate(zip(games, players)):
        spatials[i], scalars[i] = encode_state(g, p)
        ent_feats[i], ent_masks[i] = extract_entity_features(g, p)

    return spatials, scalars, ent_feats, ent_masks
