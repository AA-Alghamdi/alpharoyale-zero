"""State-encoding contract tests for the spatial + scalar network input.

``encode_state`` turns a :class:`CRGame` into the planes/vector the policy-value
ResNet consumes. The agent only ever sees this projection, so its invariants are
part of the RL interface:

  * the tensors have exactly the advertised shapes/dtype and never contain NaNs;
  * the board is drawn from the acting player's perspective — *their* units are
    always "friendly" and always at the bottom, regardless of which seat they sit
    in (the search would be unlearnable otherwise);
  * the static map and scalar layout match the constants the rest of the stack
    indexes by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from crsim.cards import CARD_DEFS, CardType
from crsim.constants import (
    ARENA_H,
    ARENA_W,
    BRIDGE_LEFT_COLS,
    BRIDGE_RIGHT_COLS,
    CH_ENEMY_GROUND,
    CH_FRIENDLY_GROUND,
    CH_STATIC_MAP,
    MAX_ELIXIR,
    NUM_CARD_TYPES,
    RIVER_ROW_HI,
    RIVER_ROW_LO,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
)
from crsim.entities import entity_from_card
from crsim.game import CRGame
from model.features import encode_batch, encode_state


def _deploy(game: CRGame, card_type: CardType, owner: int, x: float, y: float):
    e = entity_from_card(game._alloc_eid(), owner, CARD_DEFS[card_type], x, y)
    e.deploy_timer = 0.0
    e.is_deployed = True
    game.entities.append(e)
    return e


def test_encode_state_shapes_dtype_and_finite():
    g = CRGame(seed=0)
    spatial, scalar = encode_state(g, 0)
    assert spatial.shape == (SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    assert scalar.shape == (SCALAR_FEATURES,)
    assert spatial.dtype == np.float32 and scalar.dtype == np.float32
    assert np.isfinite(spatial).all() and np.isfinite(scalar).all()


def test_encode_state_is_deterministic():
    g = CRGame(seed=0)
    _deploy(g, CardType.KNIGHT, 0, 4.0, 8.0)
    a = encode_state(g, 0)
    b = encode_state(g, 0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_perspective_flips_ownership_and_row():
    # One unit, two viewpoints. To its owner it is friendly at its own row; to
    # the opponent it is an enemy at the vertically mirrored row.
    g = CRGame(seed=0)
    _deploy(g, CardType.KNIGHT, 0, 4.0, 8.0)

    sp0, _ = encode_state(g, 0)
    sp1, _ = encode_state(g, 1)

    fr_row, fr_col = np.unravel_index(
        np.argmax(sp0[CH_FRIENDLY_GROUND]), sp0[CH_FRIENDLY_GROUND].shape
    )
    assert (fr_row, fr_col) == (8, 4)
    assert sp0[CH_ENEMY_GROUND].max() == 0.0  # no enemy non-tower units

    en_row, en_col = np.unravel_index(
        np.argmax(sp1[CH_ENEMY_GROUND]), sp1[CH_ENEMY_GROUND].shape
    )
    assert (en_row, en_col) == ((ARENA_H - 1) - 8, 4)
    assert sp1[CH_FRIENDLY_GROUND].max() == 0.0


def test_static_map_plane_is_river_and_bridges():
    g = CRGame(seed=0)
    spatial, _ = encode_state(g, 0)
    plane = spatial[CH_STATIC_MAP]

    bridge_cols = set(range(BRIDGE_LEFT_COLS[0], BRIDGE_LEFT_COLS[1] + 1)) | set(
        range(BRIDGE_RIGHT_COLS[0], BRIDGE_RIGHT_COLS[1] + 1)
    )
    for row in (RIVER_ROW_LO, RIVER_ROW_HI):
        for col in range(ARENA_W):
            expected = 0.5 if col in bridge_cols else 1.0
            assert plane[row, col] == expected
    # A non-river row is empty.
    assert plane[0].sum() == 0.0

    # The river is symmetric, so both perspectives see the same static map.
    spatial1, _ = encode_state(g, 1)
    assert np.array_equal(plane, spatial1[CH_STATIC_MAP])


def test_scalar_elixir_phase_and_hand_onehots():
    g = CRGame(seed=0)
    g.players[0].elixir = 6.0
    _, scalar = encode_state(g, 0)

    assert scalar[0] == pytest.approx(6.0 / MAX_ELIXIR)
    assert scalar[1] == pytest.approx(float(g.phase) / 3.0)

    # Four hand one-hots of width NUM_CARD_TYPES start right after the 2 globals.
    ps = g.players[0]
    for slot in range(4):
        block = scalar[2 + slot * NUM_CARD_TYPES : 2 + (slot + 1) * NUM_CARD_TYPES]
        assert block.sum() == pytest.approx(1.0)
        card_type = ps.deck[ps.hand[slot]]
        assert block[int(card_type)] == 1.0


def test_encode_batch_shapes_match_single():
    g0, g1 = CRGame(seed=0), CRGame(seed=1)
    spatials, scalars = encode_batch([g0, g1], [0, 1])
    assert spatials.shape == (2, SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    assert scalars.shape == (2, SCALAR_FEATURES)
    single_sp, single_sc = encode_state(g0, 0)
    assert np.array_equal(spatials[0], single_sp)
    assert np.array_equal(scalars[0], single_sc)
