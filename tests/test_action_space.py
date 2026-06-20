"""RL action-space contract tests.

The policy/value network, every MCTS variant, the baseline agents and the
self-play loop all talk to the simulator through a single flat action space.
This file pins that contract:

  * the layout (size, ABILITY / WAIT positions) is what every component assumes;
  * :func:`action_id_to_action` and :func:`action_to_action_id` round-trip;
  * **every** decoder (canonical + the re-exports the search/eval modules ship)
    agrees, and in particular decodes ``ABILITY_ACTION`` to an ability action.

That last point is a regression gate: two search modules used to decode
``ABILITY_ACTION`` as ``hand_slot = 4``, which silently mis-fires (and would
``IndexError`` on the 4-slot hand) instead of activating the champion ability.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from crsim.actions import action_id_to_action, action_to_action_id
from crsim.cards import CARD_DEFS, CardType
from crsim.constants import (
    ABILITY_ACTION,
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    NUM_HAND_SLOTS,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
    WAIT_ACTION,
)
from crsim.entities import entity_from_card
from crsim.game import Action, CRGame

# Every decoder that ships in the codebase. They must all behave identically.
from eval.baseline_agents import _action_from_id
from mcts.gumbel_search import _action_id_to_action as gumbel_decode
from mcts.search import _action_id_to_action as search_decode
from model.network import CRZeroNet

ALL_DECODERS = [action_id_to_action, gumbel_decode, search_decode, _action_from_id]

_TILES_PER_SLOT = ARENA_W * ARENA_H


# ---------------------------------------------------------------------------
# Layout invariants
# ---------------------------------------------------------------------------


def test_action_space_layout():
    assert _TILES_PER_SLOT == 18 * 32 == 576
    assert ABILITY_ACTION == NUM_HAND_SLOTS * _TILES_PER_SLOT == 2304
    assert WAIT_ACTION == ABILITY_ACTION + 1 == 2305
    assert ACTION_SPACE_SIZE == NUM_HAND_SLOTS * _TILES_PER_SLOT + 2 == 2306


# ---------------------------------------------------------------------------
# Round-trip: id -> Action -> id
# ---------------------------------------------------------------------------


def test_special_actions_roundtrip():
    wait = action_id_to_action(WAIT_ACTION, player=0)
    assert wait.is_wait and not wait.ability
    assert action_to_action_id(wait) == WAIT_ACTION

    ability = action_id_to_action(ABILITY_ACTION, player=1)
    assert ability.ability and ability.hand_slot < 0
    assert action_to_action_id(ability) == ABILITY_ACTION


def test_placement_roundtrip_is_identity():
    # Decode a broad sample of placement ids, re-encode, require exact round-trip.
    rng = np.random.default_rng(0)
    sample = rng.choice(ABILITY_ACTION, size=2000, replace=False)
    for action_id in [0, ABILITY_ACTION - 1, *map(int, sample)]:
        action = action_id_to_action(action_id, player=0)
        assert not action.ability and not action.is_wait
        assert 0 <= action.hand_slot < NUM_HAND_SLOTS
        assert 0 <= action.x < ARENA_W and 0 <= action.y < ARENA_H
        assert action_to_action_id(action) == action_id


def test_corner_placements_decode_to_grid_corners():
    # id == slot*W*H + x*H + y  ->  slot 0, (x, y) = (0, 0)
    a = action_id_to_action(0, player=0)
    assert (a.hand_slot, a.x, a.y) == (0, 0.0, 0.0)
    # Last placement of slot 0: x = W-1, y = H-1.
    a = action_id_to_action(_TILES_PER_SLOT - 1, player=0)
    assert (a.hand_slot, a.x, a.y) == (0, float(ARENA_W - 1), float(ARENA_H - 1))
    # First placement of slot 3.
    a = action_id_to_action(3 * _TILES_PER_SLOT, player=0)
    assert (a.hand_slot, a.x, a.y) == (3, 0.0, 0.0)


@pytest.mark.parametrize("bad_id", [-1, ACTION_SPACE_SIZE, ACTION_SPACE_SIZE + 100])
def test_out_of_range_id_raises(bad_id):
    with pytest.raises(ValueError):
        action_id_to_action(bad_id, player=0)


def test_encode_rejects_invalid_actions():
    with pytest.raises(ValueError):
        action_to_action_id(Action(player=0, hand_slot=NUM_HAND_SLOTS, x=0.0, y=0.0))
    with pytest.raises(ValueError):
        action_to_action_id(Action(player=0, hand_slot=0, x=float(ARENA_W), y=0.0))


# ---------------------------------------------------------------------------
# All shipped decoders agree (regression gate for the ABILITY drift bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decode", ALL_DECODERS)
def test_every_decoder_handles_ability(decode):
    a = decode(ABILITY_ACTION, 0)
    assert a.ability is True, f"{decode.__module__}.{decode.__name__} dropped the ABILITY case"
    # The historical bug decoded ABILITY as a placement in a non-existent slot.
    assert not (0 <= a.hand_slot < NUM_HAND_SLOTS)


def test_all_decoders_agree():
    ids = [0, 1, 575, 576, 2303, ABILITY_ACTION, WAIT_ACTION]
    for action_id in ids:
        decoded = [d(action_id, 0) for d in ALL_DECODERS]
        first = decoded[0]
        for other in decoded[1:]:
            assert (first.hand_slot, first.x, first.y, first.ability, first.is_wait) == (
                other.hand_slot,
                other.x,
                other.y,
                other.ability,
                other.is_wait,
            )


# ---------------------------------------------------------------------------
# Integration with the simulator's legal-action mask
# ---------------------------------------------------------------------------


def test_valid_mask_actions_all_applyable():
    # Every id the game marks legal must decode and apply without raising —
    # this is exactly what the buggy ABILITY decoder violated.
    g = CRGame(seed=0)
    g.players[0].elixir = 10.0
    mask = g.get_valid_actions_mask(0)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask[WAIT_ACTION]
    legal_ids = np.flatnonzero(mask)
    assert len(legal_ids) > NUM_HAND_SLOTS  # at least some placements + wait
    for action_id in legal_ids:
        action = action_id_to_action(int(action_id), player=0)
        g.clone().apply_action(action)  # must not raise (e.g. no IndexError)


def test_ability_action_id_activates_champion():
    # Decoding ABILITY_ACTION and applying it must fire the champion ability,
    # not poke a non-existent hand slot.
    g = CRGame(seed=0)
    champ = entity_from_card(g._alloc_eid(), 0, CARD_DEFS[CardType.SKELETON_KING], 9.0, 6.0)
    champ.deploy_timer = 0.0
    champ.is_deployed = True
    g.entities.append(champ)
    g.players[0].elixir = 8.0

    assert g.get_valid_actions_mask(0)[ABILITY_ACTION]

    def n_skeletons() -> int:
        return sum(
            1
            for e in g.entities
            if e.alive and e.owner == 0 and e.card_type == CardType.SKELETONS
        )

    before = n_skeletons()
    g.apply_action(action_id_to_action(ABILITY_ACTION, player=0))
    assert n_skeletons() - before == 6, (
        "Skeleton King ability (via ABILITY_ACTION) must summon 6 skeletons"
    )


# ---------------------------------------------------------------------------
# Network policy head matches the action space
# ---------------------------------------------------------------------------


def test_network_policy_dim_matches_action_space():
    net = CRZeroNet(n_res_blocks=1, n_filters=16)
    spatial = torch.zeros(1, SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    scalar = torch.zeros(1, SCALAR_FEATURES)
    logits, value = net(spatial, scalar)
    assert logits.shape == (1, ACTION_SPACE_SIZE)
    assert value.shape == (1,)
