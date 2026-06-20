"""Tests for model entity-feature extraction.

These verify that the per-entity features the RL agent observes correctly
surface the modern mechanics (champion-ability readiness and evolved state),
so the policy can learn *when* to activate an ability or deploy an evolution.
"""

from __future__ import annotations

import pytest

from crsim.cards import CARD_DEFS, CardType
from crsim.entities import entity_from_card
from crsim.game import CRGame
from model.features import extract_entity_features


def _fresh_game() -> CRGame:
    return CRGame(seed=0)


def _spawn(
    game: CRGame,
    card_type: CardType,
    owner: int,
    x: float,
    y: float,
    *,
    evolved: bool = False,
):
    e = entity_from_card(
        game._alloc_eid(), owner, CARD_DEFS[card_type], x, y, evolved=evolved
    )
    e.deploy_timer = 0.0
    e.is_deployed = True
    game.entities.append(e)
    return e


def _row_for(game: CRGame, entity, player: int):
    """Return the feature row the extractor produces for ``entity``."""
    features, mask = extract_entity_features(game, player)
    alive = [e for e in game.entities if e.alive]
    idx = alive.index(entity)
    assert mask[idx]
    return features[idx]


def test_champion_ability_ready_is_encoded():
    g = _fresh_game()
    champ = _spawn(g, CardType.SKELETON_KING, 0, 9.0, 6.0)

    row = _row_for(g, champ, 0)
    assert row[25] == 1.0  # is_champion
    assert row[33] == 1.0  # ability ready (no cooldown)
    assert row[34] == 0.0  # no cooldown fraction remaining


def test_champion_ability_cooldown_fraction_is_encoded():
    g = _fresh_game()
    champ = _spawn(g, CardType.SKELETON_KING, 0, 9.0, 6.0)
    champ.ability_cooldown_timer = champ.ability_cooldown / 2.0

    row = _row_for(g, champ, 0)
    assert row[33] == 0.0  # not ready while on cooldown
    assert row[34] == pytest.approx(0.5, rel=0.02)


def test_non_champion_has_no_ability_flags():
    g = _fresh_game()
    knight = _spawn(g, CardType.KNIGHT, 0, 9.0, 6.0)

    row = _row_for(g, knight, 0)
    assert row[25] == 0.0  # not a champion
    assert row[33] == 0.0
    assert row[34] == 0.0


def test_evolved_state_is_encoded():
    g = _fresh_game()
    evolved = _spawn(g, CardType.KNIGHT, 0, 9.0, 6.0, evolved=True)
    plain = _spawn(g, CardType.KNIGHT, 0, 8.0, 6.0, evolved=False)

    assert _row_for(g, evolved, 0)[23] == 1.0
    assert _row_for(g, plain, 0)[23] == 0.0


def test_active_ability_effects_are_encoded():
    g = _fresh_game()
    aq = _spawn(g, CardType.ARCHER_QUEEN, 0, 9.0, 16.0)
    aq.invisible_timer = 3.5
    aq.attack_speed_timer = 3.5
    knight = _spawn(g, CardType.KNIGHT, 0, 8.0, 6.0, evolved=True)
    knight.evo_shield = True

    aq_row = _row_for(g, aq, 0)
    assert aq_row[35] == 1.0  # cloak / invisible active
    assert aq_row[36] == 1.0  # attack-speed boost active

    knight_row = _row_for(g, knight, 0)
    assert knight_row[38] == 1.0  # evo one-shot shield active
