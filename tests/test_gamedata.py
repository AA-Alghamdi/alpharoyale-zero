"""Verify the authentic-data overlay matches known tournament-standard values.

These anchors are pinned against the Clash Royale wiki (Level 11 / King Tower
Level 11). If the CSV pipeline or scaling regresses, these break.
"""

from __future__ import annotations

import pytest

from crsim.cards import AUTHENTIC_STAT_REPORT, CARD_DEFS, CardType
from crsim.gamedata import GameData, scale_stat


def test_overlay_is_active():
    # The vast majority of cards should be sourced from authentic data.
    assert len(AUTHENTIC_STAT_REPORT) >= 90


def test_scale_stat_anchors():
    # Knight (Common) Level 1 -> Level 11.
    assert round(scale_stat(660, "Common")) == 1766
    # Giant (Rare) Level 1 -> Level 11.
    assert abs(scale_stat(1900, "Rare") - 4091) <= 2
    # Champions are already at L11.
    assert scale_stat(1000, "Champion") == 1000
    # Zero stays zero.
    assert scale_stat(0, "Common") == 0


@pytest.mark.parametrize(
    "card, field, expected, tol",
    [
        (CardType.KNIGHT, "hp", 1766, 1),
        (CardType.KNIGHT, "damage_per_hit", 202, 2),
        (CardType.GIANT, "hp", 4091, 2),
        (CardType.MUSKETEER, "damage_per_hit", 218, 4),
        (CardType.WIZARD, "damage_per_hit", 281, 3),
        (CardType.FIREBALL, "damage_per_hit", 689, 2),
        (CardType.ARROWS, "damage_per_hit", 322, 2),
        (CardType.ZAP, "damage_per_hit", 192, 2),
    ],
)
def test_authentic_stat_anchors(card, field, expected, tol):
    actual = getattr(CARD_DEFS[card], field)
    assert abs(actual - expected) <= tol, f"{card.name}.{field}={actual} (want ~{expected})"


def test_ranged_units_get_projectile_damage_and_splash():
    # Wizard's damage + splash live on its projectile, not the character row.
    wizard = CARD_DEFS[CardType.WIZARD]
    assert wizard.damage_per_hit > 0
    assert abs(wizard.splash_radius - 1.2) < 0.01


def test_authentic_metadata_matches_known_cards():
    # Hit speed and elixir cost come straight from the data.
    assert CARD_DEFS[CardType.KNIGHT].hit_speed == pytest.approx(1.1)
    assert CARD_DEFS[CardType.GOBLINS].cost == 2
    assert CARD_DEFS[CardType.GIANT].target_mode.name == "BUILDINGS"
    assert CARD_DEFS[CardType.MINIONS].is_flying is True


def test_gamedata_loads_full_roster():
    game = GameData()
    assert game.available
    assert len(game.characters) >= 100
    assert "Knight" in game.characters
    assert game.spell_others["Fireball"]["ManaCost"] == "4"
