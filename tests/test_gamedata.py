"""Verify the authentic-data overlay matches known tournament-standard values.

These anchors are pinned against the Clash Royale wiki (Level 11 / King Tower
Level 11), sourced from the decoded game data in ``cr_engine/gamedata_full``.
If the data pipeline or scaling regresses, these break.
"""

from __future__ import annotations

import pytest

from crsim.cards import AUTHENTIC_STAT_REPORT, CARD_DEFS, CardType
from crsim.gamedata import GameData, scale_stat


def test_overlay_is_active():
    # The bulk of the roster is sourced from authentic data (the newest cards
    # absent from this snapshot keep hand-coded values).
    assert len(AUTHENTIC_STAT_REPORT) >= 85


def test_scale_stat_is_exact_per_rarity():
    # Exact per-level table from rarities.json (index 9 - relative_level).
    assert round(scale_stat(690, "Common")) == 1766   # Knight L1 690 -> L11
    assert round(scale_stat(1930, "Rare")) == 4092     # Giant
    assert round(scale_stat(2350, "Epic")) == 3760     # P.E.K.K.A
    assert round(scale_stat(3300, "Legendary")) == 3993  # Mega Knight
    assert scale_stat(1000, "Champion") == 1000        # Champions at base
    assert scale_stat(0, "Common") == 0


@pytest.mark.parametrize(
    "card, field, expected, tol",
    [
        (CardType.KNIGHT, "hp", 1766, 1),
        (CardType.KNIGHT, "damage_per_hit", 202, 2),
        (CardType.GIANT, "hp", 4092, 2),
        (CardType.PEKKA, "hp", 3760, 2),
        (CardType.PEKKA, "damage_per_hit", 816, 2),
        (CardType.MUSKETEER, "damage_per_hit", 218, 4),
        (CardType.WIZARD, "damage_per_hit", 282, 3),
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
    assert wizard.splash_radius == pytest.approx(1.5)
    # Musketeer is single-target: its projectile has no splash radius.
    assert CARD_DEFS[CardType.MUSKETEER].splash_radius == 0.0


def test_authentic_metadata_matches_known_cards():
    assert CARD_DEFS[CardType.KNIGHT].hit_speed == pytest.approx(1.2)
    assert CARD_DEFS[CardType.GOBLINS].cost == 2
    assert CARD_DEFS[CardType.GIANT].target_mode.name == "BUILDINGS"
    assert CARD_DEFS[CardType.MINIONS].is_flying is True


def test_gamedata_loads_full_roster():
    game = GameData()
    assert game.available
    assert len(game.characters) >= 100
    assert "Knight" in game.characters
    assert game.characters["Knight"]["hitpoints"] == 690  # L1 base
    assert "Cannon" in game.buildings
    assert game.spells["Poison"]["life_duration"] > 0
    # Exact level scaling table is present.
    assert game.level_multiplier("Common") == pytest.approx(2.56)


def test_plural_skey_troops_are_overlaid():
    # Multi-unit troops carry a plural sc_key ("Goblins") but the stat row is
    # keyed by the singular entity ("Goblin"); the singular fallback must match.
    from crsim.gamedata import authentic_core_fields
    game = GameData()
    for ct in (
        CardType.GOBLINS,
        CardType.BARBARIANS,
        CardType.SKELETONS,
        CardType.BATS,
        CardType.MINIONS,
        CardType.SKELETON_ARMY,
        CardType.MINION_HORDE,
    ):
        assert authentic_core_fields(game, ct) is not None, ct.name


def test_skeleton_army_units_scale_by_entity_rarity():
    # Skeleton Army is an Epic card but spawns Common Skeletons; per-unit HP must
    # match the Skeletons card, not scale by the card's Epic tier.
    assert CARD_DEFS[CardType.SKELETON_ARMY].hp == pytest.approx(
        CARD_DEFS[CardType.SKELETONS].hp
    )


def test_compound_cards_keep_air_ground_targeting():
    # Ram Rider / Goblin Giant ride a building-targeting mount, but the rider hits
    # air+ground. The single-entity model must not be flipped to BUILDINGS.
    assert CARD_DEFS[CardType.RAM_RIDER].target_mode.name == "AIR_GROUND"
    assert CARD_DEFS[CardType.GOBLIN_GIANT].target_mode.name == "AIR_GROUND"
