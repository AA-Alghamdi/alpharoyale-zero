"""Cross-engine card-stat conformance gate.

Compares crsim ``CARD_DEFS`` against the vendored stats of a second, independent
Clash Royale engine (samdickson22/clash-simulator). Two independent decodings of
authentic game data agreeing is stronger evidence of fidelity than golden tests
alone.

This runs in CI off the committed fixture (no external repo needed). The gate is
a *regression* gate: the set of cards/stats that diverge beyond tolerance must
equal :data:`KNOWN_DIVERGENCES`. A new divergence (a crsim change that moves a
stat away from the oracle) fails the test; resolving one also fails it, forcing
the table below to be kept honest.
"""
from __future__ import annotations

import pytest

from verification.conformance import (
    AGREE,
    NA,
    divergence_set,
    run_conformance,
)

# Every (crsim_card, stat) that diverges beyond tolerance, with the reason and
# which engine matches canonical (tournament-standard, level-11) Clash Royale.
# crsim wins most of these: the oracle has several outright data errors.
KNOWN_DIVERGENCES: dict[tuple[str, str], str] = {
    ("GOBLIN_HUT", "mana"): "crsim correct: Goblin Hut is 5 elixir; oracle says 4.",
    ("GOBLIN_HUT", "hp"): "Spawner HP differs; neither matches a clean L11 value.",
    ("LUMBERJACK", "mana"): "crsim correct: Lumberjack is 4 elixir; oracle says 5.",
    ("LUMBERJACK", "range"): "crsim correct: Lumberjack is melee (~0.7); oracle has 4.5.",
    ("LUMBERJACK", "speed"): "crsim correct: Lumberjack is very fast (4); oracle has 2.",
    ("LUMBERJACK", "damage"): "oracle ~171 is closer to canonical than crsim 242.",
    ("GOBLIN_DRILL", "hp"): "crsim ~1440 matches canonical L11; oracle 2593 is inflated.",
    ("GOBLIN_DRILL", "sight_range"): "Drill is a spawner; sight value is a placeholder either side.",
    ("CANNON_CART", "hp"): "Shield/body split differs: oracle 1833 double-counts vs crsim 893.",
    ("RASCALS", "hp"): "Multi-unit card: crsim models the boy; oracle uses a combined value.",
    ("RASCALS", "hit_speed"): "Multi-unit card: boy vs girls hit speed differ across engines.",
    ("SUSPICIOUS_BUSH", "hit_speed"): "Bush spawns on trigger; hit_speed is a placeholder either side.",
    ("FIRE_SPIRIT", "range"): "0.5-tile difference on a kamikaze AoE; within real-world ambiguity.",
    ("MONK", "hp"): "Champion; ~8% L11 scaling difference, both plausible.",
    ("BERSERKER", "hit_speed"): "Recent card; hit-speed convention differs ~17%.",
    ("BOSS_BANDIT", "damage"): "Very recent card; canonical value uncertain (~10%).",
    ("RUNE_GIANT", "damage"): "Recent card; buff-giant damage uncertain (~21%).",
}

# Classic cards whose core stats both engines should nail. These get explicit
# assertions on top of the regression gate.
CORE_CARDS = [
    "KNIGHT",
    "GIANT",
    "MUSKETEER",
    "VALKYRIE",
    "MINI_PEKKA",
    "BARBARIANS",
    "WIZARD",
    "ARCHERS",
    "PEKKA",
    "WITCH",
]


@pytest.fixture(scope="module")
def conformance():
    return run_conformance()


def test_coverage(conformance):
    _, meta = conformance
    assert meta["mapped"] >= 110, f"name mapping regressed: only {meta['mapped']} cards mapped"


def test_divergences_match_known(conformance):
    comparisons, _ = conformance
    live = divergence_set(comparisons)
    known = set(KNOWN_DIVERGENCES)

    new = sorted(live - known)
    resolved = sorted(known - live)
    assert not new, (
        "New cross-engine divergence(s) appeared — investigate whether crsim "
        f"regressed, then annotate in KNOWN_DIVERGENCES: {new}"
    )
    assert not resolved, (
        "Divergence(s) no longer present — remove them from KNOWN_DIVERGENCES: "
        f"{resolved}"
    )


def test_known_divergences_are_real(conformance):
    """Guard against stale annotations: every documented divergence still diverges."""
    comparisons, _ = conformance
    live = divergence_set(comparisons)
    assert set(KNOWN_DIVERGENCES) <= live


@pytest.mark.parametrize("card", CORE_CARDS)
def test_core_cards_agree(conformance, card):
    comparisons, _ = conformance
    by_name = {c.crsim_name: c for c in comparisons}
    assert card in by_name, f"{card} not mapped to an oracle card"
    cmp = by_name[card]
    for stat in ("mana", "hp", "damage", "hit_speed", "speed"):
        r = cmp.stats.get(stat)
        if r is None or r.status == NA:
            continue
        assert r.status == AGREE, (
            f"{card}.{stat} unexpectedly diverges: crsim={r.ours} oracle={r.oracle} "
            f"({r.pct:.0f}%)"
        )
