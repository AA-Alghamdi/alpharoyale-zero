"""Opt-in behavioural cross-engine gate (1v1 duels).

Skipped automatically when the oracle clone (samdickson22/clash-simulator) or its
deps are unavailable — i.e. it does not run in CI. Run it locally with the oracle
checked out next to this repo, or set ``$ORACLE_REPO``.

Asserts the two engines agree on every duel winner, and on time-to-kill for
single-unit duels (multi-unit swarms differ in spawn geometry across engines, so
only the winner is checked there).
"""
from __future__ import annotations

import pytest

from crsim.cards import CARD_DEFS, CardType

try:
    from verification.behavioral import DUELS, _resolve_oracle, run_duels

    _ORACLE = _resolve_oracle(None)
    run_duels(_ORACLE, DUELS[:1])  # smoke: oracle importable + deps present
    _AVAILABLE = True
    _SKIP_REASON = ""
except Exception as exc:  # noqa: BLE001 - any failure means "oracle not usable here"
    _AVAILABLE = False
    _SKIP_REASON = f"oracle engine unavailable: {exc}"

pytestmark = [
    pytest.mark.cross_engine,
    pytest.mark.skipif(not _AVAILABLE, reason=_SKIP_REASON),
]


def _is_single_unit(card: str) -> bool:
    return CARD_DEFS[CardType[card]].spawn_count == 1


@pytest.fixture(scope="module")
def duel_results():
    return run_duels(_ORACLE)


def test_duel_winners_agree(duel_results):
    mismatched = [(r.a, r.b, r.crsim_winner, r.oracle_winner) for r in duel_results if not r.winners_agree]
    assert not mismatched, f"engines disagree on duel winner: {mismatched}"


def test_single_unit_duel_ttk_agrees(duel_results):
    bad = [
        (r.a, r.b, r.crsim_ttk, r.oracle_ttk)
        for r in duel_results
        if _is_single_unit(r.a) and _is_single_unit(r.b) and not r.ttk_agree
    ]
    assert not bad, f"time-to-kill diverges beyond tolerance: {bad}"
