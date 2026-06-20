"""Tests for the order-independent Elo ladder over a frozen agent pool.

Two layers: the rating fit is checked on synthetic pairwise tables where the
right answer is known by construction, and a small deterministic round-robin
checks that the whole pipeline ranks the scripted bots in their true strength
order (Meta > Heuristic > Wait).
"""

from __future__ import annotations

import math

from eval.baseline_agents import HeuristicAgent, MetaAgent, WaitAgent
from eval.ladder import fit_elo, format_standings, rank_pool, run_round_robin


def _sweep(winner: str, loser: str, n: int = 10) -> tuple[str, str, tuple[int, int, int]]:
    """A clean sweep: ``winner`` beats ``loser`` ``n``-0 in canonical order."""
    key = (winner, loser) if winner <= loser else (loser, winner)
    res = (n, 0, 0) if key == (winner, loser) else (0, n, 0)
    return (*key, res)


def _table(*sweeps) -> dict[tuple[str, str], tuple[int, int, int]]:
    return {(a, b): res for a, b, res in sweeps}


def test_fit_elo_orders_by_strength():
    # Transitive pool: A beats B, B beats C, A beats C.
    pairwise = _table(
        _sweep("A", "B"),
        _sweep("B", "C"),
        _sweep("A", "C"),
    )
    elo = fit_elo(pairwise, ["A", "B", "C"], anchor=1000.0)
    assert elo["A"] > elo["B"] > elo["C"]
    # Mean-anchored.
    assert abs(sum(elo.values()) / 3 - 1000.0) < 1e-6
    assert all(math.isfinite(v) for v in elo.values())


def test_fit_elo_symmetric_pool_is_flat():
    # Everyone draws everyone -> identical ratings at the anchor.
    pairwise = {
        ("A", "B"): (0, 0, 10),
        ("A", "C"): (0, 0, 10),
        ("B", "C"): (0, 0, 10),
    }
    elo = fit_elo(pairwise, ["A", "B", "C"], anchor=1000.0)
    assert abs(elo["A"] - 1000.0) < 1e-6
    assert abs(elo["A"] - elo["B"]) < 1e-6
    assert abs(elo["B"] - elo["C"]) < 1e-6


def test_fit_elo_is_order_independent():
    # The fit depends only on aggregate counts, not on dict/name iteration order.
    pairwise = _table(
        _sweep("A", "B", 7),
        _sweep("B", "C", 6),
        _sweep("A", "C", 9),
    )
    elo_1 = fit_elo(pairwise, ["A", "B", "C"], anchor=1000.0)
    elo_2 = fit_elo(pairwise, ["C", "A", "B"], anchor=1000.0)
    for name in ("A", "B", "C"):
        assert abs(elo_1[name] - elo_2[name]) < 1e-6


def test_fit_elo_undefeated_stays_finite():
    # A sweeps everyone; without a prior its strength would diverge to +inf.
    pairwise = _table(
        _sweep("A", "B"),
        _sweep("A", "C"),
        _sweep("B", "C"),
    )
    elo = fit_elo(pairwise, ["A", "B", "C"], anchor=1000.0, prior_games=2.0)
    assert all(math.isfinite(v) for v in elo.values())
    assert elo["A"] > elo["B"] > elo["C"]


def test_run_round_robin_covers_every_pair():
    agents = {"wait": WaitAgent(), "heuristic": HeuristicAgent()}
    pairwise = run_round_robin(agents, n_games=2, seed=0)
    assert set(pairwise) == {("heuristic", "wait")}
    wins_h, wins_w, draws = pairwise[("heuristic", "wait")]
    assert wins_h + wins_w + draws == 2


def test_rank_pool_orders_scripted_bots():
    # Deterministic pool (no RNG agents): Meta beats Heuristic beats Wait, so the
    # fitted ladder must reproduce that strict order.
    agents = {
        "wait": WaitAgent(),
        "heuristic": HeuristicAgent(),
        "meta": MetaAgent(),
    }
    standings = rank_pool(agents, n_games=6, seed=0)
    names_by_rank = [e.name for e in standings]
    assert names_by_rank == ["meta", "heuristic", "wait"]
    # Every agent played the same number of games and the table renders.
    assert len({e.games for e in standings}) == 1
    assert "meta" in format_standings(standings)
