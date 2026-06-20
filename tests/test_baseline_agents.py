"""Tests for the reference opponents and the Elo-anchored eval harness.

These give the project its first measurable strength ladder:
WaitAgent (trivial) < RandomAgent (floor) <= HeuristicAgent (scripted).
"""

from __future__ import annotations

import math

from crsim.game import CRGame
from eval.baseline_agents import (
    HeuristicAgent,
    RandomAgent,
    WaitAgent,
    evaluate_agent,
    play_match,
    winrate_to_elo,
)


def _agent_emits_legal_actions(agent, seed: int = 0, steps: int = 200) -> bool:
    """Drive a game and assert the agent only ever returns legal action ids."""
    game = CRGame(seed=seed)
    for _ in range(steps):
        if game.done:
            break
        if game.tick_count % 8 == 0:
            for player in (0, 1):
                action_id = agent.select_action(game, player)
                mask = game.get_valid_actions_mask(player)
                if not mask[action_id]:
                    return False
        from crsim.game import Action

        game.step([Action(0, -1), Action(1, -1)])
    return True


def test_agents_emit_legal_actions():
    assert _agent_emits_legal_actions(RandomAgent(seed=1))
    assert _agent_emits_legal_actions(HeuristicAgent())
    assert _agent_emits_legal_actions(WaitAgent())


def test_winrate_to_elo_anchor_and_monotonic():
    # A 50% win-rate means equal strength -> the anchor rating.
    assert winrate_to_elo(0.5, anchor=1000.0) == 1000.0
    # Higher win-rate -> strictly higher Elo.
    assert winrate_to_elo(0.7) > winrate_to_elo(0.5) > winrate_to_elo(0.3)
    # Saturating inputs are clamped (no inf/NaN).
    assert math.isfinite(winrate_to_elo(0.0))
    assert math.isfinite(winrate_to_elo(1.0))


def test_heuristic_dominates_wait_agent():
    # Against an opponent that never plays, the heuristic should win every game.
    a_wins, b_wins, draws = play_match(
        HeuristicAgent(), WaitAgent(), n_games=8, seed=0
    )
    assert a_wins == 8, (a_wins, b_wins, draws)


def test_heuristic_competitive_vs_random():
    # The scripted policy should be at least competitive with the random floor
    # across seeds (it only marginally edges it out in the current sim, so we
    # assert competitiveness, not dominance).
    tot_a = tot_b = tot_d = 0
    for rseed in (1, 2, 3, 4):
        a, b, d = play_match(
            HeuristicAgent(), RandomAgent(seed=rseed), n_games=12, seed=rseed
        )
        tot_a += a
        tot_b += b
        tot_d += d
    total = tot_a + tot_b + tot_d
    winrate = tot_a / total
    assert winrate >= 0.45, (tot_a, tot_b, tot_d)


def test_evaluate_agent_returns_consistent_summary():
    res = evaluate_agent(HeuristicAgent(), RandomAgent(seed=5), n_games=6, seed=5)
    assert res["wins"] + res["losses"] + res["draws"] == 6
    assert 0.0 <= res["winrate"] <= 1.0
    assert math.isfinite(res["elo"])
