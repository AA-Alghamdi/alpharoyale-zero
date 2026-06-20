"""Fixed reference opponents and an Elo-anchored evaluation harness.

The project previously had no opponent stronger than "place random cards", so
there was no way to measure whether a trained agent was actually any good. This
module provides:

  * ``RandomAgent``   — a weak floor (random legal placements).
  * ``HeuristicAgent``— a simple but non-trivial scripted policy (defend the
    threatened lane, push when elixir is banked). This is the R2 "beat a
    heuristic" rung from the strategy roadmap.
  * ``SearchAgent``   — adapts a ``GumbelMuZeroSearch`` (neural agent) to the
    same interface so a model can be measured against the fixed baselines.
  * ``play_match`` / ``evaluate_agent`` — run games and convert win-rate into an
    Elo gap against a fixed-rating anchor.

All agents implement ``select_action(game, player) -> action_id`` and act only
on decision ticks (the match runner enforces the cadence), so they decide at a
human-like ~2-3 actions/sec rather than every 50ms tick.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from crsim.cards import CARD_DEFS, EntityKind
from crsim.constants import (
    ABILITY_ACTION,
    ARENA_H,
    ARENA_W,
    BRIDGE_LEFT_COLS,
    BRIDGE_RIGHT_COLS,
    NUM_HAND_SLOTS,
    RIVER_ROW_HI,
    RIVER_ROW_LO,
    WAIT_ACTION,
)
from crsim.game import Action, CRGame, GameResult


@runtime_checkable
class Agent(Protocol):
    """Anything that can pick an action id for a player in a game state."""

    def select_action(self, game: CRGame, player: int) -> int:
        ...


def _place_action_id(
    mask: np.ndarray, slot: int, x: int, y: int
) -> int | None:
    """Nearest legal placement cell for ``slot`` to the desired ``(x, y)``.

    Returns the flat action id, or ``None`` if the slot has no legal cell.
    """
    base = slot * ARENA_W * ARENA_H
    best_id: int | None = None
    best_d = float("inf")
    for cx in range(ARENA_W):
        for cy in range(ARENA_H):
            aid = base + cx * ARENA_H + cy
            if mask[aid]:
                d = (cx - x) ** 2 + (cy - y) ** 2
                if d < best_d:
                    best_d = d
                    best_id = aid
    return best_id


class WaitAgent:
    """Does nothing — the trivial lower bound (only ever banks elixir)."""

    def select_action(self, game: CRGame, player: int) -> int:
        return WAIT_ACTION


class RandomAgent:
    """Picks a random legal placement with probability ``play_prob``, else waits."""

    def __init__(self, play_prob: float = 0.35, seed: int = 0) -> None:
        self.play_prob = play_prob
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: CRGame, player: int) -> int:
        if self.rng.random() > self.play_prob:
            return WAIT_ACTION
        mask = game.get_valid_actions_mask(player)
        legal = np.where(mask)[0]
        # Drop WAIT so this actually plays something when it decides to.
        legal = legal[legal != WAIT_ACTION]
        if len(legal) == 0:
            return WAIT_ACTION
        return int(self.rng.choice(legal))


class HeuristicAgent:
    """A simple but non-trivial scripted policy.

    Two rules, in priority order:
      1. **Defend.** If an enemy troop is on our half, meet the most-advanced
         one head-on with our highest-DPS affordable troop.
      2. **Pressure.** Otherwise, if elixir is at/above ``push_elixir``, drop
         the cheapest affordable troop at a bridge, alternating lanes so both
         enemy princess towers are threatened.

    The low default ``push_elixir`` (spend early, never bank to the cap) and
    the alternating two-lane pressure were chosen empirically: in the current
    simulator, hoarding elixir or committing to a single lane loses to a random
    opponent's spread placement. With these rules it edges out the random floor.
    """

    def __init__(self, push_elixir: float = 3.0) -> None:
        self.push_elixir = push_elixir
        self._lane = 0

    def _on_our_half(self, player: int, y: float) -> bool:
        return y < RIVER_ROW_LO if player == 0 else y > RIVER_ROW_HI

    def _troops(self, game: CRGame, player: int) -> list[tuple[int, object]]:
        ps = game.players[player]
        out = []
        for slot in range(NUM_HAND_SLOTS):
            card_def = CARD_DEFS[ps.deck[ps.hand[slot]]]
            if card_def.kind == EntityKind.TROOP and ps.elixir >= card_def.cost:
                out.append((slot, card_def))
        return out

    def select_action(self, game: CRGame, player: int) -> int:
        troops = self._troops(game, player)
        if not troops:
            return WAIT_ACTION
        mask = game.get_valid_actions_mask(player)

        threats = [
            e
            for e in game.entities
            if e.alive and not e.is_tower and e.owner != player
            and self._on_our_half(player, e.y)
        ]

        if threats:
            # Counter the most-advanced threat head-on with our best DPS troop.
            slot = max(troops, key=lambda s: s[1].dps)[0]
            target = min(
                threats,
                key=lambda e: e.y if player == 0 else (ARENA_H - e.y),
            )
            ty = int(target.y) - 1 if player == 0 else int(target.y) + 1
            aid = _place_action_id(mask, slot, int(target.x), ty)
            return aid if aid is not None else WAIT_ACTION

        ps = game.players[player]
        if ps.elixir >= self.push_elixir:
            slot = min(troops, key=lambda s: s[1].cost)[0]
            col = BRIDGE_RIGHT_COLS[0] if self._lane % 2 == 0 else BRIDGE_LEFT_COLS[0]
            self._lane += 1
            py = RIVER_ROW_LO - 1 if player == 0 else RIVER_ROW_HI + 1
            aid = _place_action_id(mask, slot, col, py)
            return aid if aid is not None else WAIT_ACTION

        return WAIT_ACTION


class SearchAgent:
    """Adapt a GumbelMuZeroSearch (neural agent) to the Agent interface."""

    def __init__(self, searcher, deterministic: bool = True) -> None:
        self.searcher = searcher
        self.deterministic = deterministic

    def select_action(self, game: CRGame, player: int) -> int:
        action_id, _ = self.searcher.select_action(
            game, player, deterministic=self.deterministic
        )
        return int(action_id)


def _action_from_id(action_id: int, player: int) -> Action:
    if action_id == WAIT_ACTION:
        return Action(player=player, hand_slot=-1)
    if action_id == ABILITY_ACTION:
        return Action(player=player, hand_slot=-1, ability=True)
    slot = action_id // (ARENA_W * ARENA_H)
    remainder = action_id % (ARENA_W * ARENA_H)
    x = remainder // ARENA_H
    y = remainder % ARENA_H
    return Action(player=player, hand_slot=slot, x=float(x), y=float(y))


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    n_games: int = 20,
    decision_interval: int = 8,
    seed: int = 0,
) -> tuple[int, int, int]:
    """Play ``n_games`` between two agents, alternating sides.

    Agents act only on decision ticks; both players wait in between. Returns
    ``(a_wins, b_wins, draws)``.
    """
    a_wins = b_wins = draws = 0
    interval = max(1, decision_interval)

    for game_idx in range(n_games):
        a_is_p0 = game_idx % 2 == 0
        p0_agent = agent_a if a_is_p0 else agent_b
        p1_agent = agent_b if a_is_p0 else agent_a

        game = CRGame(seed=seed + game_idx)
        while not game.done:
            if game.tick_count % interval == 0:
                a0 = _action_from_id(p0_agent.select_action(game, 0), 0)
                a1 = _action_from_id(p1_agent.select_action(game, 1), 1)
            else:
                a0 = Action(player=0, hand_slot=-1)
                a1 = Action(player=1, hand_slot=-1)
            game.step([a0, a1])

        if game.result == GameResult.P0_WIN:
            winner_a = a_is_p0
        elif game.result == GameResult.P1_WIN:
            winner_a = not a_is_p0
        else:
            winner_a = None

        if winner_a is True:
            a_wins += 1
        elif winner_a is False:
            b_wins += 1
        else:
            draws += 1

    return a_wins, b_wins, draws


def winrate_to_elo(winrate: float, anchor: float = 1000.0) -> float:
    """Convert a win-rate against a fixed-rating opponent into an Elo estimate."""
    p = min(max(winrate, 1e-3), 1 - 1e-3)
    return anchor + 400.0 * math.log10(p / (1 - p))


def evaluate_agent(
    agent: Agent,
    baseline: Agent,
    n_games: int = 20,
    decision_interval: int = 8,
    baseline_elo: float = 1000.0,
    seed: int = 0,
) -> dict[str, float]:
    """Measure ``agent`` against a fixed ``baseline`` and estimate its Elo."""
    a_wins, b_wins, draws = play_match(
        agent, baseline, n_games=n_games, decision_interval=decision_interval, seed=seed
    )
    total = a_wins + b_wins + draws
    winrate = (a_wins + 0.5 * draws) / max(total, 1)
    return {
        "wins": float(a_wins),
        "losses": float(b_wins),
        "draws": float(draws),
        "winrate": winrate,
        "elo": winrate_to_elo(winrate, anchor=baseline_elo),
    }
