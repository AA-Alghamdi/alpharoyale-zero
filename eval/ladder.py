"""Order-independent Elo ladder over a frozen pool of agents.

The existing :class:`eval.tournament.Tournament` rates models with *online*
Elo: each game nudges the ratings by a K-factor, so the final numbers depend on
the order games were played and a single agent can only be measured against
whatever it happened to face. That is fine for tracking a moving training run,
but it is the wrong tool for ranking a **frozen pool** (the reference bots, or a
set of checkpoints) where we want one stable, reproducible rating per agent.

This module plays a full round-robin and then fits Bradley-Terry / Elo ratings
by maximum likelihood (minorization-maximization), which depends only on the
aggregate win/loss/draw counts — not the order. A small Bayesian prior keeps
undefeated or winless agents finite, and ratings are mean-anchored so they are
comparable across runs.

The pool is any mapping ``name -> Agent`` (see :mod:`eval.baseline_agents`), so
a trained :class:`SearchAgent` can be dropped straight in next to the scripted
bots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from eval.baseline_agents import Agent, play_match

# pairwise[(a, b)] = (wins_a, wins_b, draws), stored once per unordered pair
# with the two names in sorted order.
Pairwise = dict[tuple[str, str], tuple[int, int, int]]


@dataclass
class LadderEntry:
    name: str
    elo: float
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws


def run_round_robin(
    agents: dict[str, Agent],
    n_games: int = 20,
    decision_interval: int = 8,
    seed: int = 0,
) -> Pairwise:
    """Play every unordered pair ``n_games`` times (sides alternate internally).

    Returns the aggregate ``pairwise`` result table consumed by :func:`fit_elo`.
    """
    names = list(agents)
    pairwise: Pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            wa, wb, draws = play_match(
                agents[a],
                agents[b],
                n_games=n_games,
                decision_interval=decision_interval,
                seed=seed + i * 1000 + j,
            )
            key = (a, b) if a <= b else (b, a)
            pairwise[key] = (wa, wb, draws) if key == (a, b) else (wb, wa, draws)
    return pairwise


def fit_elo(
    pairwise: Pairwise,
    names: list[str],
    anchor: float = 1000.0,
    prior_games: float = 2.0,
    max_iters: int = 10_000,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Fit Elo ratings from aggregate pairwise results (order-independent).

    Maximises the Bradley-Terry likelihood with the standard MM iteration in
    rating-strength space ``gamma_i = 10 ** (elo_i / 400)`` (draws count as half
    a win to each side). ``prior_games`` virtual draws against an anchor-rated
    phantom regularise extreme (un)defeated records. Ratings are returned
    mean-anchored to ``anchor``.
    """
    # Aggregate per-agent score (wins + half draws) and per-pair game counts.
    score: dict[str, float] = {n: 0.0 for n in names}
    games: dict[tuple[str, str], int] = {}
    for (a, b), (wa, wb, draws) in pairwise.items():
        score[a] += wa + 0.5 * draws
        score[b] += wb + 0.5 * draws
        n = wa + wb + draws
        games[(a, b)] = games.get((a, b), 0) + n
        games[(b, a)] = games.get((b, a), 0) + n

    gamma_anchor = 1.0  # phantom opponent sits at the (relative) origin
    gamma = {n: 1.0 for n in names}

    for _ in range(max_iters):
        max_delta = 0.0
        for n in names:
            # Effective wins include the prior's half-win.
            wins = score[n] + 0.5 * prior_games
            denom = prior_games / (gamma[n] + gamma_anchor)
            for m in names:
                if m == n:
                    continue
                n_games = games.get((n, m), 0)
                if n_games:
                    denom += n_games / (gamma[n] + gamma[m])
            new_gamma = wins / denom if denom > 0 else gamma[n]
            max_delta = max(max_delta, abs(new_gamma - gamma[n]))
            gamma[n] = new_gamma
        if max_delta < tol:
            break

    elos = {n: 400.0 * math.log10(g) for n, g in gamma.items()}
    shift = anchor - sum(elos.values()) / len(elos)
    return {n: e + shift for n, e in elos.items()}


def rank_pool(
    agents: dict[str, Agent],
    n_games: int = 20,
    decision_interval: int = 8,
    seed: int = 0,
    anchor: float = 1000.0,
) -> list[LadderEntry]:
    """Run the round-robin and return standings sorted by Elo (highest first)."""
    names = list(agents)
    pairwise = run_round_robin(
        agents, n_games=n_games, decision_interval=decision_interval, seed=seed
    )
    elos = fit_elo(pairwise, names, anchor=anchor)

    wld = {n: [0, 0, 0] for n in names}  # wins, losses, draws
    for (a, b), (wa, wb, draws) in pairwise.items():
        wld[a][0] += wa
        wld[a][1] += wb
        wld[a][2] += draws
        wld[b][0] += wb
        wld[b][1] += wa
        wld[b][2] += draws

    entries = [
        LadderEntry(name=n, elo=elos[n], wins=wld[n][0], losses=wld[n][1], draws=wld[n][2])
        for n in names
    ]
    entries.sort(key=lambda e: e.elo, reverse=True)
    return entries


def format_standings(entries: list[LadderEntry]) -> str:
    """Render standings as a fixed-width table."""
    lines = [f"{'#':<3}{'agent':<16}{'elo':>6}  {'W':>4}{'L':>4}{'D':>4}{'win%':>7}"]
    for rank, e in enumerate(entries, 1):
        winrate = e.score / e.games if e.games else 0.0
        lines.append(
            f"{rank:<3}{e.name:<16}{e.elo:>6.0f}  "
            f"{e.wins:>4}{e.losses:>4}{e.draws:>4}{winrate * 100:>6.1f}%"
        )
    return "\n".join(lines)
