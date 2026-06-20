#!/usr/bin/env python3
"""Rank a frozen pool of agents on an order-independent Elo ladder.

By default it ranks the scripted reference bots against each other, which gives
the project a fixed strength yardstick:

    python scripts/ladder.py --games 20

Point ``--checkpoint`` at a trained model to drop a neural ``SearchAgent`` into
the same pool and read its Elo straight off the ladder:

    python scripts/ladder.py --checkpoint checkpoints/best.pt --games 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.baseline_agents import (
    HeuristicAgent,
    MetaAgent,
    RandomAgent,
    WaitAgent,
)
from eval.ladder import format_standings, rank_pool


def build_pool(checkpoint: str | None, seed: int) -> dict:
    pool = {
        "wait": WaitAgent(),
        "random": RandomAgent(seed=seed),
        "heuristic": HeuristicAgent(),
        "meta": MetaAgent(),
    }
    if checkpoint:
        import torch

        from eval.baseline_agents import SearchAgent
        from mcts.gumbel_search import GumbelConfig, GumbelMuZeroSearch
        from model.network import CRZeroNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model = CRZeroNet()
        model.load_state_dict(state.get("model_state_dict", state))
        model.to(device).eval()
        searcher = GumbelMuZeroSearch(
            model=model, config=GumbelConfig(n_simulations=16), device=device
        )
        pool["model"] = SearchAgent(searcher)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20, help="games per pairing")
    parser.add_argument("--decision-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor", type=float, default=1000.0)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="add a neural SearchAgent"
    )
    args = parser.parse_args()

    pool = build_pool(args.checkpoint, args.seed)
    standings = rank_pool(
        pool,
        n_games=args.games,
        decision_interval=args.decision_interval,
        seed=args.seed,
        anchor=args.anchor,
    )
    print(format_standings(standings))


if __name__ == "__main__":
    main()
