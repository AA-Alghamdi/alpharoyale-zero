#!/usr/bin/env python3
"""Evaluate a trained model against baselines or other checkpoints.

Usage:
    # Evaluate checkpoint vs random policy
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --baseline random

    # Pit two checkpoints against each other
    python scripts/evaluate.py --checkpoint checkpoints/step_100000.pt \
                               --opponent checkpoints/step_050000.pt

    # Watch a game (text rendering)
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --watch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from crsim.constants import ARENA_H, ARENA_W
from crsim.game import CRGame, GameResult
from mcts.search import MCTSConfig, MCTSPlayer, _action_id_to_action
from model.network import CRZeroNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("evaluate")


class RandomPlayer:
    """Baseline: selects a random valid action each tick."""

    def select_action(self, game: CRGame, player: int) -> int:
        mask = game.get_valid_actions_mask(player)
        valid_ids = np.where(mask)[0]
        return int(np.random.choice(valid_ids))


def load_model(path: str, device: torch.device) -> CRZeroNet:
    model = CRZeroNet()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def render_game_state(game: CRGame) -> str:
    """Simple text rendering of the arena."""
    grid = [["." for _ in range(ARENA_W)] for _ in range(ARENA_H)]

    for e in game.entities:
        if not e.alive:
            continue
        x, y = int(round(e.x)), int(round(e.y))
        x = max(0, min(ARENA_W - 1, x))
        y = max(0, min(ARENA_H - 1, y))

        if e.is_king_tower:
            grid[y][x] = "K" if e.owner == 0 else "k"
        elif e.is_tower:
            grid[y][x] = "P" if e.owner == 0 else "p"
        else:
            name = e.card_type.name[0]
            grid[y][x] = name.upper() if e.owner == 0 else name.lower()

    # River
    from crsim.constants import BRIDGE_LEFT_COLS, BRIDGE_RIGHT_COLS, RIVER_ROW_HI, RIVER_ROW_LO
    for y in (RIVER_ROW_LO, RIVER_ROW_HI):
        for x in range(ARENA_W):
            if grid[y][x] == ".":
                is_bridge = (
                    BRIDGE_LEFT_COLS[0] <= x <= BRIDGE_LEFT_COLS[1]
                    or BRIDGE_RIGHT_COLS[0] <= x <= BRIDGE_RIGHT_COLS[1]
                )
                grid[y][x] = "=" if is_bridge else "~"

    lines = []
    for y in range(ARENA_H - 1, -1, -1):
        lines.append(f"{y:2d} {''.join(grid[y])}")

    # Status bar
    lines.append(f"   {''.join(str(x % 10) for x in range(ARENA_W))}")
    lines.append(
        f"Tick {game.tick_count} | Phase {game.phase.name} | "
        f"P0 elixir={game.players[0].elixir:.1f} | "
        f"P1 elixir={game.players[1].elixir:.1f}"
    )

    return "\n".join(lines)


def play_game(
    player_a,
    player_b,
    game: CRGame,
    watch: bool = False,
) -> GameResult:
    """Play a full game."""
    while not game.done:
        if isinstance(player_a, RandomPlayer):
            a0 = player_a.select_action(game, 0)
        else:
            a0, _ = player_a.select_action(game, 0, deterministic=True)

        if isinstance(player_b, RandomPlayer):
            a1 = player_b.select_action(game, 1)
        else:
            a1, _ = player_b.select_action(game, 1, deterministic=True)

        action_0 = _action_id_to_action(a0, 0)
        action_1 = _action_id_to_action(a1, 1)
        game.step([action_0, action_1])

        if watch and game.tick_count % 10 == 0:
            print("\033c" + render_game_state(game))

    if watch:
        print(render_game_state(game))
        print(f"\nResult: {game.result.name}")

    return game.result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent", type=str, default=None)
    parser.add_argument("--baseline", choices=["random"], default=None)
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--mcts-sims", type=int, default=200)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_a = load_model(args.checkpoint, device)
    mcts_cfg = MCTSConfig(n_simulations=args.mcts_sims, temperature=0.1)
    player_a = MCTSPlayer(model=model_a, config=mcts_cfg, device=device)

    if args.baseline == "random":
        player_b = RandomPlayer()
    elif args.opponent:
        model_b = load_model(args.opponent, device)
        player_b = MCTSPlayer(model=model_b, config=mcts_cfg, device=device)
    else:
        # Self-play evaluation
        player_b = MCTSPlayer(model=model_a, config=mcts_cfg, device=device)

    if args.watch:
        game = CRGame(seed=42)
        play_game(player_a, player_b, game, watch=True)
        return

    wins, losses, draws = 0, 0, 0
    for i in range(args.n_games):
        game = CRGame(seed=i)
        result = play_game(player_a, player_b, game)

        if result == GameResult.P0_WIN:
            wins += 1
        elif result == GameResult.P1_WIN:
            losses += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            logger.info(
                "Game %d/%d: W=%d L=%d D=%d (winrate=%.1f%%)",
                i + 1, args.n_games, wins, losses, draws,
                100 * wins / (i + 1),
            )

    print(f"\nFinal: {wins}W / {losses}L / {draws}D out of {args.n_games} games")
    print(f"Win rate: {100 * wins / args.n_games:.1f}%")


if __name__ == "__main__":
    main()
