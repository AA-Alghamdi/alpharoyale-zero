#!/usr/bin/env python3
"""Interactive play: watch the AI play or play against it (text mode).

Usage:
    # Watch AI vs AI
    python scripts/play.py --checkpoint checkpoints/best.pt --mode watch

    # Play against the AI (you control player 0)
    python scripts/play.py --checkpoint checkpoints/best.pt --mode interactive
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from crsim.cards import CARD_DEFS
from crsim.constants import ARENA_H, ARENA_W
from crsim.game import Action, CRGame
from mcts.search import MCTSConfig, MCTSPlayer, _action_id_to_action
from scripts.evaluate import load_model, render_game_state


def interactive_game(model_path: str, device: torch.device) -> None:
    """Play interactively against the AI."""
    model = load_model(model_path, device)
    mcts_cfg = MCTSConfig(n_simulations=200, temperature=0.1)
    ai_player = MCTSPlayer(model=model, config=mcts_cfg, device=device)

    game = CRGame(seed=42)

    print("=== ClashRoyale-Zero Interactive Mode ===")
    print("You are Player 0 (bottom). AI is Player 1 (top).")
    print("Commands: 'w' = wait, '<slot> <x> <y>' = play card, 'q' = quit\n")

    while not game.done:
        print("\033c" + render_game_state(game))
        print()

        # Show hand
        ps = game.players[0]
        for slot in range(4):
            card_idx = ps.hand[slot]
            card_type = ps.deck[card_idx]
            card_def = CARD_DEFS[card_type]
            affordable = "✓" if ps.elixir >= card_def.cost else "✗"
            print(f"  [{slot}] {card_type.name} (cost={card_def.cost}) {affordable}")

        print(f"\n  Elixir: {ps.elixir:.1f}")

        # Get human input
        while True:
            try:
                cmd = input("\nAction> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nQuitting.")
                return

            if cmd == "q":
                return
            if cmd == "w":
                human_action = Action(player=0, hand_slot=-1)
                break

            parts = cmd.split()
            if len(parts) == 3:
                try:
                    slot, x, y = int(parts[0]), int(parts[1]), int(parts[2])
                    if 0 <= slot < 4 and 0 <= x < ARENA_W and 0 <= y < ARENA_H:
                        human_action = Action(player=0, hand_slot=slot, x=float(x), y=float(y))
                        break
                except ValueError:
                    pass

            print("Invalid. Enter 'w' to wait, or '<slot> <x> <y>' (e.g., '0 9 5')")

        # AI turn
        ai_action_id, _ = ai_player.select_action(game, 1, deterministic=True)
        ai_action = _action_id_to_action(ai_action_id, 1)

        game.step([human_action, ai_action])

    print("\033c" + render_game_state(game))
    print(f"\n{'='*40}")
    print(f"Game Over: {game.result.name}")


def watch_game(model_path: str, device: torch.device, delay: float = 0.2) -> None:
    """Watch the AI play against itself."""
    model = load_model(model_path, device)
    mcts_cfg = MCTSConfig(n_simulations=200, temperature=0.3)
    player = MCTSPlayer(model=model, config=mcts_cfg, device=device)

    game = CRGame(seed=42)

    while not game.done:
        a0, _ = player.select_action(game, 0, deterministic=False)
        a1, _ = player.select_action(game, 1, deterministic=False)

        game.step([
            _action_id_to_action(a0, 0),
            _action_id_to_action(a1, 1),
        ])

        if game.tick_count % 5 == 0:
            print("\033c" + render_game_state(game))
            time.sleep(delay)

    print("\033c" + render_game_state(game))
    print(f"\nResult: {game.result.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["watch", "interactive"], default="watch")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--delay", type=float, default=0.2, help="Delay between frames (watch mode)"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == "interactive":
        interactive_game(args.checkpoint, device)
    else:
        watch_game(args.checkpoint, device, args.delay)


if __name__ == "__main__":
    main()
