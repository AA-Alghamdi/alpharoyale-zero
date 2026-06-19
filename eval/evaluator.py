"""Model evaluation: pit two models against each other and track Elo."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from crsim.game import CRGame, GameResult
from mcts.search import MCTSConfig, MCTSPlayer, _action_id_to_action
from model.network import CRZeroNet

logger = logging.getLogger(__name__)


@dataclass
class EloRating:
    rating: float = 1000.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def expected_score(self, opponent: EloRating) -> float:
        return 1.0 / (1.0 + 10.0 ** ((opponent.rating - self.rating) / 400.0))

    def update(self, actual_score: float, opponent: EloRating, k: float = 32.0) -> None:
        expected = self.expected_score(opponent)
        self.rating += k * (actual_score - expected)
        self.games_played += 1
        if actual_score > 0.5:
            self.wins += 1
        elif actual_score < 0.5:
            self.losses += 1
        else:
            self.draws += 1


class Evaluator:
    """Evaluate models by playing games between them."""

    def __init__(
        self,
        device: torch.device | None = None,
        mcts_simulations: int = 200,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.mcts_simulations = mcts_simulations
        self.elo_history: list[tuple[int, float]] = []  # (step, elo)

    def play_match(
        self,
        model_a: CRZeroNet,
        model_b: CRZeroNet,
        n_games: int = 100,
    ) -> tuple[int, int, int]:
        """Play n_games between model_a (player 0) and model_b (player 1).

        Returns (a_wins, b_wins, draws).
        """
        mcts_cfg = MCTSConfig(
            n_simulations=self.mcts_simulations,
            temperature=0.1,  # near-greedy for evaluation
        )

        player_a = MCTSPlayer(model=model_a, config=mcts_cfg, device=self.device)
        player_b = MCTSPlayer(model=model_b, config=mcts_cfg, device=self.device)

        a_wins, b_wins, draws = 0, 0, 0

        for game_idx in range(n_games):
            # Alternate sides
            if game_idx % 2 == 0:
                p0_player, p1_player = player_a, player_b
                a_is_p0 = True
            else:
                p0_player, p1_player = player_b, player_a
                a_is_p0 = False

            game = CRGame(seed=game_idx)
            move = 0

            while not game.done:
                a0_id, _ = p0_player.select_action(game, 0, deterministic=True)
                a1_id, _ = p1_player.select_action(game, 1, deterministic=True)

                action_0 = _action_id_to_action(a0_id, 0)
                action_1 = _action_id_to_action(a1_id, 1)
                game.step([action_0, action_1])
                move += 1

            # Determine result
            if game.result == GameResult.P0_WIN:
                if a_is_p0:
                    a_wins += 1
                else:
                    b_wins += 1
            elif game.result == GameResult.P1_WIN:
                if a_is_p0:
                    b_wins += 1
                else:
                    a_wins += 1
            else:
                draws += 1

        logger.info(
            "Match result: A wins=%d, B wins=%d, draws=%d (A winrate=%.1f%%)",
            a_wins, b_wins, draws,
            100 * a_wins / max(n_games, 1),
        )

        return a_wins, b_wins, draws

    def evaluate_checkpoint(
        self,
        new_model: CRZeroNet,
        best_model: CRZeroNet,
        step: int,
        n_games: int = 100,
        promotion_threshold: float = 0.55,
    ) -> bool:
        """Evaluate new_model against best_model.

        Returns True if new_model should be promoted (win rate > threshold).
        """
        a_wins, b_wins, draws = self.play_match(new_model, best_model, n_games)

        total = a_wins + b_wins + draws
        win_rate = a_wins / max(total, 1)

        # Update Elo
        elo_new = EloRating(rating=1000.0)
        elo_best = EloRating(rating=1000.0)
        if self.elo_history:
            elo_best.rating = self.elo_history[-1][1]
            elo_new.rating = elo_best.rating

        for _ in range(a_wins):
            elo_new.update(1.0, elo_best)
            elo_best.update(0.0, elo_new)
        for _ in range(b_wins):
            elo_new.update(0.0, elo_best)
            elo_best.update(1.0, elo_new)
        for _ in range(draws):
            elo_new.update(0.5, elo_best)
            elo_best.update(0.5, elo_new)

        promoted = win_rate >= promotion_threshold
        self.elo_history.append((step, elo_new.rating if promoted else elo_best.rating))

        logger.info(
            "Evaluation at step %d: win_rate=%.1f%% elo=%.0f promoted=%s",
            step,
            win_rate * 100,
            elo_new.rating,
            promoted,
        )

        return promoted
