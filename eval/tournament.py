"""ELO tournament system for checkpoint evaluation.

Runs round-robin tournaments between checkpoints to track training
progress and select the best model. Supports multiple deck matchups.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from crsim.game import CRGame, GameResult
from eval.evaluator import EloRating
from training.curriculum import DECK_ARCHETYPES

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single game."""
    player_a: str
    player_b: str
    deck_a: str
    deck_b: str
    winner: str  # "a", "b", or "draw"
    crowns_a: int = 0
    crowns_b: int = 0
    ticks: int = 0
    timestamp: float = 0.0


@dataclass
class TournamentConfig:
    n_games_per_matchup: int = 10
    n_deck_matchups: int = 5
    mcts_simulations: int = 16
    temperature: float = 0.1
    max_ticks_per_game: int = 1080
    checkpoint_dir: str = "checkpoints"
    results_path: str = "eval/tournament_results.json"


class Tournament:
    """Round-robin tournament between model checkpoints.

    Tracks ELO ratings across matchups with different deck archetypes.
    """

    def __init__(self, config: TournamentConfig | None = None) -> None:
        self.config = config or TournamentConfig()
        self.ratings: dict[str, EloRating] = {}
        self.match_history: list[MatchResult] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_player(self, name: str, initial_elo: float = 1000.0) -> None:
        if name not in self.ratings:
            self.ratings[name] = EloRating(rating=initial_elo)

    def run_match(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        name_a: str,
        name_b: str,
    ) -> tuple[int, int, int]:
        """Play a match between two models across multiple deck matchups.

        Returns (a_wins, b_wins, draws) total.
        """
        from mcts.gumbel_search import GumbelConfig, GumbelMuZeroSearch

        gumbel_cfg = GumbelConfig(
            n_simulations=self.config.mcts_simulations,
            temperature=self.config.temperature,
        )
        searcher_a = GumbelMuZeroSearch(
            model=model_a, config=gumbel_cfg, device=self.device,
        )
        searcher_b = GumbelMuZeroSearch(
            model=model_b, config=gumbel_cfg, device=self.device,
        )

        self.add_player(name_a)
        self.add_player(name_b)

        total_a_wins, total_b_wins, total_draws = 0, 0, 0

        # Select deck matchups
        n_matchups = min(self.config.n_deck_matchups, len(DECK_ARCHETYPES))
        archetype_indices = np.random.choice(
            len(DECK_ARCHETYPES), size=n_matchups * 2, replace=True,
        )

        for m_idx in range(n_matchups):
            deck_a_name, deck_a = DECK_ARCHETYPES[archetype_indices[m_idx * 2]]
            deck_b_name, deck_b = DECK_ARCHETYPES[archetype_indices[m_idx * 2 + 1]]

            for game_idx in range(self.config.n_games_per_matchup):
                # Alternate sides each game
                if game_idx % 2 == 0:
                    p0_searcher, p1_searcher = searcher_a, searcher_b
                    a_is_p0 = True
                else:
                    p0_searcher, p1_searcher = searcher_b, searcher_a
                    a_is_p0 = False

                game = CRGame(
                    deck_p0=list(deck_a),
                    deck_p1=list(deck_b),
                    seed=game_idx + m_idx * 1000,
                )

                while not game.done:
                    from mcts.gumbel_search import _action_id_to_action

                    a0_id, _ = p0_searcher.select_action(
                        game, 0, deterministic=True,
                    )
                    a1_id, _ = p1_searcher.select_action(
                        game, 1, deterministic=True,
                    )
                    game.step([
                        _action_id_to_action(a0_id, 0),
                        _action_id_to_action(a1_id, 1),
                    ])

                # Record result
                if game.result == GameResult.P0_WIN:
                    winner = "a" if a_is_p0 else "b"
                elif game.result == GameResult.P1_WIN:
                    winner = "b" if a_is_p0 else "a"
                else:
                    winner = "draw"

                if winner == "a":
                    total_a_wins += 1
                    self.ratings[name_a].update(1.0, self.ratings[name_b])
                    self.ratings[name_b].update(0.0, self.ratings[name_a])
                elif winner == "b":
                    total_b_wins += 1
                    self.ratings[name_b].update(1.0, self.ratings[name_a])
                    self.ratings[name_a].update(0.0, self.ratings[name_b])
                else:
                    total_draws += 1
                    self.ratings[name_a].update(0.5, self.ratings[name_b])
                    self.ratings[name_b].update(0.5, self.ratings[name_a])

                self.match_history.append(MatchResult(
                    player_a=name_a,
                    player_b=name_b,
                    deck_a=deck_a_name,
                    deck_b=deck_b_name,
                    winner=winner,
                    crowns_a=game._count_crowns(0 if a_is_p0 else 1),
                    crowns_b=game._count_crowns(1 if a_is_p0 else 0),
                    ticks=game.tick_count,
                    timestamp=time.time(),
                ))

        logger.info(
            "Match %s vs %s: %d-%d-%d (ELO: %s=%.0f, %s=%.0f)",
            name_a, name_b,
            total_a_wins, total_b_wins, total_draws,
            name_a, self.ratings[name_a].rating,
            name_b, self.ratings[name_b].rating,
        )

        return total_a_wins, total_b_wins, total_draws

    def run_tournament(
        self,
        models: dict[str, torch.nn.Module],
    ) -> dict[str, float]:
        """Run round-robin tournament between all models.

        Parameters
        ----------
        models : dict mapping name → model

        Returns
        -------
        dict mapping name → final ELO rating
        """
        names = list(models.keys())
        logger.info("Starting tournament with %d players: %s", len(names), names)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self.run_match(
                    models[names[i]], models[names[j]],
                    names[i], names[j],
                )

        # Sort by ELO
        rankings = sorted(
            self.ratings.items(),
            key=lambda x: x[1].rating,
            reverse=True,
        )

        logger.info("=== Tournament Results ===")
        for rank, (name, elo) in enumerate(rankings, 1):
            logger.info(
                "#%d %s: ELO=%.0f W=%d L=%d D=%d",
                rank, name, elo.rating, elo.wins, elo.losses, elo.draws,
            )

        return {name: elo.rating for name, elo in self.ratings.items()}

    def load_checkpoint_models(
        self,
        checkpoint_dir: str | None = None,
        model_class: type | None = None,
    ) -> dict[str, torch.nn.Module]:
        """Load all checkpoint models from directory.

        Returns dict mapping checkpoint name → model.
        """
        from model.transformer_net import CRStarNet

        cls = model_class or CRStarNet
        cp_dir = Path(checkpoint_dir or self.config.checkpoint_dir)
        models = {}

        for cp_path in sorted(cp_dir.glob("*.pt")):
            try:
                state = torch.load(cp_path, map_location=self.device, weights_only=True)
                model = cls()
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                else:
                    model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                models[cp_path.stem] = model
                logger.info("Loaded checkpoint: %s", cp_path.stem)
            except Exception as e:
                logger.warning("Failed to load %s: %s", cp_path, e)

        return models

    def save_results(self, path: str | None = None) -> None:
        """Save tournament results to JSON."""
        out_path = Path(path or self.config.results_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "ratings": {
                name: {
                    "elo": elo.rating,
                    "games": elo.games_played,
                    "wins": elo.wins,
                    "losses": elo.losses,
                    "draws": elo.draws,
                }
                for name, elo in self.ratings.items()
            },
            "matches": [
                {
                    "player_a": m.player_a,
                    "player_b": m.player_b,
                    "deck_a": m.deck_a,
                    "deck_b": m.deck_b,
                    "winner": m.winner,
                    "crowns_a": m.crowns_a,
                    "crowns_b": m.crowns_b,
                    "ticks": m.ticks,
                    "timestamp": m.timestamp,
                }
                for m in self.match_history
            ],
        }

        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Saved tournament results to %s", out_path)
