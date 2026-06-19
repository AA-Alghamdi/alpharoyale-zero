"""Self-play worker: generates training data by playing games against itself.

Each worker:
  1. Loads the latest model weights from the parameter server
  2. Plays N parallel games using MCTS
  3. Pushes (state, policy, outcome) tuples to the shared replay buffer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from crsim.cards import CardType
from crsim.game import Action, CRGame
from mcts.search import MCTSConfig, MCTSPlayer, _action_id_to_action
from model.features import encode_state
from model.network import CRZeroNet
from training.replay_buffer import ReplayBuffer, ReplayEntry

logger = logging.getLogger(__name__)

# Default deck pool for random deck selection
DEFAULT_CARD_POOL: list[CardType] = list(CardType)
DECK_SIZE = 8


@dataclass
class SelfPlayConfig:
    n_games: int = 64  # games per batch
    mcts_config: MCTSConfig = field(default_factory=MCTSConfig)
    temperature_threshold: int = 20  # switch to greedy after this many moves
    random_decks: bool = True
    card_pool: list[CardType] = field(default_factory=lambda: list(CardType))


def random_deck(
    pool: list[CardType],
    rng: np.random.Generator | None = None,
) -> list[CardType]:
    """Sample a random 8-card deck from the pool without replacement."""
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(len(pool), size=DECK_SIZE, replace=False)
    return [pool[int(i)] for i in indices]


class SelfPlayWorker:
    """Generates self-play games and pushes experience to the replay buffer."""

    def __init__(
        self,
        model: CRZeroNet,
        replay_buffer: ReplayBuffer,
        config: SelfPlayConfig | None = None,
        device: torch.device | None = None,
        worker_id: int = 0,
    ) -> None:
        self.model = model
        self.replay_buffer = replay_buffer
        self.config = config or SelfPlayConfig()
        self.device = device or torch.device("cpu")
        self.worker_id = worker_id
        self.rng = np.random.default_rng(seed=worker_id)

        self.mcts_player = MCTSPlayer(
            model=self.model,
            config=self.config.mcts_config,
            device=self.device,
        )

        # Stats
        self.games_played: int = 0
        self.total_positions: int = 0

    def play_game(self) -> list[ReplayEntry]:
        """Play a single self-play game and return the trajectory."""
        cfg = self.config

        # Build decks
        if cfg.random_decks:
            deck_p0 = random_deck(cfg.card_pool, self.rng)
            deck_p1 = random_deck(cfg.card_pool, self.rng)
        else:
            deck_p0 = None  # use default
            deck_p1 = None

        game = CRGame(deck_p0=deck_p0, deck_p1=deck_p1, seed=int(self.rng.integers(0, 2**31)))

        trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        # Each entry: (spatial, scalar, mcts_policy, player)

        move_count = 0

        while not game.done:
            actions_for_step: list[Action] = []

            for player in (0, 1):
                # Adjust temperature
                if move_count >= cfg.temperature_threshold:
                    self.mcts_player.config.temperature = 0.1
                else:
                    self.mcts_player.config.temperature = 1.0

                # Encode state and run MCTS
                spatial, scalar = encode_state(game, player)
                action_id, action_probs = self.mcts_player.select_action(
                    game, player, deterministic=False
                )

                # Store experience
                trajectory.append((spatial, scalar, action_probs, player))

                # Convert to action
                action = _action_id_to_action(action_id, player)
                actions_for_step.append(action)

            # Step the game with both actions
            game.step(actions_for_step)
            move_count += 1

        # Assign terminal rewards
        entries: list[ReplayEntry] = []
        for spatial, scalar, policy, player in trajectory:
            reward = game.get_reward(player)
            entries.append(ReplayEntry(
                spatial=spatial,
                scalar=scalar,
                policy=policy,
                value=reward,
            ))

        self.games_played += 1
        self.total_positions += len(entries)

        return entries

    def run_batch(self, n_games: int | None = None) -> int:
        """Play a batch of games and push all experience to the replay buffer.

        Returns the number of positions generated.
        """
        n = n_games or self.config.n_games
        total_positions = 0

        for i in range(n):
            entries = self.play_game()
            for e in entries:
                self.replay_buffer.push(e)
            total_positions += len(entries)

        logger.info(
            "Worker %d: played %d games, generated %d positions (total: %d games, %d positions)",
            self.worker_id,
            n,
            total_positions,
            self.games_played,
            self.total_positions,
        )

        return total_positions

    def update_model(self, state_dict: dict) -> None:
        """Load new model weights from the parameter server."""
        self.model.load_state_dict(state_dict)
        self.model.eval()
