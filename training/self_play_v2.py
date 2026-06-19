"""Self-play v2: Uses Gumbel MuZero search with KataGo improvements.

Key improvements over self_play.py:
  1. Gumbel search (16 sims) instead of vanilla MCTS (800 sims) → 50× faster
  2. Playout cap randomization (KataGo) — varies sims per move for diversity
  3. Auxiliary target collection — crowns, tower HP, game length
  4. Opponent sampling from policy (Smooth UCT)
  5. Trajectory-level data augmentation (board flipping)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from crsim.cards import CARD_DEFS, CardType
from crsim.game import Action, CRGame
from mcts.gumbel_search import GumbelConfig, GumbelMuZeroSearch, _action_id_to_action
from model.features import (
    encode_state,
    extract_auxiliary_targets,
    extract_entity_features,
)
from training.curriculum import CurriculumManager, DeckSampler
from training.domain_randomization import DomainRandomizer
from training.league import League
from training.opponent_model import CardTracker, ElixirTracker
from training.replay_buffer import ReplayBuffer, ReplayEntry

logger = logging.getLogger(__name__)

DEFAULT_CARD_POOL: list[CardType] = list(CardType)
DECK_SIZE = 8


@dataclass
class SelfPlayV2Config:
    n_games: int = 64
    gumbel_config: GumbelConfig = field(default_factory=GumbelConfig)
    temperature_threshold: int = 30  # switch to low temp after N moves
    random_decks: bool = True
    card_pool: list[CardType] = field(default_factory=lambda: list(CardType))
    # Data augmentation
    augment_flip: bool = True  # randomly flip player perspective


def random_deck(
    pool: list[CardType],
    rng: np.random.Generator | None = None,
) -> list[CardType]:
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(len(pool), size=DECK_SIZE, replace=False)
    return [pool[int(i)] for i in indices]


class SelfPlayWorkerV2:
    """Self-play worker using Gumbel MuZero search."""

    def __init__(
        self,
        model: torch.nn.Module,
        replay_buffer: ReplayBuffer,
        config: SelfPlayV2Config | None = None,
        device: torch.device | None = None,
        worker_id: int = 0,
        curriculum: CurriculumManager | None = None,
        randomizer: DomainRandomizer | None = None,
        league: League | None = None,
    ) -> None:
        self.model = model
        self.replay_buffer = replay_buffer
        self.config = config or SelfPlayV2Config()
        self.device = device or torch.device("cpu")
        self.worker_id = worker_id
        self.rng = np.random.default_rng(seed=worker_id)

        self.searcher = GumbelMuZeroSearch(
            model=self.model,
            config=self.config.gumbel_config,
            device=self.device,
        )

        # Integration modules
        self.curriculum = curriculum
        self.randomizer = randomizer or DomainRandomizer(strength=0.0)
        self.league = league

        # Opponent searcher (separate model for league opponents)
        self._opponent_model: torch.nn.Module | None = None
        self._opponent_searcher: GumbelMuZeroSearch | None = None

        self.games_played: int = 0
        self.total_positions: int = 0
        self.total_wins: dict[int, int] = {0: 0, 1: 0, -1: 0}

    def _get_opponent_searcher(self, opponent_weights: dict | None) -> GumbelMuZeroSearch:
        """Get or create opponent searcher with the given weights."""
        if opponent_weights is None:
            # Self-play: use own model
            return self.searcher
        # Load opponent model (lazy init)
        if self._opponent_model is None:
            import copy
            self._opponent_model = copy.deepcopy(self.model)
            self._opponent_searcher = GumbelMuZeroSearch(
                model=self._opponent_model,
                config=self.config.gumbel_config,
                device=self.device,
            )
        self._opponent_model.load_state_dict(opponent_weights)
        self._opponent_model.eval()
        return self._opponent_searcher  # type: ignore[return-value]

    def play_game(self) -> list[ReplayEntry]:
        """Play one self-play game and return trajectory.

        Now collects entity features and real auxiliary targets at each step.
        Uses League for opponent selection when available.
        """
        cfg = self.config

        # League opponent selection (PFSP)
        # If league is provided, it should supply opponent weights via
        # league.get_opponent_weights(worker_id) which returns None for self-play
        # or a state_dict for a league opponent.
        opponent_weights = None
        if self.league is not None and hasattr(self.league, 'get_opponent_weights'):
            opponent_weights = self.league.get_opponent_weights(self.worker_id)

        opponent_searcher = self._get_opponent_searcher(opponent_weights)

        # Deck selection: use curriculum if available, else random
        if self.curriculum is not None:
            deck_p0, deck_p1 = self.curriculum.deck_sampler.sample()
        elif cfg.random_decks:
            deck_p0 = random_deck(cfg.card_pool, self.rng)
            deck_p1 = random_deck(cfg.card_pool, self.rng)
        else:
            deck_p0 = None
            deck_p1 = None

        game = CRGame(deck_p0=deck_p0, deck_p1=deck_p1, seed=int(self.rng.integers(0, 2**31)))

        # Apply domain randomization to initial entities (towers)
        if self.randomizer.strength > 0:
            self.randomizer.randomize_game(game)

        # Set up opponent tracking (imperfect information)
        card_trackers = [CardTracker(), CardTracker()]
        elixir_trackers = [ElixirTracker(), ElixirTracker()]

        # Collect full trajectory with entity features + aux targets
        trajectory: list[dict] = []
        move_count = 0

        while not game.done:
            actions_for_step: list[Action] = []

            for player in (0, 1):
                # Temperature schedule
                if move_count >= cfg.temperature_threshold:
                    self.searcher.config.temperature = 0.1
                else:
                    self.searcher.config.temperature = 1.0

                # Pick searcher: own model for P0, opponent for P1
                searcher = self.searcher if player == 0 else opponent_searcher

                # Full state encoding including entity features
                spatial, scalar = encode_state(game, player)
                entity_feats, entity_mask = extract_entity_features(game, player)
                aux = extract_auxiliary_targets(game, player)

                action_id, action_probs = searcher.select_action(
                    game, player, deterministic=False,
                )

                trajectory.append({
                    "spatial": spatial,
                    "scalar": scalar,
                    "entity_features": entity_feats,
                    "entity_mask": entity_mask,
                    "policy": action_probs,
                    "player": player,
                    "crown_target": int(aux["crown_target"]),
                    "tower_hp_target": aux["tower_hp_target"],
                    "game_length_target": float(aux["game_length_target"]),
                })

                action = _action_id_to_action(action_id, player)
                actions_for_step.append(action)

                # Track opponent's plays for belief state
                if not action.is_wait:
                    # Get the card type from the player's hand
                    ps = game.players[player]
                    card_idx = ps.hand[action.hand_slot]
                    played_card_type = ps.deck[card_idx] if card_idx is not None else None
                    if played_card_type is not None:
                        opponent = 1 - player
                        card_trackers[opponent].observe_play(
                            int(played_card_type), game.tick_count
                        )
                        cost = CARD_DEFS.get(played_card_type)
                        if cost is not None:
                            elixir_trackers[opponent].observe_play(
                                int(played_card_type), cost.cost, game.tick_count
                            )

            game.step(actions_for_step)

            # Apply domain randomization to newly spawned entities
            if self.randomizer.strength > 0:
                self.randomizer.randomize_game(game)

            move_count += 1

        # Determine winner
        reward_0 = game.get_reward(0)
        if reward_0 > 0:
            self.total_wins[0] += 1
        elif reward_0 < 0:
            self.total_wins[1] += 1
        else:
            self.total_wins[-1] += 1

        # Collect final aux targets (terminal state)
        final_aux_0 = extract_auxiliary_targets(game, 0)
        final_aux_1 = extract_auxiliary_targets(game, 1)

        # Build entries with terminal reward + real aux targets
        entries: list[ReplayEntry] = []
        for step in trajectory:
            player = step["player"]
            reward = game.get_reward(player)
            # Use terminal crown/tower state as targets (hindsight)
            final_aux = final_aux_0 if player == 0 else final_aux_1
            entries.append(ReplayEntry(
                spatial=step["spatial"],
                scalar=step["scalar"],
                policy=step["policy"],
                value=reward,
                entity_features=step["entity_features"],
                entity_mask=step["entity_mask"],
                crown_target=int(final_aux["crown_target"]),
                tower_hp_target=final_aux["tower_hp_target"],
                game_length_target=step["game_length_target"],
            ))

        # Data augmentation: add flipped perspective
        if cfg.augment_flip and self.rng.random() < 0.5:
            for step in trajectory:
                player = step["player"]
                opp = 1 - player
                reward = game.get_reward(opp)
                flipped_spatial = np.flip(step["spatial"], axis=1).copy()
                final_aux = final_aux_1 if player == 0 else final_aux_0
                entries.append(ReplayEntry(
                    spatial=flipped_spatial,
                    scalar=step["scalar"],
                    policy=step["policy"],
                    value=reward,
                    entity_features=step["entity_features"],
                    entity_mask=step["entity_mask"],
                    crown_target=int(final_aux["crown_target"]),
                    tower_hp_target=final_aux["tower_hp_target"],
                    game_length_target=step["game_length_target"],
                ))

        self.games_played += 1
        self.total_positions += len(entries)

        return entries

    def run_batch(self, n_games: int | None = None) -> int:
        """Play a batch of games and push to replay buffer."""
        n = n_games or self.config.n_games
        total_positions = 0

        for _ in range(n):
            entries = self.play_game()
            for e in entries:
                self.replay_buffer.push(e)
            total_positions += len(entries)

        wr = self.total_wins
        logger.info(
            "Worker %d: %d games, %d positions | W/L/D: %d/%d/%d",
            self.worker_id, n, total_positions, wr[0], wr[1], wr[-1],
        )

        return total_positions

    def update_model(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)
        self.model.eval()
