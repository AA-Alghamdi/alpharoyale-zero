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
from crsim.constants import (
    DOUBLE_ELIXIR_START_TICKS,
    GAMMA,
    TERMINAL_CROWN_SCALE,
    W_LEAK,
)
from crsim.game import Action, CRGame, GamePhase
from mcts.gumbel_search import GumbelConfig, GumbelMuZeroSearch, _action_id_to_action
from model.features import (
    encode_state,
    extract_auxiliary_targets,
    extract_entity_features,
)
from training.curriculum import CurriculumManager
from training.domain_randomization import DomainRandomizer
from training.league import League
from training.opponent_model import CardTracker, ElixirTracker
from training.replay_buffer import ReplayBuffer, ReplayEntry

logger = logging.getLogger(__name__)


def _phase_str(game: CRGame) -> str:
    """Phase string for ElixirTracker, including the in-regulation 2x window."""
    if game.phase == GamePhase.OVERTIME:
        return "overtime"
    if game.phase == GamePhase.SUDDEN_DEATH:
        return "sudden_death"
    if game.tick_count >= DOUBLE_ELIXIR_START_TICKS:
        return "double"
    return "normal"


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
    # Decision cadence: run MCTS search once every N game ticks instead of every
    # tick. The simulator still advances every 50ms tick (so combat resolves
    # normally), but the agent only makes a placement decision every
    # decision_interval_ticks. Real Clash Royale bots act at ~5 decisions/sec;
    # searching all 20 ticks/sec is both unrealistic and ~N times more expensive.
    decision_interval_ticks: int = 8
    # Optional hard cap on simulated ticks per game (None = use real time limit).
    # Useful for fast CPU smoke tests; the terminal value is taken from the
    # game's reward at the truncation point.
    max_ticks: int | None = None


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

        # Keep the MCTS rollout horizon aligned with how often the agent
        # actually re-decides, so each simulation looks a full decision window
        # ahead rather than a single 50ms tick.
        self.config.gumbel_config.rollout_ticks = self.config.decision_interval_ticks

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
        decision_interval = max(1, cfg.decision_interval_ticks)

        while not game.done:
            # Truncation for fast smoke runs.
            if cfg.max_ticks is not None and game.tick_count >= cfg.max_ticks:
                break

            # Only run search (and record a training position) on decision
            # ticks; otherwise advance the simulator with WAIT so combat still
            # resolves every tick. This is the difference between ~14k searches
            # per game and a tractable number.
            is_decision_tick = (game.tick_count % decision_interval) == 0

            if not is_decision_tick:
                game.step([
                    Action(player=0, hand_slot=-1),
                    Action(player=1, hand_slot=-1),
                ])
                continue

            actions_for_step = []

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
                    # reward-shaping snapshots (read-only)
                    "phi": game.potential(player),
                    "tick": game.tick_count,
                    "leak_at": game.players[player]._leaked_elixir,
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
                        card_def = CARD_DEFS.get(played_card_type)
                        if card_def is not None:
                            elixir_trackers[opponent].observe_play(
                                card_def.cost, game.tick_count, _phase_str(game)
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

        # Build entries with a discounted-to-terminal value via potential-based
        # shaping (Ng et al. 1999): each stored position gets a value target
        # reflecting how the board actually evolved after that move — not one
        # global +/-1 broadcast to all ~70-140 positions. Potential-based form
        # keeps the optimal policy identical to pure win/loss. The leak term is
        # the single non-potential penalty (and it rewards correct play: spending
        # elixir to avoid a leak is exactly what wins, so it can't be farmed).
        # NOTE: flip-augmentation was removed (it paired a flipped board with the
        # opponent's reward + the original player's policy/entity tensors). The
        # per-player pass below already records both perspectives correctly.
        entries: list[ReplayEntry] = []

        def _build_for(persp: int) -> None:
            steps = [s for s in trajectory if s["player"] == persp]
            if not steps:
                return
            final_aux = final_aux_0 if persp == 0 else final_aux_1
            return_value = game.get_reward(persp) * TERMINAL_CROWN_SCALE
            built: list[ReplayEntry] = []
            for i in range(len(steps) - 1, -1, -1):
                s = steps[i]
                nxt = steps[i + 1] if i + 1 < len(steps) else None
                if nxt is not None:
                    dt = max(nxt["tick"] - s["tick"], 1)
                    g_dt = GAMMA ** dt
                    leak_iv = max(nxt["leak_at"] - s["leak_at"], 0.0)
                    r_shape = g_dt * nxt["phi"] - s["phi"] - W_LEAK * leak_iv
                    return_value = r_shape + g_dt * return_value
                value = float(np.clip(return_value, -1.0, 1.0))
                built.append(ReplayEntry(
                    spatial=s["spatial"],
                    scalar=s["scalar"],
                    policy=s["policy"],
                    value=value,
                    entity_features=s["entity_features"],
                    entity_mask=s["entity_mask"],
                    crown_target=int(final_aux["crown_target"]),
                    tower_hp_target=final_aux["tower_hp_target"],
                    game_length_target=s["game_length_target"],
                ))
            built.reverse()
            entries.extend(built)

        for persp in (0, 1):
            _build_for(persp)

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
