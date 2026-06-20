"""Vectorized self-play — run many games in lockstep with batched NN evals.

``SelfPlayWorkerV2`` plays games one at a time, and within a game evaluates one
state per neural-network forward pass. That is fine on CPU but leaves a GPU
almost idle: the forward pass for a batch of 1 costs nearly as much wall-clock
time as a batch of hundreds.

``VectorizedSelfPlayWorker`` plays ``n_envs`` games simultaneously and steps
them in lockstep. On every decision tick it gathers the ``(game, player)``
states across *all* live games and runs them through :class:`BatchedGumbelSearch`,
so each tree search — and each leaf evaluation inside it — is one big batched
forward pass instead of ``2 * n_envs`` tiny ones. This is the primitive that
turns a GPU into thousands of self-play games per hour (see ``SCALE-UP.md``).

It produces the *exact same* ``ReplayEntry`` format as ``SelfPlayWorkerV2``
(entity features + auxiliary targets included), so it is a drop-in replacement
for the data-generation half of the training loop.

League play: to keep evaluations batchable, opponent selection happens once per
*wave* of ``n_envs`` games (rather than once per game). A wave is either pure
self-play (both sides use the current model) or all games share a single frozen
league opponent for player 1 — both are fully batched. PFSP variety comes from
re-selecting between waves.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch

from crsim.cards import CARD_DEFS
from crsim.game import Action, CRGame
from mcts.batched_gumbel import BatchedGumbelSearch
from mcts.gumbel_search import _action_id_to_action
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
from training.self_play_v2 import (
    SelfPlayV2Config,
    _phase_str,
    random_deck,
)

logger = logging.getLogger(__name__)


class _Env:
    """Per-game mutable state during a vectorized wave."""

    __slots__ = (
        "game", "trajectory", "card_trackers", "elixir_trackers",
        "live", "move_count",
    )

    def __init__(self, game: CRGame) -> None:
        self.game = game
        self.trajectory: list[dict] = []
        self.card_trackers = [CardTracker(), CardTracker()]
        self.elixir_trackers = [ElixirTracker(), ElixirTracker()]
        self.live = True
        self.move_count = 0


class VectorizedSelfPlayWorker:
    """Self-play worker that batches NN evaluations across parallel games."""

    def __init__(
        self,
        model: torch.nn.Module,
        replay_buffer: ReplayBuffer,
        config: SelfPlayV2Config | None = None,
        device: torch.device | None = None,
        worker_id: int = 0,
        n_envs: int = 32,
        max_eval_batch: int = 512,
        curriculum: CurriculumManager | None = None,
        randomizer: DomainRandomizer | None = None,
        league: League | None = None,
    ) -> None:
        self.model = model
        self.replay_buffer = replay_buffer
        self.config = config or SelfPlayV2Config()
        self.device = device or torch.device("cpu")
        self.worker_id = worker_id
        self.n_envs = max(1, n_envs)
        self.rng = np.random.default_rng(seed=worker_id)

        # Keep the rollout horizon aligned with the decision interval.
        self.config.gumbel_config.rollout_ticks = self.config.decision_interval_ticks

        self.searcher = BatchedGumbelSearch(
            model=self.model,
            config=self.config.gumbel_config,
            device=self.device,
            max_eval_batch=max_eval_batch,
        )
        self.max_eval_batch = max_eval_batch

        self.curriculum = curriculum
        self.randomizer = randomizer or DomainRandomizer(strength=0.0)
        self.league = league

        self._opponent_model: torch.nn.Module | None = None
        self._opponent_searcher: BatchedGumbelSearch | None = None

        self.games_played: int = 0
        self.total_positions: int = 0
        self.total_wins: dict[int, int] = {0: 0, 1: 0, -1: 0}

    # ------------------------------------------------------------------ #
    def _get_opponent_searcher(
        self, opponent_weights: dict | None,
    ) -> BatchedGumbelSearch:
        if opponent_weights is None:
            return self.searcher
        if self._opponent_model is None:
            self._opponent_model = copy.deepcopy(self.model)
            self._opponent_searcher = BatchedGumbelSearch(
                model=self._opponent_model,
                config=self.config.gumbel_config,
                device=self.device,
                max_eval_batch=self.max_eval_batch,
            )
        self._opponent_model.load_state_dict(opponent_weights)
        self._opponent_model.eval()
        return self._opponent_searcher  # type: ignore[return-value]

    def _new_game(self) -> CRGame:
        if self.curriculum is not None:
            deck_p0, deck_p1 = self.curriculum.deck_sampler.sample()
        elif self.config.random_decks:
            deck_p0 = random_deck(self.config.card_pool, self.rng)
            deck_p1 = random_deck(self.config.card_pool, self.rng)
        else:
            deck_p0 = None
            deck_p1 = None
        game = CRGame(
            deck_p0=deck_p0, deck_p1=deck_p1,
            seed=int(self.rng.integers(0, 2**31)),
        )
        if self.randomizer.strength > 0:
            self.randomizer.randomize_game(game)
        return game

    # ------------------------------------------------------------------ #
    def play_wave(self, n_games: int) -> list[ReplayEntry]:
        """Play ``n_games`` in parallel and return all replay entries."""
        cfg = self.config
        interval = max(1, cfg.decision_interval_ticks)

        # Opponent selection once per wave (keeps the batch single-model).
        opponent_weights = None
        if self.league is not None and hasattr(self.league, "get_opponent_weights"):
            opponent_weights = self.league.get_opponent_weights(self.worker_id)
        opp_searcher = self._get_opponent_searcher(opponent_weights)
        self_play = opp_searcher is self.searcher

        envs = [_Env(self._new_game()) for _ in range(n_games)]
        tick = 0

        while any(e.live for e in envs):
            live_envs = [e for e in envs if e.live]
            # Mark truncated / finished games.
            for e in live_envs:
                if e.game.done or (
                    cfg.max_ticks is not None and e.game.tick_count >= cfg.max_ticks
                ):
                    e.live = False
            live_envs = [e for e in envs if e.live]
            if not live_envs:
                break

            is_decision_tick = (tick % interval) == 0
            if not is_decision_tick:
                wait = [Action(0, -1), Action(1, -1)]
                for e in live_envs:
                    e.game.step(wait)
                tick += 1
                continue

            # Temperature schedule (move_count is identical across live envs).
            move_count = live_envs[0].move_count
            temp = 0.1 if move_count >= cfg.temperature_threshold else 1.0
            self.searcher.config.temperature = temp
            opp_searcher.config.temperature = temp

            # Gather batched search instances, grouped by which model evaluates.
            main_items: list[tuple[CRGame, int]] = []
            main_owner: list[tuple[_Env, int]] = []
            opp_items: list[tuple[CRGame, int]] = []
            opp_owner: list[tuple[_Env, int]] = []

            for e in live_envs:
                # Player 0 always uses the main model.
                main_items.append((e.game, 0))
                main_owner.append((e, 0))
                # Player 1 uses main model in self-play, else the league opponent.
                if self_play:
                    main_items.append((e.game, 1))
                    main_owner.append((e, 1))
                else:
                    opp_items.append((e.game, 1))
                    opp_owner.append((e, 1))

            probs_by_owner: dict[tuple[int, int], np.ndarray] = {}
            main_probs = self.searcher.search_many(main_items)
            for (e, p), pr in zip(main_owner, main_probs, strict=True):
                probs_by_owner[(id(e), p)] = pr
            if opp_items:
                opp_probs = opp_searcher.search_many(opp_items)
                for (e, p), pr in zip(opp_owner, opp_probs, strict=True):
                    probs_by_owner[(id(e), p)] = pr

            # Record trajectory + sample actions + step each game.
            for e in live_envs:
                actions_for_step = []
                for player in (0, 1):
                    spatial, scalar = encode_state(e.game, player)
                    entity_feats, entity_mask = extract_entity_features(e.game, player)
                    aux = extract_auxiliary_targets(e.game, player)
                    action_probs = probs_by_owner[(id(e), player)]

                    action_id = int(
                        np.random.choice(len(action_probs), p=action_probs)
                    )

                    e.trajectory.append({
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
                    self._track_play(e, player, action)

                e.game.step(actions_for_step)
                if self.randomizer.strength > 0:
                    self.randomizer.randomize_game(e.game)
                e.move_count += 1

            tick += 1

        # Build entries from each finished game.
        all_entries: list[ReplayEntry] = []
        for e in envs:
            all_entries.extend(self._finalize(e))
        return all_entries

    # ------------------------------------------------------------------ #
    def _track_play(self, e: _Env, player: int, action: Action) -> None:
        if action.is_wait:
            return
        ps = e.game.players[player]
        card_idx = ps.hand[action.hand_slot]
        played_card_type = ps.deck[card_idx] if card_idx is not None else None
        if played_card_type is None:
            return
        opponent = 1 - player
        e.card_trackers[opponent].observe_play(int(played_card_type), e.game.tick_count)
        card_def = CARD_DEFS.get(played_card_type)
        if card_def is not None:
            e.elixir_trackers[opponent].observe_play(
                card_def.cost, e.game.tick_count, _phase_str(e.game)
            )

    def _finalize(self, e: _Env) -> list[ReplayEntry]:
        game = e.game
        reward_0 = game.get_reward(0)
        if reward_0 > 0:
            self.total_wins[0] += 1
        elif reward_0 < 0:
            self.total_wins[1] += 1
        else:
            self.total_wins[-1] += 1

        final_aux_0 = extract_auxiliary_targets(game, 0)
        final_aux_1 = extract_auxiliary_targets(game, 1)

        entries: list[ReplayEntry] = []
        for step in e.trajectory:
            player = step["player"]
            reward = game.get_reward(player)
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

        if self.config.augment_flip and self.rng.random() < 0.5:
            for step in e.trajectory:
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

    # ------------------------------------------------------------------ #
    def run_batch(self, n_games: int | None = None) -> int:
        """Play ``n_games`` total (in waves of ``n_envs``) and push to buffer."""
        n = n_games or self.config.n_games
        total_positions = 0
        remaining = n
        while remaining > 0:
            wave = min(self.n_envs, remaining)
            entries = self.play_wave(wave)
            for entry in entries:
                self.replay_buffer.push(entry)
            total_positions += len(entries)
            remaining -= wave

        wr = self.total_wins
        logger.info(
            "VecWorker %d: %d games (n_envs=%d), %d positions | W/L/D: %d/%d/%d",
            self.worker_id, n, self.n_envs, total_positions, wr[0], wr[1], wr[-1],
        )
        return total_positions

    def update_model(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)
        self.model.eval()
