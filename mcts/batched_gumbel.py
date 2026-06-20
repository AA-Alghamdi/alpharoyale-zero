"""Batched Gumbel MuZero search across many game states at once.

This is the search counterpart to :mod:`model.batched_eval`. The single-game
:class:`~mcts.gumbel_search.GumbelMuZeroSearch` runs one neural-network forward
pass per evaluated state, so stepping ``N`` self-play games means ``N`` separate
(tiny) forward passes — terrible GPU utilization.

:class:`BatchedGumbelSearch` runs the *same* Gumbel search algorithm (Gumbel
Top-k + Sequential Halving + improved policy) but for a whole list of
``(game, player)`` instances simultaneously, batching every neural-network
evaluation:

  * the **root** evaluation of all instances (and their opponents) into one
    forward pass, and
  * every **leaf** evaluation within a Sequential-Halving phase, across all
    instances, into chunked forward passes.

The game logic (cloning, stepping, rolling forward the decision window) is
identical to the single-game search, so results match it up to the unavoidable
stochasticity of Dirichlet/Gumbel noise and opponent sampling. The win is that
the expensive part — the NN — now sees large batches.

The opponent action inside a rollout is sampled from the opponent's *root*
policy, exactly as in the single-game search (where ``sim_game`` is a fresh
clone of the root before our action is applied, so the opponent evaluation is
always at the root state). We therefore evaluate each opponent's root policy
once, in the same batched root pass, instead of re-evaluating per simulation.
"""

from __future__ import annotations

import numpy as np
import torch

from crsim.constants import ACTION_SPACE_SIZE, WAIT_ACTION
from crsim.game import Action, CRGame
from mcts.gumbel_search import (
    GumbelConfig,
    GumbelNode,
    _action_id_to_action,
    _completed_q_value,
    _improved_policy,
    _sequential_halving_schedule,
)
from model.batched_eval import (
    encode_request,
    forward_encoded,
    model_has_entity_support,
)


class BatchedGumbelSearch:
    """Gumbel MuZero search that batches NN evals across many instances."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: GumbelConfig | None = None,
        device: torch.device | None = None,
        max_eval_batch: int = 512,
    ) -> None:
        self.model = model
        self.config = config or GumbelConfig()
        self.device = device or torch.device("cpu")
        self.max_eval_batch = max_eval_batch
        self.model.to(self.device)
        self.model.eval()
        self._has_entity = model_has_entity_support(model)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def search_many(
        self, instances: list[tuple[CRGame, int]],
    ) -> list[np.ndarray]:
        """Run Gumbel search for every ``(game, player)`` and return policies.

        Returns a list aligned with ``instances``; each element is an
        ``(ACTION_SPACE_SIZE,)`` float32 action-probability vector (post
        temperature), exactly like ``GumbelMuZeroSearch.search``.
        """
        if not instances:
            return []

        cfg = self.config

        # One playout count for the whole batch tick (KataGo-style).
        if cfg.playout_cap_randomization:
            n_sims = int(np.random.randint(cfg.playout_cap_min, cfg.playout_cap_max + 1))
        else:
            n_sims = cfg.n_simulations

        # --- Batched root evaluation (self + opponent) ----------------- #
        # Request list: for each instance, our state then the opponent state.
        root_reqs: list[dict[str, np.ndarray]] = []
        for game, player in instances:
            root_reqs.append(encode_request(game, player, self._has_entity))
            root_reqs.append(encode_request(game, 1 - player, self._has_entity))
        root_pol, _root_val = forward_encoded(
            self.model, root_reqs, self.device, self._has_entity,
            max_batch=self.max_eval_batch,
        )

        # --- Per-instance root setup (noise, Gumbel top-k, children) --- #
        states: list[_InstanceState] = []
        for i, (game, player) in enumerate(instances):
            policy = root_pol[2 * i].copy()
            opp_policy = root_pol[2 * i + 1].copy()
            valid_mask = game.get_valid_actions_mask(player)
            opp_mask = game.get_valid_actions_mask(1 - player)
            states.append(
                self._make_instance_state(
                    game, player, policy, opp_policy, valid_mask, opp_mask, n_sims,
                )
            )

        # --- Sequential Halving, batched across instances per phase ---- #
        max_phases = max((len(s.schedule) for s in states), default=0)
        for phase in range(max_phases):
            self._run_phase(states, phase)

        # --- Improved policy + temperature per instance ---------------- #
        results: list[np.ndarray] = []
        for s in states:
            results.append(self._final_policy(s))
        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _make_instance_state(
        self,
        game: CRGame,
        player: int,
        policy: np.ndarray,
        opp_policy: np.ndarray,
        valid_mask: np.ndarray,
        opp_mask: np.ndarray,
        n_sims: int,
    ) -> _InstanceState:
        cfg = self.config

        # Dirichlet noise over our valid actions (mirrors single-game search).
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            noise = np.random.dirichlet([cfg.dirichlet_alpha] * n_valid)
            noise_full = np.zeros_like(policy)
            noise_full[valid_mask > 0] = noise
            policy = (1 - cfg.dirichlet_frac) * policy + cfg.dirichlet_frac * noise_full
            policy *= valid_mask
            total = policy.sum()
            if total > 0:
                policy /= total

        # Normalize opponent root policy over its valid actions (for sampling).
        opp_policy = opp_policy * opp_mask
        opp_total = opp_policy.sum()
        if opp_total > 0:
            opp_policy = opp_policy / opp_total

        # Gumbel Top-k selection of considered actions.
        valid_actions = np.where(valid_mask > 0)[0]
        n_considered = min(cfg.max_considered_actions, len(valid_actions))

        root = GumbelNode()
        root.is_expanded = True
        if n_considered > 0:
            log_priors = np.log(np.maximum(policy[valid_actions], 1e-8))
            gumbel_noise = -np.log(
                -np.log(np.random.uniform(size=len(valid_actions)) + 1e-8) + 1e-8
            )
            gumbel_scores = log_priors + gumbel_noise
            top_k_indices = np.argsort(gumbel_scores)[-n_considered:]
            selected_actions = valid_actions[top_k_indices]
            for action_id in selected_actions:
                root.children.append(
                    GumbelNode(action=int(action_id), prior=float(policy[action_id]))
                )

        schedule = (
            _sequential_halving_schedule(n_sims, n_considered)
            if n_considered > 0
            else []
        )

        return _InstanceState(
            game=game,
            player=player,
            opp_policy=opp_policy,
            root=root,
            active_children=list(root.children),
            schedule=schedule,
        )

    def _run_phase(self, states: list[_InstanceState], phase: int) -> None:
        """Run one Sequential-Halving phase across all instances, batched."""
        pending_encoded: list[dict[str, np.ndarray]] = []
        pending_children: list[GumbelNode] = []

        def flush() -> None:
            if not pending_encoded:
                return
            _pol, vals = forward_encoded(
                self.model, pending_encoded, self.device, self._has_entity,
                max_batch=self.max_eval_batch,
            )
            for child, v in zip(pending_children, vals, strict=True):
                child.visit_count += 1
                child.value_sum += float(v)
            pending_encoded.clear()
            pending_children.clear()

        for s in states:
            if phase >= len(s.schedule):
                continue
            sims_per_action = s.schedule[phase]
            for child in s.active_children:
                for _ in range(sims_per_action):
                    leaf, terminal_value = self._rollout_leaf(s, child.action)
                    if terminal_value is not None:
                        child.visit_count += 1
                        child.value_sum += terminal_value
                    else:
                        pending_encoded.append(leaf)
                        pending_children.append(child)
                        if len(pending_encoded) >= self.max_eval_batch:
                            flush()

        flush()

        # Halve: keep the top half by completed Q-value, per instance.
        for s in states:
            if phase >= len(s.schedule):
                continue
            if len(s.active_children) > 1:
                s.active_children.sort(key=_completed_q_value, reverse=True)
                s.active_children = s.active_children[: max(1, len(s.active_children) // 2)]

    def _rollout_leaf(
        self, s: _InstanceState, action_id: int,
    ) -> tuple[dict[str, np.ndarray] | None, float | None]:
        """Simulate one action from the root and return an encoded leaf.

        Mirrors the single-game search: apply our action + a sampled opponent
        action, roll the rest of the decision window forward with WAIT, then
        either return the terminal reward or the encoded state to evaluate.
        """
        cfg = self.config
        sim_game = s.game.clone()

        our_action = _action_id_to_action(action_id, s.player)
        if s.opp_policy.sum() > 0:
            opp_action_id = int(np.random.choice(len(s.opp_policy), p=s.opp_policy))
        else:
            opp_action_id = WAIT_ACTION
        opp_action = _action_id_to_action(opp_action_id, 1 - s.player)

        actions = (
            [our_action, opp_action] if s.player == 0 else [opp_action, our_action]
        )
        sim_game.step(actions)

        wait_actions = [Action(0, -1), Action(1, -1)]
        for _ in range(max(0, cfg.rollout_ticks - 1)):
            if sim_game.done:
                break
            sim_game.step(wait_actions)

        if sim_game.done:
            return None, float(sim_game.get_reward(s.player))
        return encode_request(sim_game, s.player, self._has_entity), None

    def _final_policy(self, s: _InstanceState) -> np.ndarray:
        cfg = self.config
        if not s.root.children:
            probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            probs[WAIT_ACTION] = 1.0
            return probs

        improved = _improved_policy(
            s.root, cfg.c_visit, cfg.c_scale, ACTION_SPACE_SIZE,
        )
        if cfg.temperature < 0.01:
            best = int(np.argmax(improved))
            action_probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            action_probs[best] = 1.0
        else:
            action_probs = improved ** (1.0 / cfg.temperature)
            total = action_probs.sum()
            if total > 0:
                action_probs /= total
        return action_probs.astype(np.float32, copy=False)


class _InstanceState:
    """Mutable per-instance search state during batched search."""

    __slots__ = (
        "game", "player", "opp_policy", "root", "active_children", "schedule",
    )

    def __init__(
        self,
        game: CRGame,
        player: int,
        opp_policy: np.ndarray,
        root: GumbelNode,
        active_children: list[GumbelNode],
        schedule: list[int],
    ) -> None:
        self.game = game
        self.player = player
        self.opp_policy = opp_policy
        self.root = root
        self.active_children = active_children
        self.schedule = schedule
