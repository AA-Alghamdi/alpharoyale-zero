"""Gumbel MuZero search — policy improvement with few simulations.

Key idea from "Policy improvement by planning with Gumbel" (ICLR 2022):
  - Sample actions using Gumbel-Top-k trick (no replacement)
  - Guaranteed policy improvement even with very few sims (16 instead of 800)
  - Much faster than vanilla MCTS → viable for real-time play
  - Works by using Sequential Halving to allocate simulations

This replaces the 800-simulation vanilla MCTS with 16-simulation Gumbel search,
giving ~50× speedup at inference with equivalent or better play quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from crsim.constants import ACTION_SPACE_SIZE, ARENA_H, ARENA_W, WAIT_ACTION
from crsim.game import Action, CRGame
from model.features import encode_state, extract_entity_features


@dataclass
class GumbelConfig:
    """Configuration for Gumbel MuZero search."""
    n_simulations: int = 16
    max_considered_actions: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    dirichlet_alpha: float = 0.15
    dirichlet_frac: float = 0.25
    temperature: float = 1.0
    # KataGo-style playout cap randomization
    playout_cap_randomization: bool = True
    playout_cap_min: int = 4
    playout_cap_max: int = 32


class GumbelNode:
    """Node in the Gumbel search tree."""

    __slots__ = (
        "prior", "action", "visit_count", "value_sum",
        "children", "is_expanded", "reward",
    )

    def __init__(self, action: int = -1, prior: float = 0.0) -> None:
        self.action = action
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: list[GumbelNode] = []
        self.is_expanded: bool = False
        self.reward: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, priors: np.ndarray, valid_mask: np.ndarray) -> None:
        self.is_expanded = True
        for a in range(len(priors)):
            if valid_mask[a]:
                self.children.append(GumbelNode(action=a, prior=float(priors[a])))


def _completed_q_value(node: GumbelNode, default_value: float = 0.0) -> float:
    """Q-value with a default for unvisited nodes."""
    if node.visit_count == 0:
        return default_value
    return node.q_value


def _improved_policy(
    root: GumbelNode,
    c_visit: float,
    c_scale: float,
    n_actions: int,
) -> np.ndarray:
    """Compute the improved policy from completed Q-values (Eq. 5 in the paper).

    π_improved(a) ∝ π(a) · exp(Q_completed(a) / (c_visit + max_visits) * c_scale)
    """
    if not root.children:
        return np.ones(n_actions) / n_actions

    max_visits = max(c.visit_count for c in root.children)
    sigma = c_scale * (c_visit + max_visits)

    logits = np.full(n_actions, -1e9)
    for child in root.children:
        q = _completed_q_value(child, default_value=0.0)
        logits[child.action] = math.log(max(child.prior, 1e-8)) + q / max(sigma, 1e-8)

    # Softmax
    logits -= logits.max()
    probs = np.exp(logits)
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


def _sequential_halving_schedule(n_sims: int, n_actions: int) -> list[int]:
    """Compute the number of visits per action at each halving phase.

    Sequential Halving allocates simulations in phases, halving the number
    of considered actions each phase.
    """
    if n_actions <= 1:
        return [n_sims]

    n_phases = max(1, int(math.ceil(math.log2(n_actions))))
    schedule = []
    remaining_actions = n_actions
    remaining_sims = n_sims

    for phase in range(n_phases):
        if remaining_actions <= 0 or remaining_sims <= 0:
            break
        sims_per_action = max(1, remaining_sims // (remaining_actions * (n_phases - phase)))
        schedule.append(sims_per_action)
        remaining_actions = max(1, remaining_actions // 2)
        remaining_sims -= sims_per_action * remaining_actions

    return schedule


class GumbelMuZeroSearch:
    """Gumbel MuZero search with sequential halving.

    Much faster than vanilla MCTS (16 sims vs 800) with guaranteed
    policy improvement.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: GumbelConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or GumbelConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _evaluate(
        self, game: CRGame, player: int,
    ) -> tuple[np.ndarray, float]:
        """Neural network evaluation with entity features (if model supports them)."""
        spatial, scalar = encode_state(game, player)
        valid_mask = game.get_valid_actions_mask(player)

        sp_t = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
        sc_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
        vm_t = torch.from_numpy(valid_mask).unsqueeze(0).to(self.device)

        # Check if model supports entity features (CRStarNet does, CRZeroNet doesn't)
        has_entity_support = hasattr(self.model, "entity_encoder")
        if has_entity_support:
            entity_feats, entity_mask = extract_entity_features(game, player)
            ef_t = torch.from_numpy(entity_feats).unsqueeze(0).to(self.device)
            em_t = torch.from_numpy(entity_mask).unsqueeze(0).to(self.device)
            policy, value = self.model.predict(
                sp_t, sc_t, vm_t, entity_features=ef_t, entity_mask=em_t,
            )
        else:
            policy, value = self.model.predict(sp_t, sc_t, vm_t)

        return policy.cpu().numpy()[0], float(value.cpu().numpy()[0])

    def search(
        self,
        game: CRGame,
        player: int,
    ) -> np.ndarray:
        """Run Gumbel search and return action probabilities.

        Uses the Gumbel-Top-k trick + Sequential Halving for efficient
        action selection with very few simulations.
        """
        cfg = self.config

        # Determine actual playout count (KataGo-style randomization)
        if cfg.playout_cap_randomization:
            n_sims = np.random.randint(cfg.playout_cap_min, cfg.playout_cap_max + 1)
        else:
            n_sims = cfg.n_simulations

        # Get root policy and value
        policy, root_value = self._evaluate(game, player)
        valid_mask = game.get_valid_actions_mask(player)

        # Add Dirichlet noise
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            noise = np.random.dirichlet([cfg.dirichlet_alpha] * n_valid)
            noise_full = np.zeros_like(policy)
            j = 0
            for i in range(len(policy)):
                if valid_mask[i]:
                    noise_full[i] = noise[j]
                    j += 1
            policy = (1 - cfg.dirichlet_frac) * policy + cfg.dirichlet_frac * noise_full
            policy *= valid_mask
            total = policy.sum()
            if total > 0:
                policy /= total

        # Gumbel-Top-k: sample top-k actions without replacement
        valid_actions = np.where(valid_mask > 0)[0]
        n_considered = min(cfg.max_considered_actions, len(valid_actions))

        if n_considered == 0:
            probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            probs[WAIT_ACTION] = 1.0
            return probs

        # Add Gumbel noise to log-priors and take top-k
        log_priors = np.log(np.maximum(policy[valid_actions], 1e-8))
        gumbel_noise = -np.log(-np.log(np.random.uniform(size=len(valid_actions)) + 1e-8) + 1e-8)
        gumbel_scores = log_priors + gumbel_noise
        top_k_indices = np.argsort(gumbel_scores)[-n_considered:]
        selected_actions = valid_actions[top_k_indices]

        # Create root node with only selected actions
        root = GumbelNode()
        root.is_expanded = True
        for action_id in selected_actions:
            child = GumbelNode(action=int(action_id), prior=float(policy[action_id]))
            root.children.append(child)

        # Sequential Halving
        schedule = _sequential_halving_schedule(n_sims, n_considered)
        active_children = list(root.children)

        for sims_per_action in schedule:
            # Simulate each active child
            for child in active_children:
                for _ in range(sims_per_action):
                    sim_game = game.clone()

                    # Apply this action
                    our_action = _action_id_to_action(child.action, player)
                    # Sample opponent action from policy
                    opp_policy, _ = self._evaluate(sim_game, 1 - player)
                    opp_mask = sim_game.get_valid_actions_mask(1 - player)
                    opp_policy *= opp_mask
                    opp_total = opp_policy.sum()
                    if opp_total > 0:
                        opp_policy /= opp_total
                        opp_action_id = int(np.random.choice(len(opp_policy), p=opp_policy))
                    else:
                        opp_action_id = WAIT_ACTION
                    opp_action = _action_id_to_action(opp_action_id, 1 - player)

                    actions = [our_action, opp_action] if player == 0 else [opp_action, our_action]
                    sim_game.step(actions)

                    if sim_game.done:
                        value = sim_game.get_reward(player)
                    else:
                        _, value = self._evaluate(sim_game, player)

                    child.visit_count += 1
                    child.value_sum += value

            # Halve: keep top half by Q-value
            if len(active_children) > 1:
                active_children.sort(key=lambda c: _completed_q_value(c), reverse=True)
                active_children = active_children[: max(1, len(active_children) // 2)]

        # Compute improved policy
        improved = _improved_policy(root, cfg.c_visit, cfg.c_scale, ACTION_SPACE_SIZE)

        # Apply temperature
        if cfg.temperature < 0.01:
            best = int(np.argmax(improved))
            action_probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            action_probs[best] = 1.0
        else:
            action_probs = improved ** (1.0 / cfg.temperature)
            total = action_probs.sum()
            if total > 0:
                action_probs /= total

        return action_probs

    def select_action(
        self,
        game: CRGame,
        player: int,
        deterministic: bool = False,
    ) -> tuple[int, np.ndarray]:
        """Run Gumbel search and select an action."""
        action_probs = self.search(game, player)

        if deterministic:
            action_id = int(np.argmax(action_probs))
        else:
            action_id = int(np.random.choice(len(action_probs), p=action_probs))

        return action_id, action_probs


def _action_id_to_action(action_id: int, player: int) -> Action:
    """Convert flat action id to Action."""
    if action_id == WAIT_ACTION:
        return Action(player=player, hand_slot=-1)
    slot = action_id // (ARENA_W * ARENA_H)
    remainder = action_id % (ARENA_W * ARENA_H)
    x = remainder // ARENA_H
    y = remainder % ARENA_H
    return Action(player=player, hand_slot=slot, x=float(x), y=float(y))
