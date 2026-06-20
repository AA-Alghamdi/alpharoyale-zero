"""True MuZero search — plans in latent space using the learned dynamics model.

Unlike gumbel_search.py which clones the game for each simulation, this
searches entirely in latent space:
  1. Encode current observation → latent state via model.encode()
  2. For each simulation: apply dynamics model (latent + action → next_latent + reward)
  3. Use Gumbel-Top-k + Sequential Halving for efficient action selection

This is the real MuZero algorithm — no simulator needed during inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from crsim.constants import (
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    NUM_HAND_SLOTS,
    WAIT_ACTION,
)
from crsim.game import Action, CRGame
from model.features import encode_state, extract_entity_features


@dataclass
class MuZeroConfig:
    """Configuration for true MuZero search."""
    n_simulations: int = 16
    max_considered_actions: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    discount: float = 0.997
    dirichlet_alpha: float = 0.15
    dirichlet_frac: float = 0.25
    temperature: float = 1.0
    # KataGo playout cap randomization
    playout_cap_randomization: bool = True
    playout_cap_min: int = 4
    playout_cap_max: int = 32
    # Search depth in latent space
    max_depth: int = 5


class MuZeroNode:
    """Node in the MuZero latent search tree."""

    __slots__ = (
        "action", "prior", "visit_count", "value_sum",
        "reward", "children", "is_expanded",
        "hidden_state",
    )

    def __init__(self, action: int = -1, prior: float = 0.0) -> None:
        self.action = action
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.reward: float = 0.0
        self.children: list[MuZeroNode] = []
        self.is_expanded: bool = False
        self.hidden_state: torch.Tensor | None = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _action_to_onehot(action_id: int, device: torch.device) -> torch.Tensor:
    """Convert flat action ID to one-hot vector for dynamics model.

    Layout: [card_oh(5) + x_oh(18) + y_oh(32)] = 55-d
    """
    card_oh = torch.zeros(NUM_HAND_SLOTS + 1, device=device)
    x_oh = torch.zeros(ARENA_W, device=device)
    y_oh = torch.zeros(ARENA_H, device=device)

    if action_id == WAIT_ACTION:
        card_oh[NUM_HAND_SLOTS] = 1.0  # wait token
    else:
        slot = action_id // (ARENA_W * ARENA_H)
        remainder = action_id % (ARENA_W * ARENA_H)
        x = remainder // ARENA_H
        y = remainder % ARENA_H
        card_oh[min(slot, NUM_HAND_SLOTS)] = 1.0
        x_oh[min(x, ARENA_W - 1)] = 1.0
        y_oh[min(y, ARENA_H - 1)] = 1.0

    return torch.cat([card_oh, x_oh, y_oh]).unsqueeze(0)  # (1, 55)


def _improved_policy(
    root: MuZeroNode,
    c_visit: float,
    c_scale: float,
    n_actions: int,
) -> np.ndarray:
    """Compute improved policy from completed Q-values."""
    if not root.children:
        return np.ones(n_actions, dtype=np.float32) / n_actions

    max_visits = max(c.visit_count for c in root.children)
    sigma = c_scale * (c_visit + max_visits)

    logits = np.full(n_actions, -1e9, dtype=np.float64)
    for child in root.children:
        q = child.q_value if child.visit_count > 0 else 0.0
        logits[child.action] = (
            math.log(max(child.prior, 1e-8)) + q / max(sigma, 1e-8)
        )

    logits -= logits.max()
    probs = np.exp(logits)
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs.astype(np.float32)


def _sequential_halving_schedule(n_sims: int, n_actions: int) -> list[int]:
    """Compute sims-per-action at each halving phase."""
    if n_actions <= 1:
        return [n_sims]

    n_phases = max(1, int(math.ceil(math.log2(n_actions))))
    schedule = []
    remaining_actions = n_actions
    remaining_sims = n_sims

    for phase in range(n_phases):
        if remaining_actions <= 0 or remaining_sims <= 0:
            break
        sims_per = max(
            1, remaining_sims // (remaining_actions * (n_phases - phase))
        )
        schedule.append(sims_per)
        remaining_actions = max(1, remaining_actions // 2)
        remaining_sims -= sims_per * remaining_actions

    return schedule


def _ucb_score(
    child: MuZeroNode,
    parent_visits: int,
    c_visit: float,
    c_scale: float,
    min_q: float,
    max_q: float,
) -> float:
    """PUCT score with Q-value normalization (MuZero-style)."""
    prior_score = (
        c_scale * child.prior
        * math.sqrt(parent_visits)
        / (1 + child.visit_count)
    )
    if child.visit_count > 0:
        q = child.q_value
        # Normalize Q to [0, 1] range
        if max_q > min_q:
            q = (q - min_q) / (max_q - min_q)
        else:
            q = 0.5
    else:
        q = 0.0
    return q + prior_score


class MuZeroSearch:
    """True MuZero search: plans in latent space using learned dynamics.

    Key difference from GumbelMuZeroSearch: does NOT clone the game.
    All simulation happens in the neural network's latent space via:
      dynamics_model(hidden_state, action) → (next_hidden, reward)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: MuZeroConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or MuZeroConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _initial_inference(
        self,
        game: CRGame,
        player: int,
    ) -> tuple[np.ndarray, float, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode observation → (policy, value, hidden_state, lstm_hidden)."""
        import torch.nn.functional as f_nn  # noqa: N812

        spatial, scalar = encode_state(game, player)
        entity_feats, entity_mask = extract_entity_features(game, player)
        valid_mask = game.get_valid_actions_mask(player)

        sp_t = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
        sc_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
        ef_t = torch.from_numpy(entity_feats).unsqueeze(0).to(self.device)
        em_t = torch.from_numpy(entity_mask).unsqueeze(0).to(self.device)
        vm_t = torch.from_numpy(valid_mask).unsqueeze(0).to(self.device)

        # Full forward through encoder
        core_out, lstm_hidden = self.model.encode(
            sp_t, sc_t, ef_t, em_t,
        )

        # Get policy from flat head
        logits = self.model.flat_policy_head(core_out)
        logits = logits.masked_fill(~vm_t, -1e9)
        policy = f_nn.softmax(logits, dim=-1).cpu().numpy()[0]

        # Value
        value = float(self.model.value_head(core_out).cpu().numpy()[0])

        return policy, value, core_out, lstm_hidden

    @torch.no_grad()
    def _recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action_id: int,
    ) -> tuple[float, float, torch.Tensor]:
        """Dynamics model: hidden + action → (value, reward, next_hidden)."""

        action_oh = _action_to_onehot(action_id, self.device)
        next_hidden, reward = self.model.dynamics(hidden_state, action_oh)

        # Predict value from next hidden state
        value = float(self.model.value_head(next_hidden).cpu().numpy()[0])
        reward_f = float(reward.cpu().numpy()[0])

        return value, reward_f, next_hidden

    def search(
        self,
        game: CRGame,
        player: int,
    ) -> np.ndarray:
        """Run MuZero search in latent space.

        Uses Gumbel-Top-k + Sequential Halving for efficient selection.
        """
        cfg = self.config

        # Playout cap randomization
        if cfg.playout_cap_randomization:
            n_sims = int(np.random.randint(cfg.playout_cap_min, cfg.playout_cap_max + 1))
        else:
            n_sims = cfg.n_simulations

        # Initial inference
        policy, root_value, root_hidden, _ = self._initial_inference(game, player)
        valid_mask = game.get_valid_actions_mask(player)

        # Dirichlet noise
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

        # Gumbel-Top-k action selection
        valid_actions = np.where(valid_mask > 0)[0]
        n_considered = min(cfg.max_considered_actions, len(valid_actions))

        if n_considered == 0:
            probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            probs[WAIT_ACTION] = 1.0
            return probs

        log_priors = np.log(np.maximum(policy[valid_actions], 1e-8))
        gumbel_noise = -np.log(-np.log(
            np.random.uniform(size=len(valid_actions)) + 1e-8
        ) + 1e-8)
        gumbel_scores = log_priors + gumbel_noise
        top_k_indices = np.argsort(gumbel_scores)[-n_considered:]
        selected_actions = valid_actions[top_k_indices]

        # Create root
        root = MuZeroNode()
        root.is_expanded = True
        root.hidden_state = root_hidden
        for action_id in selected_actions:
            child = MuZeroNode(
                action=int(action_id), prior=float(policy[action_id]),
            )
            root.children.append(child)

        # Sequential Halving
        schedule = _sequential_halving_schedule(n_sims, n_considered)
        active_children = list(root.children)

        for sims_per_action in schedule:
            for child in active_children:
                for _ in range(sims_per_action):
                    # Simulate from root's latent state
                    value, reward, next_hidden = self._recurrent_inference(
                        root.hidden_state, child.action,
                    )

                    # Can optionally go deeper
                    total_reward = reward
                    current_hidden = next_hidden
                    for depth in range(1, cfg.max_depth):
                        # Get policy at this latent state
                        deeper_logits = self.model.flat_policy_head(current_hidden)
                        deeper_probs = torch.softmax(deeper_logits, dim=-1)
                        # Sample action from policy
                        next_action = int(
                            torch.multinomial(deeper_probs, 1).item()
                        )
                        v, r, current_hidden = self._recurrent_inference(
                            current_hidden, next_action,
                        )
                        total_reward += (cfg.discount ** depth) * r
                        value = v

                    # Backup: discounted reward + leaf value
                    backed_up = total_reward + (
                        cfg.discount ** cfg.max_depth
                    ) * value

                    child.visit_count += 1
                    child.value_sum += backed_up

            # Halve
            if len(active_children) > 1:
                active_children.sort(
                    key=lambda c: c.q_value if c.visit_count > 0 else -1e9,
                    reverse=True,
                )
                active_children = active_children[
                    : max(1, len(active_children) // 2)
                ]

        # Compute improved policy
        improved = _improved_policy(
            root, cfg.c_visit, cfg.c_scale, ACTION_SPACE_SIZE,
        )

        # Temperature
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
        """Run MuZero search and select action."""
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
