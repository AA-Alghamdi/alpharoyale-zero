"""MCTS implementation — AlphaZero-style with simultaneous-move handling.

Key differences from vanilla AlphaZero MCTS:
  1. Simultaneous moves → we sample the opponent's action from the NN prior
     and search over our own actions only (Smooth UCT).
  2. Virtual loss for parallel search (optional multi-threaded mode).
  3. Dirichlet noise at the root for exploration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from crsim.actions import action_id_to_action as _action_id_to_action
from crsim.constants import ACTION_SPACE_SIZE
from crsim.game import Action, CRGame
from model.features import encode_state
from model.network import CRZeroNet


@dataclass
class MCTSConfig:
    n_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.15
    dirichlet_frac: float = 0.25
    temperature: float = 1.0  # 1.0 for exploration, near 0 for exploitation
    virtual_loss: float = 3.0


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = (
        "parent",
        "action",
        "prior",
        "visit_count",
        "value_sum",
        "virtual_loss_count",
        "children",
        "is_expanded",
    )

    def __init__(
        self,
        parent: MCTSNode | None = None,
        action: int = -1,
        prior: float = 0.0,
    ) -> None:
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.virtual_loss_count: int = 0
        self.children: list[MCTSNode] = []
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        n = self.visit_count + self.virtual_loss_count
        if n == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss_count * 1.0) / n

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """Upper confidence bound score for selection."""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def select_child(self, c_puct: float) -> MCTSNode:
        """Select the child with highest UCB score."""
        best_score = -float("inf")
        best_child = self.children[0]
        for child in self.children:
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, action_priors: np.ndarray, valid_mask: np.ndarray) -> None:
        """Expand node with children for all valid actions."""
        self.is_expanded = True
        for action_id in range(len(action_priors)):
            if valid_mask[action_id]:
                child = MCTSNode(
                    parent=self,
                    action=action_id,
                    prior=float(action_priors[action_id]),
                )
                self.children.append(child)

    def backup(self, value: float) -> None:
        """Back-propagate the evaluation value up the tree."""
        node: MCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # flip for opponent
            node = node.parent


class MCTSPlayer:
    """MCTS-based player using a neural network for evaluation."""

    def __init__(
        self,
        model: CRZeroNet,
        config: MCTSConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or MCTSConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _evaluate(
        self, game: CRGame, player: int
    ) -> tuple[np.ndarray, float]:
        """Run the neural network on a game state.

        Returns (policy_probs, value) where policy_probs sums to 1 over valid actions.
        """
        spatial, scalar = encode_state(game, player)
        valid_mask = game.get_valid_actions_mask(player)

        sp_t = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
        sc_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
        vm_t = torch.from_numpy(valid_mask).unsqueeze(0).to(self.device)

        policy, value = self.model.predict(sp_t, sc_t, vm_t)

        policy_np = policy.cpu().numpy()[0]  # (A,)
        value_f = float(value.cpu().numpy()[0])

        return policy_np, value_f

    def _sample_opponent_action(
        self, game: CRGame, opponent: int
    ) -> Action:
        """Sample opponent action from the NN prior (for simultaneous-move handling)."""
        policy, _ = self._evaluate(game, opponent)
        valid_mask = game.get_valid_actions_mask(opponent)
        # Mask and re-normalize
        masked_policy = policy * valid_mask
        total = masked_policy.sum()
        if total > 0:
            masked_policy /= total
        else:
            # Fallback to wait
            return Action(player=opponent, hand_slot=-1)

        action_id = int(np.random.choice(len(masked_policy), p=masked_policy))
        return _action_id_to_action(action_id, opponent)

    def search(
        self,
        game: CRGame,
        player: int,
    ) -> np.ndarray:
        """Run MCTS and return the action visit-count distribution.

        Returns
        -------
        action_probs : ndarray of shape (ACTION_SPACE_SIZE,)
            Probability distribution over actions based on visit counts.
        """
        cfg = self.config
        root = MCTSNode()

        # Evaluate root
        policy, _ = self._evaluate(game, player)
        valid_mask = game.get_valid_actions_mask(player)

        # Add Dirichlet noise at root
        noise = np.random.dirichlet(
            [cfg.dirichlet_alpha] * int(valid_mask.sum())
        )
        noise_full = np.zeros_like(policy)
        j = 0
        for i in range(len(policy)):
            if valid_mask[i]:
                noise_full[i] = noise[j]
                j += 1

        noisy_policy = (
            (1 - cfg.dirichlet_frac) * policy
            + cfg.dirichlet_frac * noise_full
        )
        # Re-normalize
        noisy_policy *= valid_mask
        total = noisy_policy.sum()
        if total > 0:
            noisy_policy /= total

        root.expand(noisy_policy, valid_mask)

        # Run simulations
        for _ in range(cfg.n_simulations):
            node = root
            sim_game = game.clone()
            search_path: list[MCTSNode] = [node]

            # SELECT
            while node.is_expanded and node.children:
                node = node.select_child(cfg.c_puct)
                search_path.append(node)

                # Apply our action + sampled opponent action
                our_action = _action_id_to_action(node.action, player)
                opp_action = self._sample_opponent_action(sim_game, 1 - player)
                sim_game.step([our_action, opp_action] if player == 0 else [opp_action, our_action])

                if sim_game.done:
                    break

            # EVALUATE
            if sim_game.done:
                value = sim_game.get_reward(player)
            else:
                # Expand the leaf
                leaf_policy, value = self._evaluate(sim_game, player)
                leaf_mask = sim_game.get_valid_actions_mask(player)
                if not node.is_expanded:
                    # Normalize
                    masked_p = leaf_policy * leaf_mask
                    total = masked_p.sum()
                    if total > 0:
                        masked_p /= total
                    node.expand(masked_p, leaf_mask)

            # BACKUP
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        # Extract action probabilities from visit counts
        action_probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for child in root.children:
            action_probs[child.action] = child.visit_count

        # Temperature
        if cfg.temperature < 0.01:
            # Greedy
            best = int(np.argmax(action_probs))
            action_probs[:] = 0.0
            action_probs[best] = 1.0
        else:
            # Boltzmann
            action_probs = action_probs ** (1.0 / cfg.temperature)
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
        """Run MCTS and select an action.

        Returns (action_id, action_probs).
        """
        action_probs = self.search(game, player)

        if deterministic:
            action_id = int(np.argmax(action_probs))
        else:
            action_id = int(np.random.choice(len(action_probs), p=action_probs))

        return action_id, action_probs
