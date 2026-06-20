"""Two-stage Hierarchical Action Space.

AlphaStar-style autoregressive action decomposition:
  Stage 1: WHAT to play (5 choices: 4 hand slots + WAIT)
  Stage 2: WHERE to place (continuous or fine grid)

This reduces the branching factor from 2,305 (flat) to 5 × position,
making MCTS search much more efficient.

References:
  - AlphaStar: autoregressive policy (action_type → unit → target → queuing)
  - Na??veMCTS: factored action space for RTS games
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class TwoStageActionHead(nn.Module):
    """Two-stage action head: WHAT card + WHERE to place.

    Stage 1: CardHead — P(card | state) for 5 actions
    Stage 2: PlacementHead — P(x, y | state, card) as grid or Gaussian

    The log-probability of a full action is:
        log P(action) = log P(card) + log P(x, y | card)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_hand_slots: int = 4,
        grid_w: int = 18,
        grid_h: int = 32,
        fine_grid_factor: int = 2,  # 2x = 36x64 grid
    ) -> None:
        super().__init__()
        self.n_hand_slots = n_hand_slots
        self.n_card_actions = n_hand_slots + 1  # +1 for WAIT
        self.grid_w = grid_w * fine_grid_factor
        self.grid_h = grid_h * fine_grid_factor
        self.fine_grid_factor = fine_grid_factor

        # Stage 1: What card to play
        self.card_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_card_actions),
        )

        # Stage 2: Where to place (per-card placement heads share weights)
        # Output: spatial logits over grid
        self.placement_head = nn.Sequential(
            nn.Linear(hidden_dim + self.n_card_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.grid_w * self.grid_h),
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_cards: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, hidden_dim) — encoded game state
            valid_cards: (batch, n_card_actions) — mask of playable cards

        Returns:
            card_logits: (batch, n_card_actions)
            placement_logits: (batch, n_card_actions, grid_h * grid_w)
        """
        batch = features.shape[0]

        # Stage 1: card selection
        card_logits = self.card_head(features)

        if valid_cards is not None:
            card_logits = card_logits.masked_fill(~valid_cards.bool(), float("-inf"))

        # Stage 2: placement for each possible card
        card_onehots = torch.eye(self.n_card_actions, device=features.device)
        card_onehots = card_onehots.unsqueeze(0).expand(batch, -1, -1)

        features_exp = features.unsqueeze(1).expand(-1, self.n_card_actions, -1)
        combined = torch.cat([features_exp, card_onehots], dim=-1)

        placement_logits = self.placement_head(
            combined.reshape(-1, combined.shape[-1])
        ).reshape(batch, self.n_card_actions, -1)

        return card_logits, placement_logits

    def sample_action(
        self,
        features: torch.Tensor,
        valid_cards: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a full action autoregressively.

        Returns:
            card_idx: (batch,) — selected card (0-4)
            tile_idx: (batch,) — selected tile (0 to grid_w*grid_h-1)
            log_prob: (batch,) — total log probability
        """
        card_logits, placement_logits = self.forward(features, valid_cards)

        # Sample card
        card_probs = F.softmax(card_logits / temperature, dim=-1)
        card_dist = torch.distributions.Categorical(card_probs)
        card_idx = card_dist.sample()
        card_log_prob = card_dist.log_prob(card_idx)

        # Sample placement conditioned on selected card
        # Gather placement logits for the selected card
        batch_idx = torch.arange(features.shape[0], device=features.device)
        selected_placement = placement_logits[batch_idx, card_idx]  # (batch, grid_h*grid_w)

        tile_probs = F.softmax(selected_placement / temperature, dim=-1)
        tile_dist = torch.distributions.Categorical(tile_probs)
        tile_idx = tile_dist.sample()
        tile_log_prob = tile_dist.log_prob(tile_idx)

        total_log_prob = card_log_prob + tile_log_prob

        return card_idx, tile_idx, total_log_prob

    def to_flat_action(self, card_idx: int, tile_idx: int) -> int:
        """Convert two-stage action to flat action index.

        Flat action format: card * (grid_w * grid_h) + tile
        WAIT action = n_hand_slots * grid_w * grid_h
        """
        if card_idx >= self.n_hand_slots:  # WAIT
            return self.n_hand_slots * self.grid_w * self.grid_h
        return card_idx * (self.grid_w * self.grid_h) + tile_idx

    def from_flat_action(self, flat_action: int) -> tuple[int, int, float, float]:
        """Convert flat action to (hand_slot, tile_idx, tile_x, tile_y).

        tile_x, tile_y are in original (pre-fine-grid) coordinate space.
        """
        total_card_actions = self.n_hand_slots * self.grid_w * self.grid_h
        if flat_action >= total_card_actions:
            return -1, -1, 0.0, 0.0  # WAIT

        hand_slot = flat_action // (self.grid_w * self.grid_h)
        tile_idx = flat_action % (self.grid_w * self.grid_h)
        tile_x = (tile_idx % self.grid_w) / self.fine_grid_factor
        tile_y = (tile_idx // self.grid_w) / self.fine_grid_factor

        return hand_slot, tile_idx, tile_x, tile_y


class GaussianPlacementHead(nn.Module):
    """Continuous placement via Gaussian mixture model.

    Instead of discretizing the arena into a grid, outputs a mixture
    of 2D Gaussians for sub-tile precision.

    Each Gaussian: (mu_x, mu_y, sigma_x, sigma_y, weight)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_mixtures: int = 5,
        arena_w: float = 18.0,
        arena_h: float = 32.0,
    ) -> None:
        super().__init__()
        self.n_mixtures = n_mixtures
        self.arena_w = arena_w
        self.arena_h = arena_h

        # Output: 5 params per mixture (mu_x, mu_y, log_sigma_x, log_sigma_y, logit_weight)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_mixtures * 5),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns dict with:
            mu: (batch, n_mixtures, 2) — means
            sigma: (batch, n_mixtures, 2) — std devs
            weights: (batch, n_mixtures) — mixture weights
        """
        raw = self.head(features)
        raw = raw.reshape(-1, self.n_mixtures, 5)

        mu_x = torch.sigmoid(raw[..., 0]) * self.arena_w
        mu_y = torch.sigmoid(raw[..., 1]) * self.arena_h
        sigma_x = torch.exp(raw[..., 2].clamp(-3, 1))  # 0.05 to 2.7 tiles
        sigma_y = torch.exp(raw[..., 3].clamp(-3, 1))
        weights = F.softmax(raw[..., 4], dim=-1)

        return {
            "mu": torch.stack([mu_x, mu_y], dim=-1),
            "sigma": torch.stack([sigma_x, sigma_y], dim=-1),
            "weights": weights,
        }

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a placement position.

        Returns:
            position: (batch, 2) — (x, y) in tile coordinates
            log_prob: (batch,) — log probability
        """
        params = self.forward(features)
        batch = features.shape[0]

        # Sample mixture component
        comp_dist = torch.distributions.Categorical(params["weights"])
        comp_idx = comp_dist.sample()  # (batch,)

        # Get selected component's parameters
        batch_idx = torch.arange(batch, device=features.device)
        mu = params["mu"][batch_idx, comp_idx]  # (batch, 2)
        sigma = params["sigma"][batch_idx, comp_idx]  # (batch, 2)

        # Sample from selected Gaussian
        normal = torch.distributions.Normal(mu, sigma)
        position = normal.sample()

        # Clamp to arena bounds
        position[:, 0] = position[:, 0].clamp(0, self.arena_w)
        position[:, 1] = position[:, 1].clamp(0, self.arena_h)

        # Log probability
        log_prob = comp_dist.log_prob(comp_idx) + normal.log_prob(position).sum(dim=-1)

        return position, log_prob
