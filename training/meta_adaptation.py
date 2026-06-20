"""Meta Adaptation: Surgery + Fine-tuning for balance patches.

When Clash Royale releases a balance patch (every 2-4 weeks), card stats
change. This module handles adapting the trained model without retraining
from scratch.

Approach from OpenAI Five (Dota 2):
  "Surgery" — modify network weights to preserve behavior when game
  mechanics change. OpenAI did 20+ surgeries over 10 months without
  losing performance.

Techniques:
  1. Weight scaling: scale card embeddings proportionally to stat changes
  2. New card initialization: interpolate from similar cards
  3. Action space extension: extend output layer for new actions
  4. Fine-tuning: short self-play run (1-2 days) on new balance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CardStatChange:
    """Represents a stat change for a single card."""

    card_id: int
    card_name: str
    old_hp: float
    new_hp: float
    old_damage: float
    new_damage: float
    old_hit_speed: float
    new_hit_speed: float
    is_new_card: bool = False
    similar_card_id: int | None = None  # for new card initialization


def apply_surgery(
    model: nn.Module,
    changes: list[CardStatChange],
    embedding_layer_name: str = "card_embedding",
) -> None:
    """Apply surgery to model weights based on balance patch changes.

    For each changed card:
    - Scale the card's embedding weights proportionally to stat changes
    - The intuition: if a card's HP doubled, its embedding should roughly
      double in the HP-correlated dimensions

    For new cards:
    - Initialize embedding as weighted average of similar existing cards
    """
    # Find the card embedding layer
    embedding = None
    for name, module in model.named_modules():
        if embedding_layer_name in name and isinstance(module, nn.Embedding):
            embedding = module
            break

    if embedding is None:
        logger.warning(f"Could not find embedding layer '{embedding_layer_name}'")
        return

    with torch.no_grad():
        for change in changes:
            if change.is_new_card:
                _init_new_card(embedding, change)
            else:
                _scale_existing_card(embedding, change)

    logger.info(f"Applied surgery for {len(changes)} card changes")


def _scale_existing_card(embedding: nn.Embedding, change: CardStatChange) -> None:
    """Scale existing card's embedding based on stat changes."""
    idx = change.card_id

    if idx >= embedding.num_embeddings:
        logger.warning(f"Card {change.card_name} (id={idx}) exceeds embedding size")
        return

    # Compute overall stat multiplier
    # Use geometric mean of stat ratios to avoid bias
    ratios = []
    if change.old_hp > 0 and change.new_hp > 0:
        ratios.append(change.new_hp / change.old_hp)
    if change.old_damage > 0 and change.new_damage > 0:
        ratios.append(change.new_damage / change.old_damage)
    if change.old_hit_speed > 0 and change.new_hit_speed > 0:
        # Inverse: faster hit speed = stronger
        ratios.append(change.old_hit_speed / change.new_hit_speed)

    if not ratios:
        return

    # Geometric mean
    overall_ratio = np.prod(ratios) ** (1.0 / len(ratios))

    # Apply subtle scaling (don't want to distort too much)
    # Scale by sqrt of ratio to be conservative
    scale = overall_ratio ** 0.5

    # Clamp to prevent extreme modifications
    scale = max(0.7, min(1.5, scale))

    embedding.weight[idx] *= scale
    logger.info(
        f"Scaled {change.card_name} embedding by {scale:.3f} "
        f"(HP: {change.old_hp:.0f}->{change.new_hp:.0f}, "
        f"DMG: {change.old_damage:.0f}->{change.new_damage:.0f})"
    )


def _init_new_card(embedding: nn.Embedding, change: CardStatChange) -> None:
    """Initialize a new card's embedding from a similar existing card."""
    idx = change.card_id

    # Extend embedding if needed
    if idx >= embedding.num_embeddings:
        old_weight = embedding.weight.data.clone()
        new_embed = nn.Embedding(idx + 1, embedding.embedding_dim)
        new_embed.weight.data[:old_weight.shape[0]] = old_weight
        embedding.weight = new_embed.weight
        embedding.num_embeddings = idx + 1

    if change.similar_card_id is not None and change.similar_card_id < embedding.num_embeddings:
        # Copy from similar card
        embedding.weight.data[idx] = embedding.weight.data[change.similar_card_id].clone()
        logger.info(
            f"Initialized {change.card_name} from card {change.similar_card_id}"
        )
    else:
        # Random initialization around the mean of all card embeddings
        mean_embed = embedding.weight.data.mean(dim=0)
        std_embed = embedding.weight.data.std(dim=0)
        embedding.weight.data[idx] = mean_embed + std_embed * torch.randn_like(mean_embed) * 0.1
        logger.info(f"Initialized {change.card_name} from mean embedding + noise")


class MetaTracker:
    """Tracks current meta from live API data.

    Periodically fetches card usage rates and win rates from RoyaleAPI
    to bias training toward meta-relevant matchups.
    """

    def __init__(self) -> None:
        self.card_usage: dict[str, float] = {}  # card_name -> usage rate
        self.card_win_rate: dict[str, float] = {}  # card_name -> win rate
        self.deck_usage: dict[str, float] = {}  # deck_hash -> usage rate
        self.last_update: float = 0

    def get_meta_weight(self, card_names: list[str]) -> float:
        """Get meta-relevance weight for a deck.

        Higher weight = more meta-relevant = should be trained more.
        """
        if not self.card_usage:
            return 1.0

        # Average usage rate of cards in the deck
        weights = []
        for name in card_names:
            usage = self.card_usage.get(name, 0.01)
            weights.append(usage)

        return sum(weights) / len(weights) if weights else 1.0

    def load_from_file(self, path: str) -> None:
        """Load meta data from a saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.card_usage = data.get("card_usage", {})
        self.card_win_rate = data.get("card_win_rate", {})
        self.deck_usage = data.get("deck_usage", {})

    def save_to_file(self, path: str) -> None:
        """Save meta data to JSON."""
        data = {
            "card_usage": self.card_usage,
            "card_win_rate": self.card_win_rate,
            "deck_usage": self.deck_usage,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
