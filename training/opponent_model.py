"""Opponent Modeling for Imperfect Information.

In Clash Royale, key hidden information includes:
  - Opponent's hand (4 cards visible from 8-card deck)
  - Opponent's elixir count
  - Opponent's next card in cycle

This module implements:
  1. Card tracking: deterministic inference of opponent's remaining cards
  2. Elixir tracking: estimating opponent's current elixir from play timing
  3. Belief network: neural net that predicts opponent's hidden state

Inspired by:
  - ReBeL (Brown & Sandholm 2020): RL+Search with Public Belief States
  - StratFormer (2025): Two-phase curriculum for opponent modeling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CardTracker:
    """Deterministic card tracking for opponent's hand.

    In CR, each player has an 8-card deck. As cards are played,
    they cycle back to the hand after all other cards have been used.
    """

    deck_size: int = 8
    # Observed cards: maps card_id -> times_seen
    observed_cards: dict[int, int] = field(default_factory=dict)
    # Cards currently in cycle (played but not yet recycled)
    in_cycle: list[int] = field(default_factory=list)
    # Known deck (filled as we observe cards)
    known_deck: list[int] = field(default_factory=list)

    def observe_play(self, card_id: int, tick: int) -> None:
        """Record that the opponent played a card."""
        self.observed_cards[card_id] = self.observed_cards.get(card_id, 0) + 1

        if card_id not in self.known_deck:
            self.known_deck.append(card_id)

        self.in_cycle.append(card_id)

        # When cycle is full (4 cards), the first card returns to hand
        if len(self.in_cycle) > 4:
            self.in_cycle.pop(0)

    @property
    def deck_known(self) -> bool:
        """True once we've seen all 8 unique cards."""
        return len(self.known_deck) >= self.deck_size

    @property
    def cards_in_hand(self) -> list[int]:
        """Infer which cards are likely in opponent's hand.

        After seeing all 8 cards, we know exactly which 4 are in hand
        (the ones NOT in the current cycle of 4).
        """
        if not self.deck_known:
            return []
        return [c for c in self.known_deck if c not in self.in_cycle]

    @property
    def next_card(self) -> int | None:
        """Predict the next card in opponent's cycle."""
        if len(self.in_cycle) >= 4 and self.deck_known:
            # Next card is the one that's been out of hand longest
            for card in self.known_deck:
                if card not in self.in_cycle[-3:]:
                    return card
        return None

    def get_probabilities(self, num_card_types: int) -> np.ndarray:
        """Get probability distribution over all card types for opponent's hand.

        Returns shape (num_card_types,) with P(card_in_hand) for each type.
        """
        probs = np.zeros(num_card_types, dtype=np.float32)

        if self.deck_known:
            for card in self.cards_in_hand:
                if card < num_card_types:
                    probs[card] = 1.0
        else:
            # Unknown cards: uniform over unseen card types
            n_unknown = self.deck_size - len(self.known_deck)
            if n_unknown > 0:
                # Known cards in hand get probability 1
                for card in self.known_deck:
                    if card not in self.in_cycle and card < num_card_types:
                        probs[card] = 1.0

                # Unknown slots: uniform prior
                n_hand_unknown = 4 - sum(probs > 0)
                if n_hand_unknown > 0:
                    unseen = [i for i in range(num_card_types) if i not in self.known_deck]
                    if unseen:
                        p = n_hand_unknown / len(unseen)
                        for card in unseen:
                            probs[card] = min(p, 1.0)

        return probs


@dataclass
class ElixirTracker:
    """Estimate opponent's elixir from observed plays.

    We know:
    - Starting elixir: 5
    - Regen rate: 1/2.8s normal, 1/1.4s overtime, 1/0.93s sudden death
    - When opponent plays a card, they spent that card's elixir cost
    """

    current_estimate: float = 5.0
    last_update_tick: int = 0
    confidence: float = 1.0  # decreases over time without observations

    # Tick durations for different phases
    ticks_per_elixir_normal: float = 2.8 / 0.05  # 56 Rust ticks per elixir
    ticks_per_elixir_overtime: float = 1.4 / 0.05  # 28 ticks
    ticks_per_elixir_sudden: float = 0.93 / 0.05  # ~19 ticks

    def observe_play(self, card_cost: int, tick: int, phase: str = "normal") -> None:
        """Update elixir estimate when opponent plays a card."""
        # First, add regen since last update
        self._apply_regen(tick, phase)

        # Subtract card cost
        self.current_estimate = max(0, self.current_estimate - card_cost)
        self.last_update_tick = tick
        self.confidence = min(1.0, self.confidence + 0.2)

    def update(self, tick: int, phase: str = "normal") -> None:
        """Update elixir estimate at current tick (without observing a play)."""
        self._apply_regen(tick, phase)
        # Confidence decays when we don't observe plays
        elapsed = tick - self.last_update_tick
        self.confidence *= 0.999  # slow decay

    def _apply_regen(self, tick: int, phase: str) -> None:
        elapsed = tick - self.last_update_tick
        if elapsed <= 0:
            return

        if phase == "overtime":
            ticks_per = self.ticks_per_elixir_overtime
        elif phase == "sudden_death":
            ticks_per = self.ticks_per_elixir_sudden
        else:
            ticks_per = self.ticks_per_elixir_normal

        regen = elapsed / ticks_per
        self.current_estimate = min(10.0, self.current_estimate + regen)
        self.last_update_tick = tick


class BeliefNetwork(nn.Module):
    """Neural network that predicts opponent's hidden state.

    Input: visible game state (entity features, own hand, visible plays)
    Output:
      - opponent_hand: P(card_i in hand) for each card type
      - opponent_elixir: estimated elixir (regression)
      - opponent_next_play: P(card_i is next play)

    Architecture: small transformer that attends over visible entities
    and play history to infer hidden state.

    Training: during self-play, both players' states are known. We train
    this network to predict the opponent's true state from only the
    visible information (auxiliary loss alongside main RL objective).

    Reference: StratFormer (2025) — two-phase curriculum:
      Phase 1: Train belief net while playing GTO (game-theory optimal)
      Phase 2: Learn to exploit detected opponent patterns
    """

    def __init__(
        self,
        num_card_types: int = 121,
        entity_dim: int = 64,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_card_types = num_card_types

        # Encode play history (sequence of (card_id, tick, x, y))
        self.play_encoder = nn.Sequential(
            nn.Linear(num_card_types + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encode visible game state
        self.state_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer for attending over history + state
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.hand_head = nn.Linear(hidden_dim, num_card_types)  # P(card in hand)
        self.elixir_head = nn.Linear(hidden_dim, 1)  # elixir estimate
        self.next_play_head = nn.Linear(hidden_dim, num_card_types)  # P(next play)

    def forward(
        self,
        state_features: torch.Tensor,
        play_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state_features: (batch, n_entities, entity_dim)
            play_history: (batch, n_plays, num_card_types + 3)

        Returns:
            hand_probs: (batch, num_card_types) — P(card in hand)
            elixir: (batch, 1) — estimated elixir
            next_play_probs: (batch, num_card_types) — P(next card played)
        """
        state_enc = self.state_encoder(state_features)  # (B, E, H)
        play_enc = self.play_encoder(play_history)  # (B, P, H)

        # Concatenate and attend
        combined = torch.cat([state_enc, play_enc], dim=1)  # (B, E+P, H)
        attended = self.transformer(combined)  # (B, E+P, H)

        # Pool over sequence
        pooled = attended.mean(dim=1)  # (B, H)

        hand_logits = self.hand_head(pooled)
        hand_probs = torch.sigmoid(hand_logits)  # independent binary for each card

        elixir = self.elixir_head(pooled)  # regression

        next_play_logits = self.next_play_head(pooled)
        next_play_probs = torch.softmax(next_play_logits, dim=-1)

        return hand_probs, elixir, next_play_probs

    def compute_loss(
        self,
        hand_probs: torch.Tensor,
        elixir_pred: torch.Tensor,
        next_play_probs: torch.Tensor,
        true_hand: torch.Tensor,
        true_elixir: torch.Tensor,
        true_next_play: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary losses for opponent modeling.

        These losses are added to the main RL loss with a small weight.
        """
        hand_loss = nn.functional.binary_cross_entropy(hand_probs, true_hand)
        elixir_loss = nn.functional.mse_loss(elixir_pred.squeeze(-1), true_elixir)
        next_play_loss = nn.functional.cross_entropy(
            torch.log(next_play_probs + 1e-8), true_next_play
        )

        return {
            "hand_loss": hand_loss,
            "elixir_loss": elixir_loss,
            "next_play_loss": next_play_loss,
            "opponent_model_loss": hand_loss + 0.1 * elixir_loss + 0.5 * next_play_loss,
        }
