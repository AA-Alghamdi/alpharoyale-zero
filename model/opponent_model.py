"""Opponent modeling — track opponent's play history and infer hidden state.

In real Clash Royale:
  - You can't see opponent's hand (which 4 of 8 cards they hold)
  - You can't see opponent's elixir count (must estimate)
  - You don't know their deck until cards are played

This module provides:
  1. PlayHistoryTracker: records what cards the opponent played and when
  2. OpponentBeliefState: infers likely hand/deck/elixir from observations
  3. Feature vectors for the neural network
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crsim.constants import (
    MAX_ELIXIR,
    NUM_CARD_TYPES,
    TICK_DURATION,
)

PLAY_HISTORY_LEN = 20  # remember last N opponent card plays
BELIEF_FEATURE_DIM = 128  # output feature vector size


@dataclass
class CardPlayEvent:
    card_type: int
    tick: int
    x: float
    y: float


class PlayHistoryTracker:
    """Track opponent's card play history during a game."""

    def __init__(self) -> None:
        self.history: list[CardPlayEvent] = []
        self.cards_seen: set[int] = set()
        self.last_play_tick: dict[int, int] = {}  # card_type → last tick played

    def record_play(
        self, card_type: int, tick: int, x: float, y: float,
    ) -> None:
        event = CardPlayEvent(card_type, tick, x, y)
        self.history.append(event)
        self.cards_seen.add(card_type)
        self.last_play_tick[card_type] = tick

    def reset(self) -> None:
        self.history.clear()
        self.cards_seen.clear()
        self.last_play_tick.clear()


class OpponentBeliefState:
    """Infer opponent's hidden state from observations.

    Tracks:
      - Known cards in opponent's deck (from observations)
      - Estimated elixir count
      - Card cycle position (which card is coming next)
      - Win condition / deck archetype probability
    """

    def __init__(self) -> None:
        self.tracker = PlayHistoryTracker()
        self.estimated_elixir: float = 5.0
        self.elixir_regen_rate: float = TICK_DURATION / 2.8
        self.deck_probs: np.ndarray = np.ones(NUM_CARD_TYPES, dtype=np.float32)
        self.deck_probs /= self.deck_probs.sum()
        self._last_tick: int = 0

    def observe_play(
        self,
        card_type: int,
        cost: int,
        tick: int,
        x: float,
        y: float,
    ) -> None:
        """Record opponent playing a card."""
        self.tracker.record_play(card_type, tick, x, y)
        self.estimated_elixir -= cost
        self.estimated_elixir = max(0.0, self.estimated_elixir)

        # Increase probability for cards we've seen in the deck
        if 0 <= card_type < NUM_CARD_TYPES:
            self.deck_probs[card_type] = min(
                self.deck_probs[card_type] + 0.3, 1.0,
            )
            self.deck_probs /= self.deck_probs.sum()

    def tick_update(self, current_tick: int) -> None:
        """Update elixir estimate based on time elapsed."""
        elapsed_ticks = current_tick - self._last_tick
        self._last_tick = current_tick
        self.estimated_elixir += self.elixir_regen_rate * elapsed_ticks
        self.estimated_elixir = min(self.estimated_elixir, MAX_ELIXIR)

    def get_cycle_features(self, current_tick: int) -> np.ndarray:
        """Compute card-cycle features: how soon each seen card might return.

        In CR, after playing card X, it goes to end of queue.
        With 4 in hand and 8 in deck, it returns after 4 more plays.
        """
        cycle = np.zeros(NUM_CARD_TYPES, dtype=np.float32)
        for ct, last_tick in self.tracker.last_play_tick.items():
            if 0 <= ct < NUM_CARD_TYPES:
                ticks_since = current_tick - last_tick
                cycle[ct] = min(ticks_since / 20.0, 1.0)
        return cycle

    def encode(self, current_tick: int) -> np.ndarray:
        """Encode belief state as a feature vector for the NN.

        Returns (BELIEF_FEATURE_DIM,) float32 vector:
          [0:NUM_CARD_TYPES]     — deck probability per card type
          [NUM_CARD_TYPES:2*N]   — cycle features (time since last play)
          [2*N:2*N+1]           — estimated elixir (normalized)
          [2*N+1:2*N+2]         — number of unique cards seen (normalized)
          [2*N+2:]              — zero-padded
        """
        n = NUM_CARD_TYPES
        features = np.zeros(BELIEF_FEATURE_DIM, dtype=np.float32)

        # Deck probabilities
        features[:n] = self.deck_probs

        # Cycle features
        features[n:2 * n] = self.get_cycle_features(current_tick)

        # Estimated elixir
        features[2 * n] = self.estimated_elixir / MAX_ELIXIR

        # Cards seen count
        features[2 * n + 1] = len(self.tracker.cards_seen) / 8.0

        # Recent play count (last 20 ticks)
        recent = sum(
            1 for e in self.tracker.history
            if current_tick - e.tick < 20
        )
        features[2 * n + 2] = min(recent / 5.0, 1.0)

        return features

    def reset(self) -> None:
        self.tracker.reset()
        self.estimated_elixir = 5.0
        self.deck_probs = np.ones(NUM_CARD_TYPES, dtype=np.float32)
        self.deck_probs /= self.deck_probs.sum()
        self._last_tick = 0
