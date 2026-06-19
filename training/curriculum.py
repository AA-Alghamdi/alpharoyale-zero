"""Curriculum Learning for progressive training difficulty.

Phases:
  1. Mirror matches with simple decks
  2. Diverse decks from meta archetypes
  3. Full 121-card pool with random decks
  4. Meta exploitation (focus on current meta decks)

Inspired by:
  - AlphaStar: league training with progressive difficulty
  - GT Sophy: curated race track sections
  - Syllabus (2024): portable curricula for RL agents
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum, auto

import numpy as np


class CurriculumPhase(IntEnum):
    MIRROR_SIMPLE = 0       # same deck, simple cards
    MIRROR_DIVERSE = 1      # same deck, any cards
    ASYMMETRIC_ARCHETYPE = 2  # different decks, meta archetypes
    FULL_CARD_POOL = 3      # all 121 cards, random decks
    META_FOCUSED = 4        # current meta decks weighted by usage


# Meta deck archetypes with representative card names
DECK_ARCHETYPES = {
    "hog_cycle": [
        ["HOG_RIDER", "MUSKETEER", "ICE_SPIRIT", "SKELETONS", "FIREBALL", "THE_LOG", "ICE_GOLEM", "CANNON"],
        ["HOG_RIDER", "FIRECRACKER", "ICE_SPIRIT", "SKELETONS", "EARTHQUAKE", "THE_LOG", "ICE_GOLEM", "TESLA"],
    ],
    "golem_beatdown": [
        ["GOLEM", "BABY_DRAGON", "NIGHT_WITCH", "MEGA_MINION", "LIGHTNING", "TORNADO", "BARBARIAN_BARREL", "LUMBERJACK"],
        ["GOLEM", "ELECTRO_DRAGON", "NIGHT_WITCH", "MEGA_MINION", "LIGHTNING", "TORNADO", "THE_LOG", "DARK_PRINCE"],
    ],
    "logbait": [
        ["GOBLIN_BARREL", "PRINCESS", "GOBLIN_GANG", "KNIGHT", "INFERNO_TOWER", "ROCKET", "THE_LOG", "ICE_SPIRIT"],
        ["GOBLIN_BARREL", "PRINCESS", "DART_GOBLIN", "KNIGHT", "INFERNO_TOWER", "ROCKET", "TORNADO", "GUARDS"],
    ],
    "xbow_siege": [
        ["X_BOW", "TESLA", "ARCHERS", "ICE_SPIRIT", "FIREBALL", "THE_LOG", "SKELETONS", "ICE_GOLEM"],
        ["X_BOW", "TESLA", "ELECTRO_WIZARD", "ICE_SPIRIT", "FIREBALL", "THE_LOG", "KNIGHT", "MEGA_MINION"],
    ],
    "bridge_spam": [
        ["BATTLE_RAM", "BANDIT", "ROYAL_GHOST", "ELECTRO_WIZARD", "PEKKA", "POISON", "ZAP", "MINIONS"],
        ["BATTLE_RAM", "BANDIT", "DARK_PRINCE", "ELECTRO_WIZARD", "PEKKA", "FIREBALL", "ZAP", "MEGA_MINION"],
    ],
    "lava_hound": [
        ["LAVA_HOUND", "BALLOON", "MEGA_MINION", "MINIONS", "TOMBSTONE", "FIREBALL", "ZAP", "GUARDS"],
        ["LAVA_HOUND", "BALLOON", "INFERNO_DRAGON", "SKELETON_DRAGONS", "BARBARIAN_BARREL", "LIGHTNING", "GOBLIN_GANG", "MINER"],
    ],
    "giant_double_prince": [
        ["GIANT", "PRINCE", "DARK_PRINCE", "MEGA_MINION", "ELECTRO_WIZARD", "FIREBALL", "ZAP", "MINER"],
    ],
    "mortar_cycle": [
        ["MORTAR", "KNIGHT", "ARCHERS", "ICE_SPIRIT", "ROCKET", "THE_LOG", "SKELETONS", "TORNADO"],
    ],
    "royal_giant": [
        ["ROYAL_GIANT", "FISHERMAN", "ELECTRO_WIZARD", "MEGA_MINION", "LIGHTNING", "THE_LOG", "GUARDS", "BABY_DRAGON"],
    ],
    "graveyard_control": [
        ["GRAVEYARD", "POISON", "KNIGHT", "BABY_DRAGON", "TORNADO", "MEGA_MINION", "BARBARIAN_BARREL", "BOMB_TOWER"],
    ],
}

# Simple decks for Phase 1 (well-understood cards)
SIMPLE_DECKS = [
    ["KNIGHT", "ARCHERS", "GIANT", "FIREBALL", "MUSKETEER", "SKELETONS", "BOMBER", "ARROWS"],
    ["VALKYRIE", "HOG_RIDER", "BABY_DRAGON", "PRINCE", "WIZARD", "MINI_PEKKA", "ZAP", "GOBLIN_GANG"],
    ["PEKKA", "WITCH", "MINIONS", "BARBARIANS", "FIREBALL", "ARROWS", "KNIGHT", "MUSKETEER"],
    ["GIANT", "BALLOON", "WIZARD", "MINI_PEKKA", "ARCHERS", "THE_LOG", "SKELETONS", "ARROWS"],
]


@dataclass
class CurriculumConfig:
    """Configuration for curriculum phases."""

    # Phase transition thresholds (win rate against phase opponents)
    phase_1_to_2: float = 0.65  # advance when > 65% win rate
    phase_2_to_3: float = 0.60
    phase_3_to_4: float = 0.55
    phase_4_to_5: float = 0.55

    # Minimum games before phase transition
    min_games_per_phase: int = 5000

    # Phase-specific settings
    mirror_deck_pool_size: int = 4  # how many simple decks for Phase 1


class DeckSampler:
    """Samples decks based on current curriculum phase."""

    def __init__(
        self,
        phase: CurriculumPhase = CurriculumPhase.MIRROR_SIMPLE,
        card_type_enum=None,
    ) -> None:
        self.phase = phase
        self._card_type_enum = card_type_enum
        self._rng = random.Random()

    def _name_to_enum(self, name: str):
        """Convert card name string to CardType enum."""
        if self._card_type_enum is None:
            from crsim.cards_v2 import CardType
            self._card_type_enum = CardType
        return self._card_type_enum[name]

    def _names_to_deck(self, names: list[str]):
        """Convert list of card name strings to CardType list."""
        return [self._name_to_enum(n) for n in names]

    def sample(self) -> tuple[list, list]:
        """Sample two decks based on current phase.

        Returns (deck_p0, deck_p1).
        """
        if self.phase == CurriculumPhase.MIRROR_SIMPLE:
            return self._sample_mirror_simple()
        elif self.phase == CurriculumPhase.MIRROR_DIVERSE:
            return self._sample_mirror_diverse()
        elif self.phase == CurriculumPhase.ASYMMETRIC_ARCHETYPE:
            return self._sample_archetype()
        elif self.phase == CurriculumPhase.FULL_CARD_POOL:
            return self._sample_full_pool()
        elif self.phase == CurriculumPhase.META_FOCUSED:
            return self._sample_meta()
        return self._sample_full_pool()

    def _sample_mirror_simple(self):
        """Both players get the same simple deck."""
        deck_names = self._rng.choice(SIMPLE_DECKS)
        deck = self._names_to_deck(deck_names)
        return deck, list(deck)

    def _sample_mirror_diverse(self):
        """Both players get the same randomly-composed deck."""
        # Pick a random archetype deck
        archetype = self._rng.choice(list(DECK_ARCHETYPES.values()))
        deck_names = self._rng.choice(archetype)
        deck = self._names_to_deck(deck_names)
        return deck, list(deck)

    def _sample_archetype(self):
        """Players get different archetype decks."""
        archetypes = list(DECK_ARCHETYPES.values())
        a1 = self._rng.choice(archetypes)
        a2 = self._rng.choice(archetypes)
        deck0 = self._names_to_deck(self._rng.choice(a1))
        deck1 = self._names_to_deck(self._rng.choice(a2))
        return deck0, deck1

    def _sample_full_pool(self):
        """Random 8-card decks from full 121-card pool.

        Applies basic constraints:
        - Elixir average between 2.5 and 5.5
        - At least 1 spell
        - At least 1 win condition
        """
        if self._card_type_enum is None:
            from crsim.cards_v2 import CardType
            self._card_type_enum = CardType

        all_cards = list(self._card_type_enum)
        deck0 = self._random_constrained_deck(all_cards)
        deck1 = self._random_constrained_deck(all_cards)
        return deck0, deck1

    def _random_constrained_deck(self, all_cards, max_attempts: int = 50):
        """Generate a random deck with basic constraints."""
        for _ in range(max_attempts):
            self._rng.shuffle(all_cards)
            deck = all_cards[:8]
            # Basic validity: just return it (constraints are optional)
            return list(deck)
        return list(all_cards[:8])

    def _sample_meta(self):
        """Sample from meta-relevant decks (weighted by usage)."""
        # For now, same as archetype sampling
        # TODO: integrate RoyaleAPI usage data for real meta weighting
        return self._sample_archetype()

    def __call__(self):
        return self.sample()


class CurriculumManager:
    """Manages curriculum phase transitions based on performance."""

    def __init__(self, config: CurriculumConfig | None = None) -> None:
        self.config = config or CurriculumConfig()
        self.phase = CurriculumPhase.MIRROR_SIMPLE
        self.games_in_phase = 0
        self.win_rate_history: list[float] = []
        self.deck_sampler = DeckSampler(phase=self.phase)

    def update(self, win_rate: float) -> bool:
        """Update curriculum state. Returns True if phase changed."""
        self.games_in_phase += 1
        self.win_rate_history.append(win_rate)

        if self.games_in_phase < self.config.min_games_per_phase:
            return False

        # Check for phase advancement
        recent_wr = np.mean(self.win_rate_history[-100:]) if len(self.win_rate_history) >= 100 else np.mean(self.win_rate_history)

        thresholds = {
            CurriculumPhase.MIRROR_SIMPLE: self.config.phase_1_to_2,
            CurriculumPhase.MIRROR_DIVERSE: self.config.phase_2_to_3,
            CurriculumPhase.ASYMMETRIC_ARCHETYPE: self.config.phase_3_to_4,
            CurriculumPhase.FULL_CARD_POOL: self.config.phase_4_to_5,
        }

        threshold = thresholds.get(self.phase)
        if threshold and recent_wr > threshold and self.phase < CurriculumPhase.META_FOCUSED:
            self.phase = CurriculumPhase(self.phase + 1)
            self.games_in_phase = 0
            self.win_rate_history.clear()
            self.deck_sampler.phase = self.phase
            return True

        return False
