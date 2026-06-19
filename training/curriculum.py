"""Deck archetype curriculum for self-play training.

Instead of random 8-card decks from 42 cards, use real deck archetypes
that represent how the game is actually played. This dramatically
improves training quality because:
  1. Agents learn real strategies instead of random card soup
  2. Cards interact correctly (e.g., Giant + Musketeer push)
  3. Deck balance means games are competitive, not one-sided stomps
"""

from __future__ import annotations

import random

from crsim.cards import CardType

# Real competitive deck archetypes with synergistic card combinations
DECK_ARCHETYPES: list[tuple[str, list[CardType]]] = [
    ("Hog 2.6 Cycle", [
        CardType.HOG_RIDER, CardType.MUSKETEER, CardType.ICE_SPIRIT,
        CardType.CANNON, CardType.FIREBALL, CardType.LOG,
        CardType.GOBLINS, CardType.KNIGHT,
    ]),
    ("Giant Beatdown", [
        CardType.GIANT, CardType.MUSKETEER, CardType.WIZARD,
        CardType.MINI_PEKKA, CardType.FIREBALL, CardType.ZAP,
        CardType.ARCHERS, CardType.TOMBSTONE,
    ]),
    ("Golem Beatdown", [
        CardType.GOLEM, CardType.BABY_DRAGON, CardType.MEGA_KNIGHT,
        CardType.WITCH, CardType.LIGHTNING, CardType.TORNADO,
        CardType.GOBLINS, CardType.TOMBSTONE,
    ]),
    ("PEKKA Bridge Spam", [
        CardType.PEKKA, CardType.BANDIT, CardType.DARK_PRINCE,
        CardType.ELECTRO_WIZARD, CardType.MINIONS, CardType.POISON,
        CardType.FIREBALL, CardType.ZAP,
    ]),
    ("Lava Hound", [
        CardType.LAVA_HOUND, CardType.BALLOON, CardType.MINIONS,
        CardType.BABY_DRAGON, CardType.TOMBSTONE, CardType.FIREBALL,
        CardType.ARROWS, CardType.GUARDS,
    ]),
    ("Log Bait", [
        CardType.GOBLIN_BARREL, CardType.GUARDS, CardType.PRINCE,
        CardType.INFERNO_TOWER, CardType.FIREBALL, CardType.LOG,
        CardType.ICE_SPIRIT, CardType.KNIGHT,
    ]),
    ("Sparky", [
        CardType.SPARKY, CardType.GIANT, CardType.WIZARD,
        CardType.MINIONS, CardType.ZAP, CardType.FIREBALL,
        CardType.GOBLINS, CardType.CANNON,
    ]),
    ("Mega Knight Control", [
        CardType.MEGA_KNIGHT, CardType.BANDIT, CardType.INFERNO_TOWER,
        CardType.SPEAR_GOBLINS, CardType.ZAP, CardType.FIREBALL,
        CardType.MINIONS, CardType.GOBLINS,
    ]),
    ("Prince Double Prince", [
        CardType.PRINCE, CardType.DARK_PRINCE, CardType.GIANT,
        CardType.MUSKETEER, CardType.ZAP, CardType.FIREBALL,
        CardType.GOBLINS, CardType.ARCHERS,
    ]),
    ("Ice Wizard Control", [
        CardType.ICE_WIZARD, CardType.TORNADO, CardType.VALKYRIE,
        CardType.MUSKETEER, CardType.HOG_RIDER, CardType.FIREBALL,
        CardType.LOG, CardType.GOBLINS,
    ]),
    ("Balloon Freeze", [
        CardType.BALLOON, CardType.FREEZE, CardType.KNIGHT,
        CardType.MUSKETEER, CardType.MINIONS, CardType.ARROWS,
        CardType.TOMBSTONE, CardType.FIRE_SPIRITS,
    ]),
    ("Tesla Cycle", [
        CardType.TESLA, CardType.HOG_RIDER, CardType.ICE_SPIRIT,
        CardType.GOBLINS, CardType.FIREBALL, CardType.LOG,
        CardType.ARCHERS, CardType.KNIGHT,
    ]),
    ("PEKKA Hog", [
        CardType.PEKKA, CardType.HOG_RIDER, CardType.ELECTRO_WIZARD,
        CardType.MINIONS, CardType.POISON, CardType.ZAP,
        CardType.GOBLINS, CardType.CANNON,
    ]),
    ("Giant Poison", [
        CardType.GIANT, CardType.POISON, CardType.MUSKETEER,
        CardType.DARK_PRINCE, CardType.GUARDS, CardType.ZAP,
        CardType.MEGA_KNIGHT, CardType.ARCHERS,
    ]),
    ("Golem Clone", [
        CardType.GOLEM, CardType.BABY_DRAGON, CardType.WITCH,
        CardType.LIGHTNING, CardType.TORNADO, CardType.GUARDS,
        CardType.SPEAR_GOBLINS, CardType.ELIXIR_COLLECTOR,
    ]),
]


class DeckCurriculum:
    """Manage deck selection during self-play training.

    Phases:
      1. Early training: Use only simple archetypes (Hog, Giant, Log Bait)
      2. Mid training: Mix in more complex archetypes
      3. Late training: Full archetype pool + some random decks for robustness
    """

    def __init__(self, all_card_types: list[CardType] | None = None) -> None:
        self.all_cards = all_card_types or list(CardType)
        self.step = 0

    def select_deck(self) -> list[CardType]:
        """Select a deck based on current training phase."""
        phase = self._current_phase()

        if phase == 0:
            # Early: simple archetypes only
            pool = DECK_ARCHETYPES[:5]
        elif phase == 1:
            # Mid: all archetypes
            pool = DECK_ARCHETYPES
        else:
            # Late: 80% archetype, 20% random for robustness
            if random.random() < 0.2:
                return self._random_deck()
            pool = DECK_ARCHETYPES

        _, deck = random.choice(pool)
        return list(deck)

    def select_matchup(self) -> tuple[list[CardType], list[CardType]]:
        """Select both decks for a self-play game."""
        return self.select_deck(), self.select_deck()

    def advance(self) -> None:
        self.step += 1

    def _current_phase(self) -> int:
        if self.step < 10_000:
            return 0
        if self.step < 50_000:
            return 1
        return 2

    def _random_deck(self) -> list[CardType]:
        """Generate a random but somewhat balanced deck."""
        cards = list(self.all_cards)
        random.shuffle(cards)
        return cards[:8]
