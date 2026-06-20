"""Domain Randomization for sim-to-real robustness.

Randomizes card stats during training so the agent learns policies
that are robust to simulator inaccuracies. Inspired by OpenAI Five
(Dota 2) and OpenAI's Rubik's Cube hand (ADR).

Usage:
    from training.domain_randomization import DomainRandomizer
    randomizer = DomainRandomizer(strength=0.1)
    randomized_game = randomizer.apply(game)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class DomainRandomizer:
    """Randomizes game parameters to improve sim-to-real transfer.

    Applies multiplicative noise to card stats during training.
    At test time, set strength=0 to use nominal stats.

    OpenAI Five approach: "when a test player was consistently beating
    our bot, we increased our training randomizations and the test
    player started to lose."
    """

    strength: float = 0.1  # ±10% randomization range
    randomize_hp: bool = True
    randomize_damage: bool = True
    randomize_speed: bool = True
    randomize_hit_speed: bool = True
    randomize_range: bool = False  # conservative: range is very sensitive
    randomize_elixir_regen: bool = True
    randomize_deploy_time: bool = True

    # Automatic Domain Randomization (ADR) state
    # If enabled, strength increases as agent improves
    adr_enabled: bool = False
    adr_min_strength: float = 0.02
    adr_max_strength: float = 0.25
    adr_step: float = 0.01  # increase per performance threshold hit

    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def _noise(self) -> float:
        """Generate multiplicative noise factor: 1.0 ± strength."""
        return 1.0 + self._rng.uniform(-self.strength, self.strength)

    def randomize_entity_stats(self, entity) -> None:
        """Apply randomization to a single entity's stats in-place."""
        if self.strength <= 0:
            return

        if self.randomize_hp:
            factor = self._noise()
            entity.hp *= factor
            entity.max_hp *= factor

        if self.randomize_damage:
            entity.dps *= self._noise()

        if self.randomize_speed and entity.speed > 0:
            entity.speed *= self._noise()
            entity.base_speed = entity.speed

        if self.randomize_hit_speed and entity.attack_interval > 0:
            entity.attack_interval *= self._noise()

        if self.randomize_range and entity.attack_range > 0:
            entity.attack_range *= self._noise()

        if self.randomize_deploy_time and entity.deploy_timer > 0:
            entity.deploy_timer *= self._noise()

    def randomize_game(self, game) -> None:
        """Apply randomization to all entities in a game."""
        if self.strength <= 0:
            return

        for entity in game.entities:
            if entity.alive:
                self.randomize_entity_stats(entity)

        # Randomize elixir regen rate slightly
        if self.randomize_elixir_regen:
            for ps in game.players:
                ps.elixir *= self._noise()

    def step_adr(self, win_rate: float, threshold: float = 0.55) -> None:
        """Automatic Domain Randomization: increase difficulty when agent improves.

        Call after each evaluation round. If win_rate > threshold,
        increase randomization strength.
        """
        if not self.adr_enabled:
            return

        if win_rate > threshold:
            self.strength = min(self.strength + self.adr_step, self.adr_max_strength)
        elif win_rate < threshold - 0.1:
            self.strength = max(self.strength - self.adr_step * 0.5, self.adr_min_strength)
