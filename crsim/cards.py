"""Card definitions and stats for all 20 core cards."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from crsim.constants import SPEED_FAST, SPEED_MEDIUM, SPEED_SLOW, SPEED_VERY_FAST


class CardType(enum.IntEnum):
    KNIGHT = 0
    ARCHERS = 1
    MUSKETEER = 2
    GIANT = 3
    MINI_PEKKA = 4
    VALKYRIE = 5
    WIZARD = 6
    HOG_RIDER = 7
    MINIONS = 8
    BABY_DRAGON = 9
    SKELETON_ARMY = 10
    GOBLIN_BARREL = 11
    FIREBALL = 12
    ARROWS = 13
    ZAP = 14
    LIGHTNING = 15
    CANNON = 16
    INFERNO_TOWER = 17
    TOMBSTONE = 18
    ELIXIR_COLLECTOR = 19


class EntityKind(enum.IntEnum):
    TROOP = 0
    SPELL = 1
    BUILDING = 2


class TargetMode(enum.IntEnum):
    GROUND = 0
    AIR_GROUND = 1
    BUILDINGS = 2
    AREA = 3  # spells


@dataclass(frozen=True, slots=True)
class CardDef:
    """Immutable definition of a card's base stats."""

    card_type: CardType
    kind: EntityKind
    cost: int
    hp: float
    dps: float
    attack_interval: float  # seconds between attacks (0 for spells)
    speed: float  # tiles per second (0 for buildings / spells)
    attack_range: float  # tiles
    target_mode: TargetMode
    is_flying: bool = False
    is_splash: bool = False
    splash_radius: float = 0.0
    spawn_count: int = 1  # how many entities a single play spawns
    spawn_hp: float = 0.0  # per-spawned-unit HP (if spawn_count > 1)
    spawn_dps: float = 0.0
    building_lifetime: float = 0.0  # seconds; 0 = not a building
    spawner_interval: float = 0.0  # seconds between spawner spawns
    # Inferno tower ramp
    inferno_dps_min: float = 0.0
    inferno_dps_max: float = 0.0

    @property
    def damage_per_hit(self) -> float:
        if self.attack_interval <= 0:
            return self.dps  # spells: dps field stores total damage
        return self.dps * self.attack_interval


# ---------------------------------------------------------------------------
# Card registry
# ---------------------------------------------------------------------------

CARD_DEFS: dict[CardType, CardDef] = {}


def _register(d: CardDef) -> None:
    CARD_DEFS[d.card_type] = d


# --- Troops ----------------------------------------------------------------

_register(CardDef(
    card_type=CardType.KNIGHT, kind=EntityKind.TROOP, cost=3,
    hp=1452, dps=167, attack_interval=1.1, speed=SPEED_MEDIUM,
    attack_range=1.2, target_mode=TargetMode.GROUND,
))

_register(CardDef(
    card_type=CardType.ARCHERS, kind=EntityKind.TROOP, cost=3,
    hp=304, dps=107, attack_interval=1.1, speed=SPEED_MEDIUM,
    attack_range=5.0, target_mode=TargetMode.AIR_GROUND,
    spawn_count=2, spawn_hp=304, spawn_dps=107,
))

_register(CardDef(
    card_type=CardType.MUSKETEER, kind=EntityKind.TROOP, cost=4,
    hp=598, dps=176, attack_interval=1.0, speed=SPEED_MEDIUM,
    attack_range=6.0, target_mode=TargetMode.AIR_GROUND,
))

_register(CardDef(
    card_type=CardType.GIANT, kind=EntityKind.TROOP, cost=5,
    hp=3344, dps=120, attack_interval=1.5, speed=SPEED_SLOW,
    attack_range=1.2, target_mode=TargetMode.BUILDINGS,
))

_register(CardDef(
    card_type=CardType.MINI_PEKKA, kind=EntityKind.TROOP, cost=4,
    hp=1056, dps=325, attack_interval=1.7, speed=SPEED_FAST,
    attack_range=1.2, target_mode=TargetMode.GROUND,
))

_register(CardDef(
    card_type=CardType.VALKYRIE, kind=EntityKind.TROOP, cost=4,
    hp=1654, dps=126, attack_interval=1.5, speed=SPEED_MEDIUM,
    attack_range=1.2, target_mode=TargetMode.GROUND,
    is_splash=True, splash_radius=1.5,
))

_register(CardDef(
    card_type=CardType.WIZARD, kind=EntityKind.TROOP, cost=5,
    hp=598, dps=176, attack_interval=1.4, speed=SPEED_MEDIUM,
    attack_range=5.5, target_mode=TargetMode.AIR_GROUND,
    is_splash=True, splash_radius=1.5,
))

_register(CardDef(
    card_type=CardType.HOG_RIDER, kind=EntityKind.TROOP, cost=4,
    hp=1408, dps=176, attack_interval=1.5, speed=SPEED_VERY_FAST,
    attack_range=1.2, target_mode=TargetMode.BUILDINGS,
))

_register(CardDef(
    card_type=CardType.MINIONS, kind=EntityKind.TROOP, cost=3,
    hp=190, dps=84, attack_interval=1.0, speed=SPEED_FAST,
    attack_range=2.0, target_mode=TargetMode.AIR_GROUND,
    is_flying=True, spawn_count=3, spawn_hp=190, spawn_dps=84,
))

_register(CardDef(
    card_type=CardType.BABY_DRAGON, kind=EntityKind.TROOP, cost=4,
    hp=1064, dps=100, attack_interval=1.5, speed=SPEED_FAST,
    attack_range=3.5, target_mode=TargetMode.AIR_GROUND,
    is_flying=True, is_splash=True, splash_radius=1.5,
))

_register(CardDef(
    card_type=CardType.SKELETON_ARMY, kind=EntityKind.TROOP, cost=3,
    hp=0, dps=0, attack_interval=1.0, speed=SPEED_FAST,
    attack_range=1.0, target_mode=TargetMode.GROUND,
    spawn_count=15, spawn_hp=67, spawn_dps=67,
))

_register(CardDef(
    card_type=CardType.GOBLIN_BARREL, kind=EntityKind.SPELL, cost=3,
    hp=0, dps=0, attack_interval=1.1, speed=SPEED_FAST,
    attack_range=1.0, target_mode=TargetMode.GROUND,
    spawn_count=3, spawn_hp=167, spawn_dps=99,
))

# --- Spells ----------------------------------------------------------------

_register(CardDef(
    card_type=CardType.FIREBALL, kind=EntityKind.SPELL, cost=4,
    hp=0, dps=572, attack_interval=0, speed=0,
    attack_range=2.5, target_mode=TargetMode.AREA,
))

_register(CardDef(
    card_type=CardType.ARROWS, kind=EntityKind.SPELL, cost=3,
    hp=0, dps=243, attack_interval=0, speed=0,
    attack_range=4.0, target_mode=TargetMode.AREA,
))

_register(CardDef(
    card_type=CardType.ZAP, kind=EntityKind.SPELL, cost=2,
    hp=0, dps=159, attack_interval=0, speed=0,
    attack_range=2.5, target_mode=TargetMode.AREA,
))

_register(CardDef(
    card_type=CardType.LIGHTNING, kind=EntityKind.SPELL, cost=6,
    hp=0, dps=877, attack_interval=0, speed=0,
    attack_range=3.5, target_mode=TargetMode.AREA,
))

# --- Buildings -------------------------------------------------------------

_register(CardDef(
    card_type=CardType.CANNON, kind=EntityKind.BUILDING, cost=3,
    hp=742, dps=127, attack_interval=0.8, speed=0,
    attack_range=5.5, target_mode=TargetMode.GROUND,
    building_lifetime=30.0,
))

_register(CardDef(
    card_type=CardType.INFERNO_TOWER, kind=EntityKind.BUILDING, cost=5,
    hp=1408, dps=40, attack_interval=0.4, speed=0,
    attack_range=6.0, target_mode=TargetMode.AIR_GROUND,
    building_lifetime=35.0,
    inferno_dps_min=40, inferno_dps_max=400,
))

_register(CardDef(
    card_type=CardType.TOMBSTONE, kind=EntityKind.BUILDING, cost=3,
    hp=422, dps=0, attack_interval=0, speed=0,
    attack_range=0, target_mode=TargetMode.GROUND,
    building_lifetime=40.0, spawner_interval=3.5,
))

_register(CardDef(
    card_type=CardType.ELIXIR_COLLECTOR, kind=EntityKind.BUILDING, cost=6,
    hp=888, dps=0, attack_interval=0, speed=0,
    attack_range=0, target_mode=TargetMode.GROUND,
    building_lifetime=70.0, spawner_interval=8.5,
))
