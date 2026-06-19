"""Card definitions and stats for all cards."""

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
    # --- Expansion: 22 new cards ---
    PRINCE = 20
    DARK_PRINCE = 21
    PEKKA = 22
    GOLEM = 23
    LAVA_HOUND = 24
    BALLOON = 25
    WITCH = 26
    ICE_WIZARD = 27
    ELECTRO_WIZARD = 28
    BANDIT = 29
    MEGA_KNIGHT = 30
    SPARKY = 31
    GUARDS = 32
    GOBLINS = 33
    SPEAR_GOBLINS = 34
    FIRE_SPIRITS = 35
    ICE_SPIRIT = 36
    LOG = 37
    TORNADO = 38
    POISON = 39
    FREEZE = 40
    TESLA = 41


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
    # New mechanics
    deploy_time: float = 1.0  # seconds after placement before unit is active
    has_charge: bool = False  # Prince/Dark Prince charge mechanic
    charge_speed: float = 0.0  # tiles/sec when charging
    charge_damage_mult: float = 2.0  # damage multiplier on charge hit
    has_shield: bool = False  # Dark Prince/Guards shield
    shield_hp: float = 0.0
    has_knockback: bool = False  # Fireball, Log knockback
    knockback_distance: float = 0.0
    stuns: bool = False  # Zap, E-Wiz stun
    stun_duration: float = 0.0  # seconds
    resets_inferno: bool = False  # Zap, Lightning, E-Wiz
    # Death spawns (Golem, Lava Hound)
    death_spawn_type: CardType | None = None
    death_spawn_count: int = 0
    death_spawn_hp: float = 0.0
    death_spawn_dps: float = 0.0
    # Spell travel time
    spell_travel_time: float = 0.0  # seconds before spell hits
    # AoE damage over time (Poison)
    dot_damage: float = 0.0
    dot_duration: float = 0.0

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

# --- Expansion cards -------------------------------------------------------

# Prince: charge mechanic, 2× first hit
_register(CardDef(
    card_type=CardType.PRINCE, kind=EntityKind.TROOP, cost=5,
    hp=1615, dps=247, attack_interval=1.4, speed=SPEED_MEDIUM,
    attack_range=1.85, target_mode=TargetMode.GROUND,
    has_charge=True, charge_speed=SPEED_VERY_FAST * 2,
    charge_damage_mult=2.0,
))

# Dark Prince: charge + shield
_register(CardDef(
    card_type=CardType.DARK_PRINCE, kind=EntityKind.TROOP, cost=4,
    hp=970, dps=156, attack_interval=1.3, speed=SPEED_MEDIUM,
    attack_range=1.2, target_mode=TargetMode.GROUND,
    is_splash=True, splash_radius=1.0,
    has_charge=True, charge_speed=SPEED_VERY_FAST * 2,
    charge_damage_mult=2.0, has_shield=True, shield_hp=199,
))

# PEKKA: slow tank, huge damage
_register(CardDef(
    card_type=CardType.PEKKA, kind=EntityKind.TROOP, cost=7,
    hp=3760, dps=329, attack_interval=1.8, speed=SPEED_SLOW,
    attack_range=1.2, target_mode=TargetMode.GROUND,
))

# Golem: splits into 2 Golemites on death
_register(CardDef(
    card_type=CardType.GOLEM, kind=EntityKind.TROOP, cost=8,
    hp=4256, dps=80, attack_interval=2.5, speed=SPEED_SLOW,
    attack_range=1.2, target_mode=TargetMode.BUILDINGS,
    death_spawn_type=CardType.GOLEM, death_spawn_count=2,
    death_spawn_hp=864, death_spawn_dps=40,
))

# Lava Hound: flying tank, splits into Pups
_register(CardDef(
    card_type=CardType.LAVA_HOUND, kind=EntityKind.TROOP, cost=7,
    hp=3000, dps=34, attack_interval=1.3, speed=SPEED_SLOW,
    attack_range=2.0, target_mode=TargetMode.BUILDINGS,
    is_flying=True,
    death_spawn_type=CardType.LAVA_HOUND, death_spawn_count=6,
    death_spawn_hp=179, death_spawn_dps=45,
))

# Balloon: flying, targets buildings
_register(CardDef(
    card_type=CardType.BALLOON, kind=EntityKind.TROOP, cost=5,
    hp=1396, dps=200, attack_interval=3.0, speed=SPEED_MEDIUM,
    attack_range=0.5, target_mode=TargetMode.BUILDINGS,
    is_flying=True,
))

# Witch: spawns skeletons
_register(CardDef(
    card_type=CardType.WITCH, kind=EntityKind.TROOP, cost=5,
    hp=693, dps=69, attack_interval=0.7, speed=SPEED_MEDIUM,
    attack_range=5.0, target_mode=TargetMode.AIR_GROUND,
    is_splash=True, splash_radius=1.0,
))

# Ice Wizard: slows targets
_register(CardDef(
    card_type=CardType.ICE_WIZARD, kind=EntityKind.TROOP, cost=3,
    hp=590, dps=75, attack_interval=1.5, speed=SPEED_MEDIUM,
    attack_range=5.5, target_mode=TargetMode.AIR_GROUND,
    is_splash=True, splash_radius=1.5,
))

# Electro Wizard: stuns + resets Inferno
_register(CardDef(
    card_type=CardType.ELECTRO_WIZARD, kind=EntityKind.TROOP, cost=4,
    hp=590, dps=100, attack_interval=1.7, speed=SPEED_FAST,
    attack_range=5.0, target_mode=TargetMode.AIR_GROUND,
    stuns=True, stun_duration=0.5, resets_inferno=True,
))

# Bandit: dash mechanic (simplified as charge)
_register(CardDef(
    card_type=CardType.BANDIT, kind=EntityKind.TROOP, cost=3,
    hp=750, dps=160, attack_interval=1.0, speed=SPEED_FAST,
    attack_range=1.2, target_mode=TargetMode.GROUND,
    has_charge=True, charge_speed=SPEED_VERY_FAST * 2.5,
    charge_damage_mult=1.5,
))

# Mega Knight: splash jump, slow/heavy
_register(CardDef(
    card_type=CardType.MEGA_KNIGHT, kind=EntityKind.TROOP, cost=7,
    hp=3300, dps=172, attack_interval=1.8, speed=SPEED_MEDIUM,
    attack_range=1.2, target_mode=TargetMode.GROUND,
    is_splash=True, splash_radius=1.8,
))

# Sparky: slow charge, huge damage
_register(CardDef(
    card_type=CardType.SPARKY, kind=EntityKind.TROOP, cost=6,
    hp=1200, dps=260, attack_interval=4.0, speed=SPEED_SLOW,
    attack_range=4.5, target_mode=TargetMode.GROUND,
    is_splash=True, splash_radius=1.0,
    stuns=True, stun_duration=0.5,
))

# Guards: shield troop
_register(CardDef(
    card_type=CardType.GUARDS, kind=EntityKind.TROOP, cost=3,
    hp=90, dps=65, attack_interval=1.1, speed=SPEED_FAST,
    attack_range=1.2, target_mode=TargetMode.GROUND,
    spawn_count=3, spawn_hp=90, spawn_dps=65,
    has_shield=True, shield_hp=199,
))

# Goblins
_register(CardDef(
    card_type=CardType.GOBLINS, kind=EntityKind.TROOP, cost=2,
    hp=167, dps=99, attack_interval=1.1, speed=SPEED_VERY_FAST,
    attack_range=1.0, target_mode=TargetMode.GROUND,
    spawn_count=3, spawn_hp=167, spawn_dps=99,
))

# Spear Goblins
_register(CardDef(
    card_type=CardType.SPEAR_GOBLINS, kind=EntityKind.TROOP, cost=2,
    hp=110, dps=52, attack_interval=1.1, speed=SPEED_VERY_FAST,
    attack_range=5.0, target_mode=TargetMode.AIR_GROUND,
    spawn_count=3, spawn_hp=110, spawn_dps=52,
))

# Fire Spirits: suicidal splash
_register(CardDef(
    card_type=CardType.FIRE_SPIRITS, kind=EntityKind.TROOP, cost=2,
    hp=91, dps=178, attack_interval=0, speed=SPEED_FAST,
    attack_range=2.0, target_mode=TargetMode.AIR_GROUND,
    is_splash=True, splash_radius=1.5,
    spawn_count=3, spawn_hp=91, spawn_dps=178,
))

# Ice Spirit: suicidal freeze
_register(CardDef(
    card_type=CardType.ICE_SPIRIT, kind=EntityKind.TROOP, cost=1,
    hp=190, dps=91, attack_interval=0, speed=SPEED_VERY_FAST,
    attack_range=2.5, target_mode=TargetMode.AIR_GROUND,
    stuns=True, stun_duration=1.0,
))

# --- New Spells ---

# Log: rolling spell, knockback
_register(CardDef(
    card_type=CardType.LOG, kind=EntityKind.SPELL, cost=2,
    hp=0, dps=240, attack_interval=0, speed=0,
    attack_range=11.1, target_mode=TargetMode.AREA,
    has_knockback=True, knockback_distance=1.5,
    spell_travel_time=2.5,
))

# Tornado: pull enemies to center
_register(CardDef(
    card_type=CardType.TORNADO, kind=EntityKind.SPELL, cost=3,
    hp=0, dps=35, attack_interval=0, speed=0,
    attack_range=5.5, target_mode=TargetMode.AREA,
    dot_damage=35, dot_duration=1.0,
))

# Poison: area damage over time
_register(CardDef(
    card_type=CardType.POISON, kind=EntityKind.SPELL, cost=4,
    hp=0, dps=75, attack_interval=0, speed=0,
    attack_range=3.5, target_mode=TargetMode.AREA,
    dot_damage=75, dot_duration=8.0,
))

# Freeze: stun in area
_register(CardDef(
    card_type=CardType.FREEZE, kind=EntityKind.SPELL, cost=4,
    hp=0, dps=96, attack_interval=0, speed=0,
    attack_range=3.0, target_mode=TargetMode.AREA,
    stuns=True, stun_duration=4.0,
))

# Fireball: add knockback + travel time
CARD_DEFS[CardType.FIREBALL] = CardDef(
    card_type=CardType.FIREBALL, kind=EntityKind.SPELL, cost=4,
    hp=0, dps=572, attack_interval=0, speed=0,
    attack_range=2.5, target_mode=TargetMode.AREA,
    has_knockback=True, knockback_distance=0.5,
    spell_travel_time=1.0,
)

# Arrows: add travel time
CARD_DEFS[CardType.ARROWS] = CardDef(
    card_type=CardType.ARROWS, kind=EntityKind.SPELL, cost=3,
    hp=0, dps=243, attack_interval=0, speed=0,
    attack_range=4.0, target_mode=TargetMode.AREA,
    spell_travel_time=3.0,
)

# Zap: add stun + inferno reset
CARD_DEFS[CardType.ZAP] = CardDef(
    card_type=CardType.ZAP, kind=EntityKind.SPELL, cost=2,
    hp=0, dps=159, attack_interval=0, speed=0,
    attack_range=2.5, target_mode=TargetMode.AREA,
    stuns=True, stun_duration=0.5, resets_inferno=True,
)

# Lightning: add stun + inferno reset
CARD_DEFS[CardType.LIGHTNING] = CardDef(
    card_type=CardType.LIGHTNING, kind=EntityKind.SPELL, cost=6,
    hp=0, dps=877, attack_interval=0, speed=0,
    attack_range=3.5, target_mode=TargetMode.AREA,
    stuns=True, stun_duration=0.5, resets_inferno=True,
)

# --- New Building ---

# Tesla: hides underground
_register(CardDef(
    card_type=CardType.TESLA, kind=EntityKind.BUILDING, cost=4,
    hp=954, dps=152, attack_interval=1.1, speed=0,
    attack_range=5.5, target_mode=TargetMode.AIR_GROUND,
    building_lifetime=30.0,
))
