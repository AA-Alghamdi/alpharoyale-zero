"""Runtime entity representation — troops, buildings, towers, and spell effects."""

from __future__ import annotations

import math
from dataclasses import dataclass

from crsim.cards import CardDef, CardType, EntityKind, TargetMode


@dataclass(slots=True)
class Entity:
    """A live entity on the battlefield."""

    eid: int  # unique id within a game
    owner: int  # 0 or 1
    card_type: CardType
    kind: EntityKind

    # Position
    x: float
    y: float

    # Combat stats (mutable copies from CardDef)
    hp: float
    max_hp: float
    dps: float
    attack_interval: float  # seconds
    attack_timer: float  # seconds until next attack; 0 = ready
    attack_range: float
    target_mode: TargetMode
    speed: float  # tiles / second

    is_flying: bool = False
    is_splash: bool = False
    splash_radius: float = 0.0

    # Building fields
    is_building: bool = False
    building_timer: float = 0.0  # remaining lifetime (seconds)
    spawner_timer: float = 0.0
    spawner_interval: float = 0.0

    # Inferno ramp
    inferno_dps_min: float = 0.0
    inferno_dps_max: float = 0.0
    inferno_ramp_time: float = 0.0  # seconds locked onto current target

    # Targeting
    target_eid: int = -1  # -1 = no target

    # Is a tower?
    is_tower: bool = False
    is_king_tower: bool = False

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def damage_per_hit(self) -> float:
        if self.attack_interval <= 0:
            return 0.0
        return self.dps * self.attack_interval

    def distance_to(self, other: Entity) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_to_pos(self, px: float, py: float) -> float:
        dx = self.x - px
        dy = self.y - py
        return math.sqrt(dx * dx + dy * dy)


def entity_from_card(
    eid: int,
    owner: int,
    card_def: CardDef,
    x: float,
    y: float,
    hp_override: float = 0.0,
    dps_override: float = 0.0,
) -> Entity:
    """Create a live Entity from a CardDef at a given position."""
    hp = hp_override if hp_override > 0 else card_def.hp
    dps = dps_override if dps_override > 0 else card_def.dps
    return Entity(
        eid=eid,
        owner=owner,
        card_type=card_def.card_type,
        kind=card_def.kind,
        x=x,
        y=y,
        hp=hp,
        max_hp=hp,
        dps=dps,
        attack_interval=card_def.attack_interval,
        attack_timer=card_def.attack_interval,  # first attack after interval
        attack_range=card_def.attack_range,
        target_mode=card_def.target_mode,
        speed=card_def.speed,
        is_flying=card_def.is_flying,
        is_splash=card_def.is_splash,
        splash_radius=card_def.splash_radius,
        is_building=card_def.kind == EntityKind.BUILDING,
        building_timer=card_def.building_lifetime,
        spawner_interval=card_def.spawner_interval,
        spawner_timer=card_def.spawner_interval,
        inferno_dps_min=card_def.inferno_dps_min,
        inferno_dps_max=card_def.inferno_dps_max,
    )


def make_tower(
    eid: int,
    owner: int,
    x: float,
    y: float,
    hp: float,
    damage: float,
    attack_range: float,
    attack_interval: float,
    is_king: bool = False,
) -> Entity:
    """Create a tower entity."""
    return Entity(
        eid=eid,
        owner=owner,
        card_type=CardType.KNIGHT,  # placeholder; towers have no card type
        kind=EntityKind.BUILDING,
        x=x,
        y=y,
        hp=hp,
        max_hp=hp,
        dps=damage / attack_interval if attack_interval > 0 else 0,
        attack_interval=attack_interval,
        attack_timer=attack_interval,
        attack_range=attack_range,
        target_mode=TargetMode.AIR_GROUND,
        speed=0.0,
        is_building=True,
        is_tower=True,
        is_king_tower=is_king,
    )
