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
    base_speed: float = 0.0  # original speed before slow effects

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

    # New mechanics
    deploy_timer: float = 0.0  # remaining deploy time; >0 means unit not active yet
    is_deployed: bool = True  # False during deploy phase

    # Charge
    is_charging: bool = False
    charge_distance: float = 0.0  # tiles traveled while charging
    charge_speed: float = 0.0
    charge_damage_mult: float = 2.0
    next_hit_is_charge: bool = False

    # Shield
    has_shield: bool = False
    shield_hp: float = 0.0

    # Stun
    stun_timer: float = 0.0  # remaining stun duration; >0 = stunned

    # Slow
    slow_timer: float = 0.0
    slow_factor: float = 0.5

    # Death spawns
    death_spawn_card_type: CardType | None = None
    death_spawn_count: int = 0
    death_spawn_hp: float = 0.0
    death_spawn_dps: float = 0.0

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def stunned(self) -> bool:
        return self.stun_timer > 0.0

    @property
    def active(self) -> bool:
        return self.alive and self.is_deployed and not self.stunned

    @property
    def damage_per_hit(self) -> float:
        if self.attack_interval <= 0:
            return 0.0
        dmg = self.dps * self.attack_interval
        if self.next_hit_is_charge:
            dmg *= self.charge_damage_mult
        return dmg

    @property
    def effective_speed(self) -> float:
        if self.is_charging:
            return self.charge_speed
        s = self.speed
        if self.slow_timer > 0:
            s *= self.slow_factor
        return s

    def distance_to(self, other: Entity) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_to_pos(self, px: float, py: float) -> float:
        dx = self.x - px
        dy = self.y - py
        return math.sqrt(dx * dx + dy * dy)

    def apply_stun(self, duration: float) -> None:
        self.stun_timer = max(self.stun_timer, duration)

    def apply_slow(self, duration: float, factor: float = 0.5) -> None:
        self.slow_timer = max(self.slow_timer, duration)
        self.slow_factor = factor

    def take_damage(self, damage: float) -> float:
        """Apply damage, considering shield. Returns actual HP damage dealt."""
        if self.has_shield and self.shield_hp > 0:
            self.shield_hp -= damage
            if self.shield_hp <= 0:
                overflow = -self.shield_hp
                self.shield_hp = 0.0
                self.has_shield = False
                if overflow > 0:
                    self.hp -= overflow
                    return overflow
                return 0.0
            return 0.0
        self.hp -= damage
        return damage


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
    deploy_time = card_def.deploy_time if card_def.kind == EntityKind.TROOP else 0.0
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
        attack_timer=card_def.attack_interval,
        attack_range=card_def.attack_range,
        target_mode=card_def.target_mode,
        speed=card_def.speed,
        base_speed=card_def.speed,
        is_flying=card_def.is_flying,
        is_splash=card_def.is_splash,
        splash_radius=card_def.splash_radius,
        is_building=card_def.kind == EntityKind.BUILDING,
        building_timer=card_def.building_lifetime,
        spawner_interval=card_def.spawner_interval,
        spawner_timer=card_def.spawner_interval,
        inferno_dps_min=card_def.inferno_dps_min,
        inferno_dps_max=card_def.inferno_dps_max,
        deploy_timer=deploy_time,
        is_deployed=deploy_time <= 0,
        charge_speed=card_def.charge_speed,
        charge_damage_mult=card_def.charge_damage_mult,
        has_shield=card_def.has_shield,
        shield_hp=card_def.shield_hp,
        death_spawn_card_type=card_def.death_spawn_type,
        death_spawn_count=card_def.death_spawn_count,
        death_spawn_hp=card_def.death_spawn_hp,
        death_spawn_dps=card_def.death_spawn_dps,
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
