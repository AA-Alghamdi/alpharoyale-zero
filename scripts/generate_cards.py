#!/usr/bin/env python3
"""Generate expanded card definitions from ClashStrategic data.

Fetches the latest card data and generates:
  1. crsim/cards_v2.py — Full 121-card CardType enum + CardDef definitions
  2. cr_engine/gamedata_v2/ — Updated CSV files for the Rust engine

Usage:
    python scripts/generate_cards.py
"""

import json
import os
import sys

# Speed mappings (tiles per second)
SPEED_MAP = {
    "slow": 1.0,
    "medium": 1.5,
    "fast": 2.5,
    "very-fast": 3.0,
    None: 0.0,  # buildings/spells
}

# Target mode mappings
TARGET_MAP = {
    frozenset(["ground"]): "GROUND",
    frozenset(["ground", "air"]): "AIR_GROUND",
    frozenset(["air", "ground"]): "AIR_GROUND",
    frozenset(["buildings"]): "BUILDINGS",
    frozenset(["air"]): "AIR_GROUND",  # rare, treat as air_ground
    frozenset(): "AIR_GROUND",  # default
}


def load_cards(path="/tmp/clash_strategic_cards.json"):
    with open(path) as f:
        data = json.load(f)
    return data.get("cards", data) if isinstance(data, dict) else data


def card_enum_name(name: str) -> str:
    """Convert display name to Python enum name."""
    n = name.upper()
    n = n.replace(" ", "_").replace(".", "").replace("-", "_").replace("'", "")
    n = n.replace("PEKKA", "PEKKA")
    return n


def get_target_mode(targets):
    if not targets:
        return "AIR_GROUND"
    ts = frozenset(t.lower() for t in targets)
    return TARGET_MAP.get(ts, "AIR_GROUND")


def get_damage_l11(card):
    dmg = card.get("damage", {})
    if isinstance(dmg, dict):
        return dmg.get("level11", 0) or 0
    return 0


def get_hp_l11(card):
    hp = card.get("hitpoints", {})
    if isinstance(hp, dict):
        return hp.get("level11", 0) or 0
    return 0


def get_tower_damage_l11(card):
    td = card.get("towerDamage", {})
    if isinstance(td, dict):
        return td.get("level11", 0) or 0
    return 0


def compute_dps(card):
    dmg = get_damage_l11(card)
    hs = card.get("hitspeed", 0)
    if hs and hs > 0:
        return round(dmg / hs, 1)
    return 0.0


def generate_cards_v2(cards):
    """Generate the crsim/cards_v2.py file content."""
    lines = []
    lines.append('"""Card definitions for all 121 Clash Royale cards (auto-generated).')
    lines.append("")
    lines.append("Generated from ClashStrategic/stats data.")
    lines.append("Tournament standard (Level 11) stats.")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import enum")
    lines.append("from dataclasses import dataclass")
    lines.append("")
    lines.append("")
    lines.append("class CardType(enum.IntEnum):")

    # Generate enum
    for i, c in enumerate(cards):
        name = card_enum_name(c["name"])
        lines.append(f"    {name} = {i}")

    lines.append("")
    lines.append("")
    lines.append("class EntityKind(enum.IntEnum):")
    lines.append("    TROOP = 0")
    lines.append("    SPELL = 1")
    lines.append("    BUILDING = 2")
    lines.append("")
    lines.append("")
    lines.append("class TargetMode(enum.IntEnum):")
    lines.append("    GROUND = 0")
    lines.append("    AIR_GROUND = 1")
    lines.append("    BUILDINGS = 2")
    lines.append("    AREA = 3")
    lines.append("")
    lines.append("")
    lines.append("@dataclass(frozen=True, slots=True)")
    lines.append("class CardDef:")
    lines.append('    """Immutable definition of a card\'s base stats at Level 11."""')
    lines.append("    card_type: CardType")
    lines.append("    kind: EntityKind")
    lines.append("    cost: int")
    lines.append("    hp: float")
    lines.append("    dps: float")
    lines.append("    damage_per_hit: float")
    lines.append("    hit_speed: float  # seconds between attacks")
    lines.append("    load_time: float  # first attack pre-load")
    lines.append("    speed: float  # tiles/sec")
    lines.append("    attack_range: float  # tiles")
    lines.append("    sight_range: float  # tiles")
    lines.append("    target_mode: TargetMode")
    lines.append("    is_flying: bool = False")
    lines.append("    is_splash: bool = False")
    lines.append("    splash_radius: float = 0.0")
    lines.append("    spawn_count: int = 1")
    lines.append("    deploy_time: float = 1.0")
    lines.append("    collision_radius: float = 0.5")
    lines.append("    has_projectile: bool = False")
    lines.append("    tower_damage: float = 0.0  # reduced damage to towers (spells)")
    lines.append("    building_lifetime: float = 0.0")
    lines.append("    has_evolution: bool = False")
    lines.append("    is_champion: bool = False")
    lines.append("    # Death spawn")
    lines.append("    death_spawn_card: CardType | None = None")
    lines.append("    death_spawn_count: int = 0")
    lines.append("")
    lines.append("")

    # Generate CARD_DEFS dictionary
    lines.append("CARD_DEFS: dict[CardType, CardDef] = {")

    for c in cards:
        enum_name = card_enum_name(c["name"])
        card_type = c.get("type", "troop")
        kind = {"troop": "EntityKind.TROOP", "spell": "EntityKind.SPELL", "building": "EntityKind.BUILDING"}[card_type]

        cost = c.get("elixirCost", 0) or 0
        hp = get_hp_l11(c)
        dmg = get_damage_l11(c)
        dps = compute_dps(c)
        hit_speed = c.get("hitspeed", 0) or 0
        load_time = c.get("loadTime", 0) or 0
        speed_str = c.get("speed")
        speed = SPEED_MAP.get(speed_str, 0.0)
        attack_range = c.get("range", 0) or 0
        if isinstance(attack_range, str):
            try:
                attack_range = float(attack_range)
            except ValueError:
                attack_range = 0
        sight_range = c.get("sightRange", 5.5) or 5.5
        target_mode = get_target_mode(c.get("targets", []))
        is_flying = bool(c.get("flying", False))
        has_projectile = bool(c.get("projectile", False))
        spawn_count = c.get("units", 1) or 1
        deploy_time = c.get("deployTime", 1.0) or 1.0
        collision_radius = c.get("collisionRadius", 0.5) or 0.5
        tower_damage = get_tower_damage_l11(c)
        has_evolution = bool(c.get("evolution", False))
        is_champion = bool(c.get("hero", False))
        duration = c.get("duration")
        building_lifetime = float(duration) if duration and card_type == "building" else 0.0

        # Determine splash
        radius = c.get("radius")
        is_splash = radius is not None and radius and float(radius) > 0
        splash_radius = float(radius) if is_splash else 0.0

        # Death spawns for known cards
        death_spawn_card = "None"
        death_spawn_count = 0
        if enum_name == "GOLEM":
            death_spawn_card = "CardType.GOLEM"  # golemites
            death_spawn_count = 2
        elif enum_name == "LAVA_HOUND":
            death_spawn_card = "CardType.LAVA_HOUND"  # lava pups
            death_spawn_count = 6

        lines.append(f"    CardType.{enum_name}: CardDef(")
        lines.append(f"        card_type=CardType.{enum_name},")
        lines.append(f"        kind={kind},")
        lines.append(f"        cost={cost},")
        lines.append(f"        hp={float(hp)},")
        lines.append(f"        dps={dps},")
        lines.append(f"        damage_per_hit={float(dmg)},")
        lines.append(f"        hit_speed={float(hit_speed)},")
        lines.append(f"        load_time={float(load_time)},")
        lines.append(f"        speed={speed},")
        lines.append(f"        attack_range={float(attack_range)},")
        lines.append(f"        sight_range={float(sight_range)},")
        lines.append(f"        target_mode=TargetMode.{target_mode},")
        if is_flying:
            lines.append(f"        is_flying=True,")
        if is_splash:
            lines.append(f"        is_splash=True,")
            lines.append(f"        splash_radius={splash_radius},")
        if spawn_count != 1:
            lines.append(f"        spawn_count={spawn_count},")
        if deploy_time != 1.0:
            lines.append(f"        deploy_time={deploy_time},")
        if collision_radius != 0.5:
            lines.append(f"        collision_radius={collision_radius},")
        if has_projectile:
            lines.append(f"        has_projectile=True,")
        if tower_damage and tower_damage != dmg:
            lines.append(f"        tower_damage={float(tower_damage)},")
        if building_lifetime > 0:
            lines.append(f"        building_lifetime={building_lifetime},")
        if has_evolution:
            lines.append(f"        has_evolution=True,")
        if is_champion:
            lines.append(f"        is_champion=True,")
        if death_spawn_count > 0:
            lines.append(f"        death_spawn_card={death_spawn_card},")
            lines.append(f"        death_spawn_count={death_spawn_count},")
        lines.append(f"    ),")

    lines.append("}")
    lines.append("")

    # Add convenience constants
    lines.append("")
    lines.append(f"NUM_CARD_TYPES: int = {len(cards)}")
    lines.append("")

    return "\n".join(lines)


def generate_rust_csv(cards, output_dir):
    """Generate CSV files for the Rust engine."""
    os.makedirs(output_dir, exist_ok=True)

    # characters.csv - troops
    with open(os.path.join(output_dir, "characters.csv"), "w") as f:
        f.write("Name,Hitpoints,Damage,HitSpeed,LoadTime,Speed,Range,SightRange,DeployTime,CollisionRadius,Projectile,IsFlying,IsSplash,SplashRadius,TargetMode,SpawnCount\n")
        for c in cards:
            if c.get("type") != "troop":
                continue
            name = c["name"].replace(" ", "").replace(".", "").replace("-", "").replace("'", "")
            hp = get_hp_l11(c)
            dmg = get_damage_l11(c)
            hs = c.get("hitspeed", 0) or 0
            lt = c.get("loadTime", 0) or 0
            spd = c.get("speed", "medium") or "medium"
            rng = c.get("range", 0) or 0
            sr = c.get("sightRange", 5.5) or 5.5
            dt = c.get("deployTime", 1.0) or 1.0
            cr_val = c.get("collisionRadius", 0.5) or 0.5
            proj = "true" if c.get("projectile") else "false"
            fly = "true" if c.get("flying") else "false"
            radius = c.get("radius")
            is_splash = "true" if radius and float(radius) > 0 else "false"
            splash_r = float(radius) if radius and float(radius) > 0 else 0
            targets = c.get("targets", [])
            if targets and "buildings" in [t.lower() for t in targets]:
                tm = "Buildings"
            elif targets and set(t.lower() for t in targets) == {"ground"}:
                tm = "Ground"
            else:
                tm = "AirGround"
            units = c.get("units", 1) or 1
            f.write(f"{name},{hp},{dmg},{hs},{lt},{spd},{rng},{sr},{dt},{cr_val},{proj},{fly},{is_splash},{splash_r},{tm},{units}\n")

    # spells.csv - spells
    with open(os.path.join(output_dir, "spells.csv"), "w") as f:
        f.write("Name,ElixirCost,Damage,TowerDamage,Radius,Duration,Type\n")
        for c in cards:
            if c.get("type") != "spell":
                continue
            name = c["name"].replace(" ", "").replace(".", "").replace("-", "").replace("'", "")
            cost = c.get("elixirCost", 0) or 0
            dmg = get_damage_l11(c)
            td = get_tower_damage_l11(c)
            radius = c.get("radius", 0) or 0
            duration = c.get("duration", 0) or 0
            f.write(f"{name},{cost},{dmg},{td},{radius},{duration},Spell\n")

    # buildings.csv
    with open(os.path.join(output_dir, "buildings.csv"), "w") as f:
        f.write("Name,ElixirCost,Hitpoints,Damage,HitSpeed,Range,Lifetime,DeployTime,IsProjectile\n")
        for c in cards:
            if c.get("type") != "building":
                continue
            name = c["name"].replace(" ", "").replace(".", "").replace("-", "").replace("'", "")
            cost = c.get("elixirCost", 0) or 0
            hp = get_hp_l11(c)
            dmg = get_damage_l11(c)
            hs = c.get("hitspeed", 0) or 0
            rng = c.get("range", 0) or 0
            lifetime = c.get("duration", 0) or 0
            dt = c.get("deployTime", 1.0) or 1.0
            proj = "true" if c.get("projectile") else "false"
            f.write(f"{name},{cost},{hp},{dmg},{hs},{rng},{lifetime},{dt},{proj}\n")


if __name__ == "__main__":
    cards = load_cards()
    print(f"Loaded {len(cards)} cards from ClashStrategic")

    # Generate Python card definitions
    py_content = generate_cards_v2(cards)
    py_path = os.path.join(os.path.dirname(__file__), "..", "crsim", "cards_v2.py")
    with open(py_path, "w") as f:
        f.write(py_content)
    print(f"Generated {py_path}")

    # Generate Rust CSV files
    csv_dir = os.path.join(os.path.dirname(__file__), "..", "cr_engine", "gamedata_v2")
    generate_rust_csv(cards, csv_dir)
    print(f"Generated CSV files in {csv_dir}")

    # Summary
    troops = [c for c in cards if c.get("type") == "troop"]
    spells = [c for c in cards if c.get("type") == "spell"]
    buildings = [c for c in cards if c.get("type") == "building"]
    evos = [c for c in cards if c.get("evolution")]
    champs = [c for c in cards if c.get("hero")]
    print(f"\nSummary: {len(troops)} troops, {len(spells)} spells, {len(buildings)} buildings")
    print(f"Evolutions: {len(evos)}, Champions: {len(champs)}")
