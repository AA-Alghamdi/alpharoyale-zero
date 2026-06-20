#!/usr/bin/env python3
"""Generate gamedata_v2/ CSVs in SC format from ClashStrategic + existing data.

Merges the existing gamedata/ (v1.9.2 APK, 52 characters) with the latest
ClashStrategic card stats to produce a complete 121-card dataset.

For cards that exist in both sources, the SC APK data is preferred (more detailed).
For new cards (post-v1.9.2), we generate entries from ClashStrategic data.
"""

import csv
import json
import os
import shutil

SPEED_TO_INT = {
    "slow": 45,
    "medium": 60,
    "fast": 100,
    "very-fast": 120,
}


def load_cs_cards():
    with open("/tmp/clash_strategic_cards.json") as f:
        data = json.load(f)
    cards = data.get("cards", data) if isinstance(data, dict) else data
    return {c["name"]: c for c in cards}


def load_existing_characters(path):
    """Load existing characters.csv entries."""
    entries = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        type_row = next(reader)
        for row in reader:
            if row[0] and not row[0].startswith("NOTINUSE"):
                entries[row[0]] = row
    return header, type_row, entries


def load_existing_spells_characters(path):
    entries = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        type_row = next(reader)
        for row in reader:
            if row[0] and not row[0].startswith("NOTINUSE"):
                entries[row[0]] = row
    return header, type_row, entries


def cs_name_to_char_name(name):
    """Convert ClashStrategic name to character name."""
    mappings = {
        "Archers": "Archer",
        "Goblins": "Goblin",
        "Minions": "Minion",
        "Barbarians": "Barbarian",
        "Skeletons": "Skeleton",
        "Skeleton Army": "Skeleton",
        "Spear Goblins": "SpearGoblin",
        "Minion Horde": "Minion",
        "Guards": "Guard",
        "Three Musketeers": "Musketeer",
        "Bats": "Bat",
        "Goblin Gang": "Goblin",
        "Elite Barbarians": "EliteBarbarian",
        "Rascals": "Rascal",
        "Wall Breakers": "WallBreaker",
        "Royal Hogs": "RoyalHog",
        "Skeleton Dragons": "SkeletonDragon",
        "Royal Recruits": "RoyalRecruit",
    }
    if name in mappings:
        return mappings[name]
    return name.replace(" ", "").replace(".", "").replace("-", "").replace("'", "")


def cs_to_char_row(cs_card, header):
    """Create a characters.csv row from ClashStrategic data."""
    name = cs_name_to_char_name(cs_card["name"])
    hp_data = cs_card.get("hitpoints", {})
    dmg_data = cs_card.get("damage", {})
    hp = hp_data.get("level11", 0) if isinstance(hp_data, dict) else 0
    dmg = dmg_data.get("level11", 0) if isinstance(dmg_data, dict) else 0

    speed_str = cs_card.get("speed", "medium") or "medium"
    speed = SPEED_TO_INT.get(speed_str, 60)

    hit_speed = cs_card.get("hitspeed", 1.0) or 1.0
    hit_speed_ms = int(float(hit_speed) * 1000)

    load_time = cs_card.get("loadTime", 0) or 0
    load_time_ms = int(float(load_time) * 1000)

    attack_range = cs_card.get("range", 1.0) or 1.0
    range_mt = int(float(attack_range) * 1000)

    sight_range = cs_card.get("sightRange", 5.5) or 5.5
    sight_range_mt = int(float(sight_range) * 1000)

    deploy_time = cs_card.get("deployTime", 1.0) or 1.0
    deploy_time_ms = int(float(deploy_time) * 1000)

    collision_radius = cs_card.get("collisionRadius", 0.5) or 0.5
    collision_radius_mt = int(float(collision_radius) * 1000)

    targets = cs_card.get("targets", []) or []
    attacks_ground = any(t.lower() in ("ground", "buildings") for t in targets)
    attacks_air = any(t.lower() in ("air", "ground") for t in targets) if len(targets) > 1 else False
    if not targets:
        attacks_ground = True
        attacks_air = True
    target_only_buildings = any(t.lower() == "buildings" for t in targets) and len(targets) == 1

    is_flying = cs_card.get("flying", False)
    flying_height = 1 if is_flying else 0
    has_projectile = cs_card.get("projectile", False)

    # Build a row matching the header length
    row = [""] * len(header)
    # Set known fields by name
    field_map = {h: i for i, h in enumerate(header)}

    def set_field(field_name, value):
        if field_name in field_map:
            row[field_map[field_name]] = str(value)

    set_field("Name", name)
    set_field("Rarity", cs_card.get("rarity", "Common"))
    set_field("SightRange", sight_range_mt)
    set_field("DeployTime", deploy_time_ms)
    set_field("Speed", speed)
    set_field("Hitpoints", hp or 100)
    set_field("HitSpeed", hit_speed_ms)
    set_field("LoadTime", load_time_ms)
    set_field("Damage", dmg or 50)
    set_field("Range", range_mt)
    set_field("AttacksGround", "TRUE" if attacks_ground else "FALSE")
    set_field("AttacksAir", "TRUE" if attacks_air else "FALSE")
    set_field("TargetOnlyBuildings", "TRUE" if target_only_buildings else "FALSE")
    set_field("CollisionRadius", collision_radius_mt)
    set_field("Mass", 500)
    set_field("FlyingHeight", flying_height)
    set_field("Projectile", "Projectile" if has_projectile else "")
    set_field("ChargeRange", 0)
    set_field("DeathDamageRadius", 0)
    set_field("DeathDamage", 0)
    set_field("DeathPushBack", 0)
    set_field("AttackPushBack", 0)
    set_field("LifeTime", 0)
    set_field("AreaDamageRadius", 0)
    set_field("CrownTowerDamagePercent", 0)
    set_field("MultipleProjectiles", 0)
    set_field("SpecialAttackInterval", 0)

    # Scale HP to base level (the CSV uses base level stats, not level 11)
    # Level 11 = base * (1 + 0.1)^10 ≈ base * 2.594
    # So base ≈ level11 / 2.594 ≈ level11 * 0.3855
    # But we want to keep level 11 stats for tournament play
    # The engine uses the CSV values directly, so just use level 11 stats

    return row


def main():
    cs_cards = load_cs_cards()
    src_dir = "cr_engine/gamedata"
    dst_dir = "cr_engine/gamedata_v2"

    # Start by copying existing data
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    # Load existing characters
    char_path = os.path.join(dst_dir, "characters.csv")
    header, type_row, existing_chars = load_existing_characters(char_path)

    # Find cards that need new character entries
    new_troop_cards = []
    for cs_name, cs_card in cs_cards.items():
        if cs_card.get("type") != "troop":
            continue
        char_name = cs_name_to_char_name(cs_name)
        if char_name not in existing_chars:
            new_troop_cards.append(cs_card)

    print(f"Existing characters: {len(existing_chars)}")
    print(f"New troop cards to add: {len(new_troop_cards)}")
    if new_troop_cards:
        print(f"  New troops: {[c['name'] for c in new_troop_cards]}")

    # Write updated characters.csv
    with open(char_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(type_row)
        for row in existing_chars.values():
            writer.writerow(row)
        for cs_card in new_troop_cards:
            row = cs_to_char_row(cs_card, header)
            writer.writerow(row)

    # Similarly update spells_characters.csv for new cards
    spell_char_path = os.path.join(dst_dir, "spells_characters.csv")
    sc_header, sc_type_row, existing_sc = load_existing_spells_characters(spell_char_path)

    new_spell_chars = []
    for cs_name, cs_card in cs_cards.items():
        if cs_card.get("type") != "troop":
            continue
        # Spell character name is the display name without spaces
        sc_name = cs_name.replace(" ", "").replace(".", "").replace("-", "").replace("'", "")
        # Also check plural forms
        if sc_name not in existing_sc and cs_name not in existing_sc:
            # Need to add spell character entry
            units = cs_card.get("units", 1) or 1
            cost = cs_card.get("elixirCost", 3) or 3
            char_name = cs_name_to_char_name(cs_name)

            row = [""] * len(sc_header)
            sc_field_map = {h: i for i, h in enumerate(sc_header)}

            def set_sc(field, val):
                if field in sc_field_map:
                    row[sc_field_map[field]] = str(val)

            set_sc("Name", sc_name)
            set_sc("Rarity", cs_card.get("rarity", "Common"))
            set_sc("ManaCost", cost)
            set_sc("SummonCharacter", char_name)
            set_sc("SummonNumber", units)
            set_sc("SummonCharacterSecond", "")
            set_sc("SummonCharacterSecondCount", 0)
            set_sc("Radius", 0)
            set_sc("CanPlaceOnBuildings", "FALSE")

            new_spell_chars.append(row)

    print(f"\nExisting spell characters: {len(existing_sc)}")
    print(f"New spell characters to add: {len(new_spell_chars)}")

    with open(spell_char_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(sc_header)
        writer.writerow(sc_type_row)
        for row in existing_sc.values():
            writer.writerow(row)
        for row in new_spell_chars:
            writer.writerow(row)

    # Update spells_other.csv for new spell cards
    spell_other_path = os.path.join(dst_dir, "spells_other.csv")
    so_header, so_type_row, existing_so = load_existing_spells_characters(spell_other_path)

    new_spells = []
    for cs_name, cs_card in cs_cards.items():
        if cs_card.get("type") != "spell":
            continue
        spell_name = cs_name.replace(" ", "").replace(".", "").replace("-", "").replace("'", "")
        if spell_name not in existing_so and cs_name not in existing_so:
            row = [""] * len(so_header)
            so_field_map = {h: i for i, h in enumerate(so_header)}

            def set_so(field, val):
                if field in so_field_map:
                    row[so_field_map[field]] = str(val)

            cost = cs_card.get("elixirCost", 3) or 3
            dmg_data = cs_card.get("damage", {})
            dmg = dmg_data.get("level11", 0) if isinstance(dmg_data, dict) else 0
            radius = cs_card.get("radius", 0) or 0
            radius_mt = int(float(radius) * 1000) if radius else 0
            duration = cs_card.get("duration", 0) or 0

            set_so("Name", spell_name)
            set_so("ManaCost", cost)
            set_so("Radius", radius_mt)
            set_so("InstantDamage", dmg or 0)
            set_so("DurationSeconds", duration)
            set_so("Pushback", 0)
            set_so("Effect", "")

            new_spells.append(row)

    print(f"\nExisting spells: {len(existing_so)}")
    print(f"New spells to add: {len(new_spells)}")

    with open(spell_other_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(so_header)
        writer.writerow(so_type_row)
        for row in existing_so.values():
            writer.writerow(row)
        for row in new_spells:
            writer.writerow(row)

    # Count totals
    final_chars = len(existing_chars) + len(new_troop_cards)
    final_sc = len(existing_sc) + len(new_spell_chars)
    final_so = len(existing_so) + len(new_spells)
    print("\n=== Final Totals ===")
    print(f"Characters: {final_chars}")
    print(f"Spell characters: {final_sc}")
    print(f"Spell others: {final_so}")
    print(f"Total playable cards: {final_sc + final_so}")


if __name__ == "__main__":
    main()
