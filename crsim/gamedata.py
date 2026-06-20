"""Authentic card stats from the decoded Supercell game data.

This is the *single source of truth* for card numbers. The JSON files in
``cr_engine/gamedata_full/`` are the decoded Clash Royale ``csv_logic`` tables
(via the RoyaleAPI/cr-api-data project — see ``gamedata_full/SOURCE.txt``), so
sourcing the simulator from them — rather than from hand-typed approximations —
is what makes the engine match the real game.

Two things make this authoritative:

  * **Exact level scaling.** ``rarities.json`` carries the real per-level
    ``power_level_multiplier`` table. Stats in the stat files are card *Level 1*
    base values; the tournament-standard (King Level 11) value is
    ``base * power_level_multiplier[9 - relative_level] / 100``. This reproduces
    the wiki values exactly (Knight 1766 HP / 202 dmg, P.E.K.K.A 3760 / 816,
    Mega Knight 3993, Champions at base).
  * **Every mechanic field.** Each character/building row carries ~300 fields
    (charge, dash, death damage/spawn, spawners, minimum range, crown-tower %,
    shields, morphs, buffs, multi-projectile, …) — the data needed to implement
    mechanics authentically rather than guessing.

``apply_authentic_stats`` overlays these numbers onto the hand-authored
``CARD_DEFS`` table, replacing the *core* stat fields while preserving the
hand-curated special-mechanic flags. The few newest cards absent from this
snapshot keep their existing values.

Unit conventions match the Python sim: millitiles ``/1000`` -> tiles,
milliseconds ``/1000`` -> seconds, speed code ``/30`` -> tiles/sec (Knight
60->2.0, Goblin 120->4.0).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

TOURNAMENT_LEVEL = 11
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "cr_engine" / "gamedata_full"


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


class GameData:
    """Decoded game data, with exact tournament-standard level scaling."""

    def __init__(self, data_dir: Path | None = None, level: int = TOURNAMENT_LEVEL) -> None:
        d = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.level = level
        cards = _load_json(d / "cards.json") or []
        self.characters = {
            c["name"]: c for c in (_load_json(d / "cards_stats_characters.json") or [])
        }
        self.buildings = {
            c["name"]: c for c in (_load_json(d / "cards_stats_building.json") or [])
        }
        self.spells = {c["name"]: c for c in (_load_json(d / "cards_stats_spell.json") or [])}
        self.projectiles = {
            c["name"]: c for c in (_load_json(d / "cards_stats_projectile.json") or [])
        }
        self.rarities = {r["name"]: r for r in (_load_json(d / "rarities.json") or [])}

        # Index playable cards by both display name and sc_key (normalised).
        self.cards_by_norm: dict[str, dict] = {}
        for c in cards:
            self.cards_by_norm.setdefault(_norm(c.get("name")), c)
            self.cards_by_norm.setdefault(_norm(c.get("sc_key")), c)
        # Normalised stat indexes. Multi-unit troops carry a *plural* sc_key
        # ("Goblins") while the stat row is keyed by the *singular* entity name
        # ("Goblin"), so we also need a singular-fallback lookup.
        self.characters_by_norm = {_norm(k): v for k, v in self.characters.items()}
        self.buildings_by_norm = {_norm(k): v for k, v in self.buildings.items()}
        self.available = bool(cards and self.characters and self.rarities)

    @staticmethod
    def _resolve(index_by_norm: dict, key: str | None) -> dict | None:
        if not key:
            return None
        n = _norm(key)
        hit = index_by_norm.get(n)
        if hit is not None:
            return hit
        # Singular fallback: "GOBLINS" -> "GOBLIN", "BATS" -> "BAT".
        if n.endswith("S"):
            return index_by_norm.get(n[:-1])
        return None

    def character_stats(self, key: str | None) -> dict | None:
        return self._resolve(self.characters_by_norm, key)

    def building_stats(self, key: str | None) -> dict | None:
        return self._resolve(self.buildings_by_norm, key)

    # ---- scaling ----
    def level_multiplier(self, rarity: str) -> float:
        r = self.rarities.get(rarity)
        if not r:
            return 1.0
        idx = 9 - r.get("relative_level", 0)  # index for King Level 11
        table = r.get("power_level_multiplier", [])
        if idx < 0 or idx >= len(table):
            return 1.0
        return table[idx] / 100.0

    def scale(self, base_value, rarity: str) -> float:
        if not base_value:
            return 0.0
        return base_value * self.level_multiplier(rarity)

    def lookup(self, card_type) -> dict | None:
        return self.cards_by_norm.get(_norm(card_type.name))


# Module-level scaling helper (kept for tests / external callers).
_SHARED: GameData | None = None


def _shared() -> GameData:
    global _SHARED
    if _SHARED is None:
        _SHARED = GameData()
    return _SHARED


def scale_stat(base_value, rarity: str) -> float:
    """Scale a Level-1 base stat to tournament standard (L11) by rarity."""
    return _shared().scale(base_value, rarity)


def _proj_damage_and_radius(game: GameData, stats: dict) -> tuple[int, float]:
    """Damage + splash radius for a ranged unit, pulled from its projectile."""
    proj_name = stats.get("projectile")
    if proj_name and proj_name in game.projectiles:
        p = game.projectiles[proj_name]
        return int(p.get("damage") or 0), (p.get("radius") or 0) / 1000.0
    return 0, 0.0


# A few compound cards have a cards.json sc_key that points at a rider/sub-unit
# rather than the HP-bearing main entity. Map them to the correct character row.
_CHAR_ALIAS: dict[str, str] = {
    "RAM_RIDER": "Ram",  # sc_key "RamRider" is the rider; "Ram" carries the HP
    # Multi-summon troops whose card has no direct stat row: map to the summoned
    # character so each unit gets authentic per-unit stats (spawn_count stays
    # hand-coded). Only homogeneous spawns are aliased; heterogeneous composites
    # (Goblin Gang, Rascals) keep their hand-curated single-entity approximation.
    "SKELETON_ARMY": "Skeleton",
    "MINION_HORDE": "Minion",
    "ROYAL_RECRUITS": "Recruit",
    "ELIXIR_GOLEM": "ElixirGolem1",
}


def _troop_or_building_fields(game: GameData, card: dict, stats: dict, is_building: bool) -> dict:
    from crsim.cards import EntityKind

    # Scale by the *entity's* own rarity, not the card's: a multi-summon card
    # (Skeleton Army is Epic) spawns base-rarity units (Skeleton is Common), and
    # the unit's stats follow the unit, not the card tier.
    rarity = stats.get("rarity") or card.get("rarity") or "Common"
    hp = game.scale(stats.get("hitpoints") or 0, rarity)
    dmg = int(stats.get("damage") or 0)
    splash = (stats.get("area_damage_radius") or 0) / 1000.0
    if dmg == 0:
        pdmg, pradius = _proj_damage_and_radius(game, stats)
        dmg = dmg or pdmg
        if splash == 0.0:
            splash = pradius
    dmg = game.scale(dmg, rarity)
    hit_speed = max((stats.get("hit_speed") or 0) / 1000.0, 0.1)
    attacks_air = bool(stats.get("attacks_air"))
    only_bldg = bool(stats.get("target_only_buildings"))
    flying = (stats.get("flying_height") or 0) > 0

    fields: dict = {
        "kind": EntityKind.BUILDING if is_building else EntityKind.TROOP,
        "cost": card.get("elixir") or None,
        "hp": hp,
        "damage_per_hit": dmg,
        "dps": (dmg / hit_speed) if hit_speed > 0 else 0.0,
        "hit_speed": hit_speed,
        "load_time": (stats.get("load_time") or 0) / 1000.0,
        "speed": (stats.get("speed") or 0) / 30.0,
        "attack_range": (stats.get("range") or 0) / 1000.0,
        "sight_range": (stats.get("sight_range") or 0) / 1000.0,
        "deploy_time": max((stats.get("deploy_time") or 0) / 1000.0, 0.0),
        "is_flying": flying,
        "splash_radius": splash,
        "collision_radius": (stats.get("collision_radius") or 0) / 1000.0 or None,
    }
    # Only upgrade is_splash to True from the data; never downgrade a
    # hand-curated splash flag (some cards splash via special mechanics the
    # standard columns don't encode).
    if splash > 0.0:
        fields["is_splash"] = True
    # Mechanic fields: only emit when present so the overlay never zeroes out a
    # hand-curated value the standard columns don't encode (e.g. Balloon's
    # death bomb, modelled outside death_damage).
    min_range = (stats.get("minimum_range") or 0) / 1000.0
    if min_range > 0:
        fields["minimum_range"] = min_range
    death_dmg = game.scale(stats.get("death_damage") or 0, rarity)
    if death_dmg > 0:
        fields["death_damage"] = death_dmg
        fields["death_damage_radius"] = (stats.get("death_damage_radius") or 0) / 1000.0 or 2.0
    # Targeting: let the data drive air/ground, but never flip a hand-curated
    # target mode to BUILDINGS. Compound cards (Ram Rider, Goblin Giant) ride a
    # building-targeting main entity yet their rider attacks air+ground, so the
    # single-entity sim deliberately models them as AIR_GROUND. Building-only
    # win-cons (Giant, Hog, Golem, ...) are already hand-coded as BUILDINGS.
    if not only_bldg:
        fields["_air"] = attacks_air
    if is_building:
        fields["building_lifetime"] = (stats.get("life_time") or 0) / 1000.0
    return fields


def _spell_fields(game: GameData, card: dict, stats: dict) -> dict:
    rarity = card.get("rarity") or "Common"
    radius = (stats.get("radius") or 0) / 1000.0
    dmg = int(stats.get("damage") or 0)
    fields: dict = {"cost": card.get("elixir") or None}
    if radius:
        fields["splash_radius"] = radius
    if dmg:
        fields["damage_per_hit"] = game.scale(dmg, rarity)
    # Number of summoned units (Goblin Barrel, Skeleton Army, ...).
    n = stats.get("spawn_max_count") or stats.get("summon_number")
    if n:
        fields["spawn_count"] = int(n)
    return fields


def authentic_core_fields(game: GameData, card_type) -> dict | None:
    """Authentic core stat fields for a card, or ``None`` if not in the data."""
    card = game.lookup(card_type)
    if not card:
        return None
    sc = card.get("sc_key") or card.get("name")
    ctype = (card.get("type") or "").lower()
    if ctype == "spell":
        stats = game.spells.get(sc)
        # Some "spells" (Goblin Barrel) summon troops; pull spawn stats too.
        return _spell_fields(game, card, stats) if stats else None
    if ctype == "building":
        stats = game.building_stats(sc)
        if not stats or not stats.get("hitpoints"):
            return None
        return _troop_or_building_fields(game, card, stats, is_building=True)
    sc = _CHAR_ALIAS.get(card_type.name, sc)
    stats = game.character_stats(sc)
    if not stats or not stats.get("hitpoints"):
        return None
    return _troop_or_building_fields(game, card, stats, is_building=False)


def crown_tower_damage_percent(game: GameData, card_type) -> float:
    """Authentic ``crown_tower_damage_percent`` for a card (0 if none).

    The reduction lives in different places per card: on the character row
    (Miner), the spell row (Zap/Freeze), or — for the damage spells whose
    numbers sit on their projectile (Fireball/Arrows/Rocket/Snowball/Log) — on
    the projectile, named ``<sc_key>Spell`` (or linked via the spell's
    ``projectile`` field, e.g. Lightning -> ``LighningSpell``).
    """
    card = game.lookup(card_type)
    if not card:
        return 0.0
    sc = card.get("sc_key") or card.get("name") or ""
    n = _norm(sc)

    for src in (game.character_stats(sc), game.building_stats(sc), game.spells.get(sc)):
        if src:
            pct = src.get("crown_tower_damage_percent")
            if pct:
                return float(pct)

    # Damage spells: number is on the projectile. Follow an explicit link first,
    # then match by ``<sc_key>Spell`` / prefix.
    spell = game.spells.get(sc)
    proj_link = spell.get("projectile") if spell else None
    if proj_link and proj_link in game.projectiles:
        pct = game.projectiles[proj_link].get("crown_tower_damage_percent")
        if pct:
            return float(pct)
    for pname, p in game.projectiles.items():
        pn = _norm(pname)
        if (pn == n + "SPELL" or pn == n + "PROJECTILE" or pn.startswith(n)) and p.get(
            "crown_tower_damage_percent"
        ):
            return float(p["crown_tower_damage_percent"])
    return 0.0


def death_spawn_unit_stats(game: GameData, card_type) -> tuple[float, float] | None:
    """Authentic ``(hp, dps)`` of a card's death-spawned unit, or None.

    The parent's character/building row names the spawned unit in
    ``death_spawn_character`` (Golem -> ``Golemite``, Lava Hound -> ``LavaPups``);
    we then scale the *spawned* unit's own row to tournament level. Bomb-style
    death spawns (Giant Skeleton -> ``GiantSkeletonBomb``) have no hitpoints and
    are modelled as death damage instead, so they return None here.
    """
    card = game.lookup(card_type)
    if not card:
        return None
    sc = card.get("sc_key")
    parent = game.character_stats(sc) or game.building_stats(sc)
    if not parent:
        return None
    spawn_name = parent.get("death_spawn_character")
    if not spawn_name:
        return None
    unit = game.character_stats(spawn_name)
    if not unit or not unit.get("hitpoints"):
        return None
    rarity = unit.get("rarity") or "Common"
    hp = game.scale(unit.get("hitpoints") or 0, rarity)
    dmg = game.scale(unit.get("damage") or 0, rarity)
    hit_speed = (unit.get("hit_speed") or 0) / 1000.0
    dps = dmg / hit_speed if hit_speed > 0 else 0.0
    return hp, dps


def build_name_map(game: GameData, card_types) -> dict:
    """For diagnostics: map each CardType to its matched card dict."""
    return {ct: game.lookup(ct) for ct in card_types if game.lookup(ct)}


def apply_authentic_stats(card_defs: dict, *, data_dir: Path | None = None) -> tuple[dict, dict]:
    """Overlay authentic stats onto ``card_defs``.

    Returns ``(new_card_defs, report)``. Cards absent from the dataset are left
    untouched. Special-mechanic flags are never overwritten here.
    """
    from crsim.cards import TargetMode

    game = GameData(data_dir=data_dir) if data_dir else _shared()
    if not game.available:
        logger.warning("Authentic game data not found; keeping hand-coded stats.")
        return card_defs, {}

    new_defs = dict(card_defs)
    report: dict = {}

    for card_type, base in card_defs.items():
        core = authentic_core_fields(game, card_type) or {}

        # Crown-tower damage reduction is sourced independently: many damage
        # spells (Fireball/Arrows/Rocket) have no core stat row to overlay, but
        # still carry an authentic crown_tower_damage_percent on their projectile.
        ctd = crown_tower_damage_percent(game, card_type)
        if ctd:
            core["crown_tower_damage_percent"] = ctd

        # Death-spawned units (Golem -> Golemites, Lava Hound -> pups): scale the
        # *spawned* unit's own row. Only refine an existing hand-coded death
        # spawn, and never zero out a value the data omits (some spawned units
        # carry their damage on a projectile row, e.g. Lava Pups).
        if base.death_spawn_count > 0:
            ds = death_spawn_unit_stats(game, card_type)
            if ds:
                ds_hp, ds_dps = ds
                if ds_hp > 0:
                    core["death_spawn_hp"] = ds_hp
                if ds_dps > 0:
                    core["death_spawn_dps"] = ds_dps

        if not core:
            continue

        air = core.pop("_air", None)
        if air is not None and base.target_mode != TargetMode.BUILDINGS:
            core["target_mode"] = TargetMode.AIR_GROUND if air else TargetMode.GROUND

        # Keep multi-spawn troops' per-unit stat mirror consistent.
        if base.spawn_hp > 0 and "hp" in core:
            core["spawn_hp"] = core["hp"]
        if base.spawn_dps > 0 and "dps" in core:
            core["spawn_dps"] = core["dps"]

        core = {k: v for k, v in core.items() if v is not None and hasattr(base, k)}
        changed = {k: (getattr(base, k), v) for k, v in core.items() if getattr(base, k) != v}
        new_defs[card_type] = dataclasses.replace(base, **core)
        if changed:
            report[card_type] = changed

    logger.info("Applied authentic stats to %d/%d cards", len(report), len(card_defs))
    return new_defs, report
