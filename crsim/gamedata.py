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
        self.available = bool(cards and self.characters and self.rarities)

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
}


def _troop_or_building_fields(game: GameData, card: dict, stats: dict, is_building: bool) -> dict:
    from crsim.cards import EntityKind, TargetMode

    rarity = card.get("rarity") or stats.get("rarity") or "Common"
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
    if only_bldg:
        fields["target_mode"] = TargetMode.BUILDINGS
    else:
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
        stats = game.buildings.get(sc)
        if not stats or not stats.get("hitpoints"):
            return None
        return _troop_or_building_fields(game, card, stats, is_building=True)
    sc = _CHAR_ALIAS.get(card_type.name, sc)
    stats = game.characters.get(sc)
    if not stats or not stats.get("hitpoints"):
        return None
    return _troop_or_building_fields(game, card, stats, is_building=False)


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
        core = authentic_core_fields(game, card_type)
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
