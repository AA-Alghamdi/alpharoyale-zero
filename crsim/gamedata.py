"""Authentic card stats from the extracted Supercell game data.

This is the *single source of truth* for card numbers. The CSVs in
``cr_engine/gamedata_v2/`` are decoded straight from the Clash Royale APK
(the ``sc/chr_*.sc`` / ``TID_SPELL_*`` fields give it away), so sourcing the
simulator from them — rather than from hand-typed approximations — is what
makes the engine match the real game.

This module is a faithful Python port of ``cr_engine/src/data.rs`` so the
Python and Rust engines agree on every number:

  * level scaling   — base (Level-1) stats × a per-rarity multiplier to reach
    tournament standard (King Level 11);
  * unit conversion — CSV stores millitiles (``/1000`` → tiles), milliseconds
    (``/1000`` → seconds) and a speed code (``/30`` → tiles/sec, matching the
    Python sim's existing convention: Knight 60→2.0, Goblin 120→4.0);
  * ranged damage   — units with an empty ``Damage`` column carry their damage
    (and splash radius) on their projectile, looked up in ``projectiles.csv``.

``apply_authentic_stats`` overlays these numbers onto the hand-authored
``CARD_DEFS`` table, replacing the *core* stat fields while preserving the
hand-curated special-mechanic flags (charge, death-spawn, stun, …) which are
implemented separately. Cards with no CSV entry keep their existing values.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

TOURNAMENT_LEVEL = 11

# Default data directory: the v2 (latest, 121-card) export shipped in the repo.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "cr_engine" / "gamedata_v2"

# ---- Level scaling (ported verbatim from cr_engine/src/data.rs) ----
# Per-rarity multiplier from the Level-1 CSV stat to Level 11 (tournament
# standard). Derived from the wiki: Knight 660→1766 = 2.6757x,
# Giant 1900→4091 = 2.1531x. These account for the differing base levels per
# rarity (Common starts L1, Rare L3, Epic L6, Legendary L9, Champion L11).
_RARITY_MULT_L11: dict[str, float] = {
    "Common": 2.6757,
    "Rare": 2.1531,
    "Epic": 1.7192,
    "Legendary": 1.3781,
    "Champion": 1.0,
    "Hero": 1.0,
}

_RARITY_BASE_LEVEL: dict[str, int] = {
    "Common": 1,
    "Rare": 3,
    "Epic": 6,
    "Legendary": 9,
    "Champion": 11,
    "Hero": 11,
}


def scale_stat(base_value: float, rarity: str, target_level: int = TOURNAMENT_LEVEL) -> float:
    """Scale a Level-1 CSV stat to ``target_level`` accounting for rarity."""
    if base_value == 0:
        return 0.0
    rarity = rarity.capitalize()  # CSVs mix 'Rare'/'rare' casing
    base_level = _RARITY_BASE_LEVEL.get(rarity, 1)
    if target_level <= base_level:
        return float(base_value)
    if target_level == 11:
        return base_value * _RARITY_MULT_L11.get(rarity, 2.6757)
    upgrades = max(0, target_level - base_level)
    return base_value * (1.1 ** upgrades)


def _is_already_l11(raw_rarity: str) -> bool:
    """The two data provenances are distinguishable by rarity casing.

    The original APK export stores **Level-1** base stats with capitalised
    rarities ("Common", "Rare", ...). The ClashStrategic-merged newer cards
    (see scripts/generate_gamedata_v2.py) were written with their **Level-11**
    values already baked in and lower-cased rarities ("common", "rare", ...).
    Only the former need scaling; scaling the latter would double-count.
    """
    r = (raw_rarity or "").strip()
    return bool(r) and r[0].islower()


def scaled_stat(base_value: float, raw_rarity: str, target_level: int = TOURNAMENT_LEVEL) -> float:
    """Scale a CSV stat, accounting for whether the row is already at L11."""
    if _is_already_l11(raw_rarity):
        return float(base_value)
    return scale_stat(base_value, raw_rarity, target_level)


# ---- CSV parsing ----

def _parse_int(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _parse_bool(s: str) -> bool:
    return (s or "").strip().lower() in ("true", "1", "yes")


def _load_csv(path: Path) -> dict[str, dict[str, str]]:
    """Parse an SC-format CSV (header row + type row) keyed by ``Name``."""
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return {}
        next(reader, None)  # skip the type-definition row
        for raw in reader:
            if not raw or not raw[0]:
                continue
            name = raw[0].strip()
            if not name or name.upper().startswith("NOT"):
                continue
            rows[name] = {header[i]: (raw[i] if i < len(raw) else "") for i in range(len(header))}
    return rows


class GameData:
    """All authentic game data, parsed once and scaled to a target level."""

    def __init__(self, data_dir: Path | None = None, level: int = TOURNAMENT_LEVEL) -> None:
        d = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.level = level
        self.characters = _load_csv(d / "characters.csv")
        self.spell_characters = _load_csv(d / "spells_characters.csv")
        self.spell_others = _load_csv(d / "spells_other.csv")
        self.spell_buildings = _load_csv(d / "spells_buildings.csv")
        self.projectiles = _load_csv(d / "projectiles.csv")
        self.area_effects = _load_csv(d / "area_effect_objects.csv")
        self.available = bool(self.characters)


# ---- CardType -> CSV name mapping ----

def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def build_name_map(game: GameData, card_types) -> dict[object, tuple[str, str]]:
    """Map each CardType to ``(csv_key, source)`` where source is one of
    ``char`` (spells_characters), ``spell`` (spells_other), ``bldg``
    (spells_buildings). Cards with no match are omitted."""
    lut: dict[str, tuple[str, str]] = {}
    for key in game.spell_characters:
        lut.setdefault(_norm(key), (key, "char"))
    for key in game.spell_others:
        lut.setdefault(_norm(key), (key, "spell"))
    for key in game.spell_buildings:
        lut.setdefault(_norm(key), (key, "bldg"))
    out: dict[object, tuple[str, str]] = {}
    for ct in card_types:
        hit = lut.get(_norm(ct.name))
        if hit is not None:
            out[ct] = hit
    return out


def _character_damage_and_splash(
    game: GameData, char: dict[str, str], rarity: str
) -> tuple[float, float]:
    """Authentic per-hit damage and splash radius (tiles) for a character.

    Ranged units leave ``Damage`` blank and carry it on their projectile.
    """
    dmg = _parse_int(char.get("Damage", ""))
    splash = _parse_int(char.get("AreaDamageRadius", "")) / 1000.0
    proj_name = (char.get("Projectile", "") or "").strip()
    if dmg == 0 and proj_name and proj_name in game.projectiles:
        proj = game.projectiles[proj_name]
        dmg = _parse_int(proj.get("Damage", ""))
        if splash == 0.0:
            splash = _parse_int(proj.get("Radius", "")) / 1000.0
    return scaled_stat(dmg, rarity), splash


def authentic_core_fields(game: GameData, card_type, source: tuple[str, str]) -> dict | None:
    """Return the authentic core stat fields for a card, or ``None``."""
    key, kind = source
    if kind in ("char", "bldg"):
        card_row = (game.spell_characters if kind == "char" else game.spell_buildings).get(key, {})
        char_name = (card_row.get("SummonCharacter", "") or key).strip()
        char = game.characters.get(char_name) or game.characters.get(key)
        if not char:
            return None
        rarity = char.get("Rarity", "Common") or "Common"
        hp = scaled_stat(_parse_int(char.get("Hitpoints", "")), rarity)
        dmg, splash = _character_damage_and_splash(game, char, rarity)
        hit_speed = max(_parse_int(char.get("HitSpeed", "")) / 1000.0, 0.1)
        attacks_air = _parse_bool(char.get("AttacksAir", ""))
        only_bldg = _parse_bool(char.get("TargetOnlyBuildings", ""))
        flying = _parse_int(char.get("FlyingHeight", "")) > 0
        fields: dict = {
            "cost": _parse_int(card_row.get("ManaCost", "")) or None,
            "hp": hp,
            "damage_per_hit": dmg,
            "dps": (dmg / hit_speed) if hit_speed > 0 else 0.0,
            "hit_speed": hit_speed,
            "load_time": _parse_int(char.get("LoadTime", "")) / 1000.0,
            "speed": _parse_int(char.get("Speed", "")) / 30.0,
            "attack_range": _parse_int(char.get("Range", "")) / 1000.0,
            "sight_range": _parse_int(char.get("SightRange", "")) / 1000.0,
            "deploy_time": max(_parse_int(char.get("DeployTime", "")) / 1000.0, 0.0),
            "is_flying": flying,
            "is_splash": splash > 0.0,
            "splash_radius": splash,
            "collision_radius": _parse_int(char.get("CollisionRadius", "")) / 1000.0 or None,
            "spawn_count": max(_parse_int(card_row.get("SummonNumber", "")), 1),
            "building_lifetime": _parse_int(char.get("LifeTime", "")) / 1000.0,
        }
        # target_mode: only override troops/buildings (spells keep AREA etc.)
        if only_bldg:
            fields["target_mode_buildings"] = True
        else:
            fields["target_mode_air"] = attacks_air
        return {k: v for k, v in fields.items() if v is not None}

    # kind == "spell"
    row = game.spell_others.get(key, {})
    rarity = row.get("Rarity", "Common") or "Common"
    inst = scaled_stat(_parse_int(row.get("InstantDamage", "")), rarity)
    radius = _parse_int(row.get("Radius", "")) / 1000.0
    fields = {
        "cost": _parse_int(row.get("ManaCost", "")) or None,
        "splash_radius": radius or None,
    }
    if inst > 0:
        fields["damage_per_hit"] = inst
    return {k: v for k, v in fields.items() if v is not None}


def apply_authentic_stats(card_defs: dict, *, data_dir: Path | None = None) -> tuple[dict, dict]:
    """Overlay authentic stats onto ``card_defs``.

    Returns ``(new_card_defs, report)`` where report maps each updated CardType
    to the fields that changed. Cards without a CSV entry are left untouched.
    Special-mechanic fields (charge/death-spawn/stun/…) are never overwritten.
    """
    from crsim.cards import TargetMode

    game = GameData(data_dir=data_dir)
    if not game.available:
        logger.warning("Authentic game data not found; keeping hand-coded stats.")
        return card_defs, {}

    name_map = build_name_map(game, list(card_defs.keys()))
    new_defs = dict(card_defs)
    report: dict = {}

    for card_type, source in name_map.items():
        base = card_defs[card_type]
        core = authentic_core_fields(game, card_type, source)
        if not core:
            continue

        tgt_air = core.pop("target_mode_air", None)
        tgt_bldg = core.pop("target_mode_buildings", None)
        if tgt_bldg:
            core["target_mode"] = TargetMode.BUILDINGS
        elif tgt_air is not None and base.target_mode != TargetMode.BUILDINGS:
            core["target_mode"] = TargetMode.AIR_GROUND if tgt_air else TargetMode.GROUND

        # Keep multi-spawn troops' per-unit stat mirror consistent.
        if base.spawn_hp > 0 and "hp" in core:
            core["spawn_hp"] = core["hp"]
        if base.spawn_dps > 0 and "dps" in core:
            core["spawn_dps"] = core["dps"]

        changed = {
            k: (getattr(base, k), v)
            for k, v in core.items()
            if hasattr(base, k) and getattr(base, k) != v
        }
        new_defs[card_type] = dataclasses.replace(base, **core)
        if changed:
            report[card_type] = changed

    logger.info("Applied authentic stats to %d/%d cards", len(report), len(card_defs))
    return new_defs, report
