"""Load authentic card stats from gamedata.json or cr-csv decoded files.

Sources:
  - gamedata.json: Full card data extracted from CR APK (clash-simulator format)
  - cr-csv: Decoded CSV files from https://github.com/smlbiobot/cr-csv
  - Supercell API: /v1/cards endpoint (limited stats)

This replaces the hand-coded card stats in cards.py with real game data,
supporting all 100+ cards with accurate HP, damage, range, speed, etc.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Tournament standard level (Level 11 in old system, Level 14 in new system)
TOURNAMENT_LEVEL = 11


@dataclass
class CardData:
    """Authentic card stats from game data."""
    name: str
    key: str  # internal key (e.g., "Knight")
    card_id: int = 0
    elixir: int = 0
    rarity: str = "Common"  # Common, Rare, Epic, Legendary, Champion
    card_type: str = "Troop"  # Troop, Spell, Building

    # Stats at tournament standard
    hitpoints: int = 0
    damage: int = 0
    dps: float = 0.0
    hit_speed: float = 1.0  # seconds
    range: float = 0.0  # tiles (melee = 0.8)
    speed: str = "Medium"  # Slow, Medium, Fast, Very Fast
    deploy_time: float = 1.0
    count: int = 1  # number spawned (e.g., Skeletons = 3)
    lifetime: float = 0.0  # for buildings
    spawn_speed: float = 0.0  # for spawners

    # Movement speed in tiles/sec
    @property
    def tiles_per_sec(self) -> float:
        speed_map = {
            "Slow": 1.0,
            "Medium": 1.5,
            "Fast": 2.5,
            "Very Fast": 3.0,
        }
        return speed_map.get(self.speed, 1.5)

    # Is ranged?
    @property
    def is_ranged(self) -> bool:
        return self.range > 2.0

    # Is flying?
    is_flying: bool = False

    # Is area damage?
    is_area_damage: bool = False

    # Spell stats
    spell_radius: float = 0.0
    spell_damage: int = 0
    crown_tower_damage: int = 0


class GameDataLoader:
    """Loads card data from various sources."""

    def __init__(self) -> None:
        self.cards: dict[str, CardData] = {}
        self.cards_by_id: dict[int, CardData] = {}

    def load_from_json(self, path: str) -> int:
        """Load from gamedata.json (clash-simulator format).

        Expected format::
            {
              "cards": {
                "Knight": {"elixir": 3, "hitpoints": 1452, "damage": 167, ...},
                ...
              }
            }
        """
        p = Path(path)
        if not p.exists():
            logger.warning("gamedata.json not found at %s", path)
            return 0

        with open(p) as f:
            data = json.load(f)

        cards_data = data.get("cards", data)
        count = 0

        for name, stats in cards_data.items():
            card = CardData(
                name=name,
                key=name,
                elixir=stats.get("elixir", 0),
                hitpoints=stats.get("hitpoints", 0),
                damage=stats.get("damage", 0),
                dps=stats.get("dps", 0.0),
                hit_speed=stats.get("hitSpeed", stats.get("hit_speed", 1.0)),
                range=stats.get("range", 0.0),
                speed=stats.get("speed", "Medium"),
                deploy_time=stats.get("deployTime", stats.get("deploy_time", 1.0)),
                count=stats.get("count", 1),
                card_type=stats.get("type", "Troop"),
                rarity=stats.get("rarity", "Common"),
                is_flying=stats.get("isFlying", stats.get("is_flying", False)),
                is_area_damage=stats.get("isAreaDamage", stats.get("is_area_damage", False)),
                spell_radius=stats.get("spellRadius", stats.get("spell_radius", 0.0)),
                spell_damage=stats.get("spellDamage", stats.get("spell_damage", 0)),
                lifetime=stats.get("lifetime", 0.0),
            )
            self.cards[name] = card
            count += 1

        logger.info("Loaded %d cards from %s", count, path)
        return count

    def load_from_csv(self, characters_csv: str, spells_csv: str | None = None) -> int:
        """Load from cr-csv decoded CSV files.

        characters.csv columns: Name, Rarity, SightRange, DeployTime, ChargeRange,
        Speed, Hitpoints, HitSpeed, LoadTime, Damage, DamagePerSecond, ...
        """
        count = 0

        if Path(characters_csv).exists():
            with open(characters_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "").strip()
                    if not name:
                        continue

                    card = CardData(
                        name=name,
                        key=name,
                        hitpoints=int(row.get("Hitpoints", 0) or 0),
                        damage=int(row.get("Damage", 0) or 0),
                        dps=float(row.get("DamagePerSecond", 0) or 0),
                        hit_speed=float(row.get("HitSpeed", 1000) or 1000) / 1000.0,
                        range=float(row.get("Range", 0) or 0) / 1000.0,
                        speed=row.get("Speed", "medium").capitalize(),
                        deploy_time=float(row.get("DeployTime", 1000) or 1000) / 1000.0,
                        rarity=row.get("Rarity", "Common"),
                        card_type="Troop",
                    )
                    self.cards[name] = card
                    count += 1

        if spells_csv and Path(spells_csv).exists():
            with open(spells_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "").strip()
                    if not name:
                        continue

                    card = CardData(
                        name=name,
                        key=name,
                        spell_radius=float(row.get("Radius", 0) or 0) / 1000.0,
                        spell_damage=int(row.get("Damage", 0) or 0),
                        crown_tower_damage=int(row.get("CrownTowerDamage", 0) or 0),
                        card_type="Spell",
                    )
                    self.cards[name] = card
                    count += 1

        logger.info("Loaded %d cards from CSV files", count)
        return count

    def load_from_api_response(self, api_data: list[dict]) -> int:
        """Load from Supercell API /v1/cards response.

        Each item: {id, name, maxLevel, iconUrls: {medium}}
        (Note: API doesn't include full stats, mainly for card ID mapping)
        """
        count = 0
        for item in api_data:
            card = CardData(
                name=item["name"],
                key=item["name"],
                card_id=item.get("id", 0),
            )
            self.cards[item["name"]] = card
            self.cards_by_id[item.get("id", 0)] = card
            count += 1

        logger.info("Loaded %d card IDs from API", count)
        return count

    def get_card(self, name: str) -> CardData | None:
        return self.cards.get(name)

    def get_all_cards(self) -> list[CardData]:
        return list(self.cards.values())

    def get_troops(self) -> list[CardData]:
        return [c for c in self.cards.values() if c.card_type == "Troop"]

    def get_spells(self) -> list[CardData]:
        return [c for c in self.cards.values() if c.card_type == "Spell"]

    def get_buildings(self) -> list[CardData]:
        return [c for c in self.cards.values() if c.card_type == "Building"]
