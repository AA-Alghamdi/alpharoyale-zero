"""Hero Form data for Clash Royale.

Source: https://royaleapi.com/blog/hero-knight-and-heroes-december-2025
       https://clashcoachai.com/guides/hero-encyclopedia

Heroes are enhanced card forms with unique activatable abilities (cost elixir).
As of June 2026: 9+ confirmed heroes.
"""

from __future__ import annotations

from dataclasses import dataclass
from crsim.cards import CardType


@dataclass(frozen=True)
class HeroDef:
    """Definition of a card's Hero Form ability."""
    card_type: CardType
    ability_name: str
    ability_cost: int  # Elixir cost to activate ability
    ability_desc: str
    cooldown: float  # seconds before ability can be used again
    # Quantified effects (for simulation)
    taunt_radius: float = 0.0  # Knight: 6.5 tiles
    shield_hp_pct: float = 0.0  # percent of max HP as shield
    damage_area: float = 0.0  # area damage radius
    duration: float = 0.0  # ability duration in seconds


HERO_DEFS: dict[CardType, HeroDef] = {
    CardType.KNIGHT: HeroDef(
        card_type=CardType.KNIGHT,
        ability_name="Triumphant Taunt",
        ability_cost=2,
        ability_desc="Gain a shield and taunt nearby enemies in 6.5-tile radius, forcing them to attack him.",
        cooldown=10.0,
        taunt_radius=6.5,
        shield_hp_pct=30.0,
    ),
    CardType.GIANT: HeroDef(
        card_type=CardType.GIANT,
        ability_name="Heroic Hurl",
        ability_cost=2,
        ability_desc="Throws the highest-hitpoint enemy troop in reach across the arena.",
        cooldown=12.0,
    ),
    CardType.MUSKETEER: HeroDef(
        card_type=CardType.MUSKETEER,
        ability_name="Trusty Turret",
        ability_cost=2,
        ability_desc="Deploys a stationary turret that inherits Musketeer's targeting and damage.",
        cooldown=15.0,
    ),
    CardType.MINI_PEKKA: HeroDef(
        card_type=CardType.MINI_PEKKA,
        ability_name="Breakfast Boost",
        ability_cost=1,
        ability_desc="Heals and gains temporary speed boost after eating pancakes.",
        cooldown=8.0,
    ),
    CardType.GOBLINS: HeroDef(
        card_type=CardType.GOBLINS,
        ability_name="Banner Brigade",
        ability_cost=1,
        ability_desc="Last Goblin defeated leaves a banner for 5s. Activate to summon 3 Goblin Brigades near it.",
        cooldown=0.0,  # One-time use per banner
    ),
    CardType.DARK_PRINCE: HeroDef(
        card_type=CardType.DARK_PRINCE,
        ability_name="Dark Charge",
        ability_cost=2,
        ability_desc="Instantly recharges and dashes toward nearest enemy with increased damage.",
        cooldown=10.0,
    ),
    CardType.BOWLER: HeroDef(
        card_type=CardType.BOWLER,
        ability_name="Boulder Barrage",
        ability_cost=3,
        ability_desc="Throws multiple boulders in a spread pattern covering a wide area.",
        cooldown=12.0,
        damage_area=3.0,
    ),
    CardType.MEGA_MINION: HeroDef(
        card_type=CardType.MEGA_MINION,
        ability_name="Aerial Strike",
        ability_cost=2,
        ability_desc="Dives down dealing area damage to ground units below.",
        cooldown=10.0,
        damage_area=2.5,
    ),
    CardType.MAGIC_ARCHER: HeroDef(
        card_type=CardType.MAGIC_ARCHER,
        ability_name="Arcane Barrage",
        ability_cost=2,
        ability_desc="Fires a volley of piercing arrows in a cone dealing damage to all units hit.",
        cooldown=12.0,
    ),
    CardType.BARBARIAN_BARREL: HeroDef(
        card_type=CardType.BARBARIAN_BARREL,
        ability_name="Rolling Thunder",
        ability_cost=1,
        ability_desc="The barrel rolls faster and further, spawning an enraged Barbarian on impact.",
        cooldown=8.0,
    ),
}


# Champion abilities (these are the 6 actual Champion cards)
@dataclass(frozen=True)
class ChampionDef:
    """Definition of a Champion card's ability."""
    card_type: CardType
    ability_name: str
    ability_cost: int
    ability_desc: str
    cooldown: float


CHAMPION_DEFS: dict[CardType, ChampionDef] = {
    CardType.ARCHER_QUEEN: ChampionDef(
        card_type=CardType.ARCHER_QUEEN,
        ability_name="Cloaking Cape",
        ability_cost=1,
        ability_desc="Becomes invisible and gains attack speed boost for 5 seconds.",
        cooldown=10.0,
    ),
    CardType.GOLDEN_KNIGHT: ChampionDef(
        card_type=CardType.GOLDEN_KNIGHT,
        ability_name="Dashing Dash",
        ability_cost=1,
        ability_desc="Dashes to up to 10 targets within range, dealing 335 damage each.",
        cooldown=8.0,
    ),
    CardType.SKELETON_KING: ChampionDef(
        card_type=CardType.SKELETON_KING,
        ability_name="Soul Summoning",
        ability_cost=2,
        ability_desc="Summons Skeletons from collected souls (max 20). Collects souls from defeated troops nearby.",
        cooldown=12.0,
    ),
    CardType.MIGHTY_MINER: ChampionDef(
        card_type=CardType.MIGHTY_MINER,
        ability_name="Explosive Escape",
        ability_cost=1,
        ability_desc="Burrows underground to a selected location, leaving a bomb at original position.",
        cooldown=10.0,
    ),
    CardType.MONK: ChampionDef(
        card_type=CardType.MONK,
        ability_name="Pensive Protection",
        ability_cost=2,
        ability_desc="Reflects all projectiles back to sender for 3 seconds. Immune to spell damage.",
        cooldown=12.0,
    ),
    CardType.LITTLE_PRINCE: ChampionDef(
        card_type=CardType.LITTLE_PRINCE,
        ability_name="Royal Rescue",
        ability_cost=3,
        ability_desc="Summons the Guardienne (his guardian) to protect and fight alongside him.",
        cooldown=15.0,
    ),
}
