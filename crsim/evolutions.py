"""Card Evolution data for Clash Royale.

Source: https://clashroyale.fandom.com/wiki/Card_Evolution
Contains evolution abilities, stat boosts, and cycle requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from crsim.cards import CardType


@dataclass(frozen=True)
class EvolutionDef:
    """Definition of a card evolution's special ability."""
    card_type: CardType
    cycles: int  # Number of cycles to activate (1 or 2)
    total_cost: int  # Total elixir spent to get evolution (cost * (cycles+1))
    stat_boost: str  # Description of stat change, or "Identical Stats"
    ability_name: str
    ability_desc: str
    # Quantified ability effects (for simulation)
    damage_reduction: float = 0.0  # e.g. Knight: 0.6 = 60% reduction
    bonus_damage_mult: float = 0.0  # e.g. Archers: 1.5 = 50% bonus
    spawn_on_attack: bool = False  # e.g. Skeletons clone on attack
    max_spawns: int = 0  # e.g. Skeletons: 8 max
    knockback_tiles: float = 0.0  # e.g. Royal Giant: 1.0 tile
    pull_radius: float = 0.0  # e.g. Valkyrie: 5.5 tiles
    stun_duration: float = 0.0  # e.g. Tesla: 0.5s
    heal_on_attack: bool = False  # e.g. Bats life leech
    hp_boost_pct: float = 0.0  # e.g. Bats: +50%
    attack_speed_boost: float = 0.0  # e.g. Barbarians: +35%


EVOLUTION_DEFS: dict[CardType, EvolutionDef] = {
    CardType.BARBARIANS: EvolutionDef(
        card_type=CardType.BARBARIANS,
        cycles=1, total_cost=10,
        stat_boost="+10% Hitpoints",
        ability_name="Blade Rage",
        ability_desc="+35% Attack Period and Movement Speed every time they attack, lasting 3 seconds.",
        hp_boost_pct=10.0,
        attack_speed_boost=35.0,
    ),
    CardType.ROYAL_GIANT: EvolutionDef(
        card_type=CardType.ROYAL_GIANT,
        cycles=1, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Royal Shockwave",
        ability_desc="Every attack creates a 2.5-tile radius knockback shockwave. Ground only.",
        knockback_tiles=1.0,
    ),
    CardType.FIRECRACKER: EvolutionDef(
        card_type=CardType.FIRECRACKER,
        cycles=2, total_cost=9,
        stat_boost="Identical Stats",
        ability_name="Spark Attack",
        ability_desc="Sparks deal damage every 0.25s for 3s after attack. 2.5-tile center, 1.2-tile shrapnel. 15% slow.",
    ),
    CardType.SKELETONS: EvolutionDef(
        card_type=CardType.SKELETONS,
        cycles=2, total_cost=3,
        stat_boost="Identical Stats",
        ability_name="Skele-Cloning",
        ability_desc="Every attack spawns another Skeleton. Stops at 8 total.",
        spawn_on_attack=True,
        max_spawns=8,
    ),
    CardType.MORTAR: EvolutionDef(
        card_type=CardType.MORTAR,
        cycles=2, total_cost=12,
        stat_boost="-1 second Attack Period",
        ability_name="Green Siege",
        ability_desc="Every Mortar shot lands with a Goblin.",
        spawn_on_attack=True,
    ),
    CardType.KNIGHT: EvolutionDef(
        card_type=CardType.KNIGHT,
        cycles=2, total_cost=9,
        stat_boost="Identical Stats",
        ability_name="Elbow Reflection",
        ability_desc="While moving/deploying, 60% damage reduction shield. Loses effect on first hit.",
        damage_reduction=0.6,
    ),
    CardType.ROYAL_RECRUITS: EvolutionDef(
        card_type=CardType.ROYAL_RECRUITS,
        cycles=1, total_cost=14,
        stat_boost="Identical Stats",
        ability_name="Charge Squad",
        ability_desc="After shield breaks + 2.5 tiles traveled, charges at Very Fast speed with 2x damage.",
    ),
    CardType.BATS: EvolutionDef(
        card_type=CardType.BATS,
        cycles=2, total_cost=6,
        stat_boost="+50% Hitpoints",
        ability_name="Life Leech",
        ability_desc="Heal on attack at 2 pulses/sec. Can overheal to 2x max HP.",
        heal_on_attack=True,
        hp_boost_pct=50.0,
    ),
    CardType.ARCHERS: EvolutionDef(
        card_type=CardType.ARCHERS,
        cycles=2, total_cost=9,
        stat_boost="+1 tile Range",
        ability_name="Power Shot",
        ability_desc="50% bonus damage to targets 4-6 tiles away.",
        bonus_damage_mult=1.5,
    ),
    CardType.ICE_SPIRIT: EvolutionDef(
        card_type=CardType.ICE_SPIRIT,
        cycles=2, total_cost=3,
        stat_boost="+0.5 tiles Splash Radius",
        ability_name="Frost Blast",
        ability_desc="Freezes for 1.1s, then again after 3s. If target dies, leaves 2-tile freeze zone.",
        stun_duration=1.1,
    ),
    CardType.VALKYRIE: EvolutionDef(
        card_type=CardType.VALKYRIE,
        cycles=2, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Whirlwind Axe",
        ability_desc="Each attack summons a 5.5-tile tornado pulling troops inward for 0.5s.",
        pull_radius=5.5,
    ),
    CardType.BOMBER: EvolutionDef(
        card_type=CardType.BOMBER,
        cycles=2, total_cost=6,
        stat_boost="Identical Stats",
        ability_name="Bomb Bounce",
        ability_desc="Bombs bounce twice after initial throw. Each 2.5 tiles apart. Can't hit same troop twice.",
    ),
    CardType.WALL_BREAKERS: EvolutionDef(
        card_type=CardType.WALL_BREAKERS,
        cycles=2, total_cost=6,
        stat_boost="Identical Stats",
        ability_name="Powder Barrels",
        ability_desc="If killed, barrels break dealing damage, then they continue running at Very Fast speed dealing 50% damage on connect.",
    ),
    CardType.TESLA: EvolutionDef(
        card_type=CardType.TESLA,
        cycles=2, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Pulsating Voltage",
        ability_desc="On surfacing/deploy, emits 6-tile radius pulse dealing damage + 0.5s stun.",
        stun_duration=0.5,
    ),
    CardType.ZAP: EvolutionDef(
        card_type=CardType.ZAP,
        cycles=2, total_cost=6,
        stat_boost="Identical Stats",
        ability_name="Short-Circuit",
        ability_desc="After initial Zap, radius stays, grows 0.5 tiles, same stun + 2x damage.",
    ),
    CardType.BATTLE_RAM: EvolutionDef(
        card_type=CardType.BATTLE_RAM,
        cycles=2, total_cost=12,
        stat_boost="Identical Stats; Barbarians: +10% HP",
        ability_name="Head-First Ram",
        ability_desc="While charging, deals moderate damage + 2-tile knockback on contact. Bounces back on connect dealing 2x damage repeatedly.",
        knockback_tiles=2.0,
        hp_boost_pct=10.0,
    ),
    CardType.WIZARD: EvolutionDef(
        card_type=CardType.WIZARD,
        cycles=1, total_cost=10,
        stat_boost="Identical Stats",
        ability_name="Fiery Field",
        ability_desc="Spawns with 25% HP shield. When destroyed, deals moderate damage + knockback in 3-tile radius.",
    ),
    CardType.GOBLIN_BARREL: EvolutionDef(
        card_type=CardType.GOBLIN_BARREL,
        cycles=2, total_cost=9,
        stat_boost="Identical Stats",
        ability_name="Decoy Barrel",
        ability_desc="Spawns 2 barrels mirrored across lanes. Real has 3 Goblins, decoy has 3 weaker Goblins (75% dmg, 60% HP).",
    ),
    CardType.GOBLIN_GIANT: EvolutionDef(
        card_type=CardType.GOBLIN_GIANT,
        cycles=1, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Tricky Sack",
        ability_desc="After reaching 50% HP, spawns a Goblin 2.5 tiles behind every 2.2 seconds.",
        spawn_on_attack=True,
    ),
    CardType.GOBLIN_DRILL: EvolutionDef(
        card_type=CardType.GOBLIN_DRILL,
        cycles=2, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Whack-A-Drill",
        ability_desc="At 66% HP goes underground + spawns 2 Goblins, re-emerges 90° around tower. Repeats at 33%.",
    ),
    CardType.GOBLIN_CAGE: EvolutionDef(
        card_type=CardType.GOBLIN_CAGE,
        cycles=1, total_cost=8,
        stat_boost="Identical Stats",
        ability_name="Raging Cage",
        ability_desc="Ground troop within 3 tiles is dragged in. Brawler hits trapped troop every 1s. Trapped unit can't attack.",
    ),
    CardType.PEKKA: EvolutionDef(
        card_type=CardType.PEKKA,
        cycles=1, total_cost=14,
        stat_boost="Identical Stats",
        ability_name="Butter-Heal",
        ability_desc="After final blow to any unit, butterfly heals based on defeated unit's HP. Can overheal to +66% max HP.",
        heal_on_attack=True,
    ),
    CardType.MEGA_KNIGHT: EvolutionDef(
        card_type=CardType.MEGA_KNIGHT,
        cycles=1, total_cost=14,
        stat_boost="Identical Stats",
        ability_name="Mega Uppercut",
        ability_desc="Every attack does 4-tile knockback toward nearest enemy Crown Tower.",
        knockback_tiles=4.0,
    ),
    CardType.ELECTRO_DRAGON: EvolutionDef(
        card_type=CardType.ELECTRO_DRAGON,
        cycles=1, total_cost=10,
        stat_boost="Identical Stats",
        ability_name="Infini-Shock",
        ability_desc="Bolts chain infinitely across units within 4 tiles. After 3rd chain: no stun, -33% damage, -20% speed.",
    ),
    CardType.MUSKETEER: EvolutionDef(
        card_type=CardType.MUSKETEER,
        cycles=2, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Sniper Shot",
        ability_desc="Gets 3 limited sniper shots with increased range and damage.",
    ),
    CardType.DART_GOBLIN: EvolutionDef(
        card_type=CardType.DART_GOBLIN,
        cycles=2, total_cost=9,
        stat_boost="Identical Stats",
        ability_name="Toxic Darts",
        ability_desc="Attacks apply poison DoT. Stacks in 3 tiers of increasing damage.",
    ),
    CardType.LUMBERJACK: EvolutionDef(
        card_type=CardType.LUMBERJACK,
        cycles=1, total_cost=8,
        stat_boost="Identical Stats",
        ability_name="Rage Ghost",
        ability_desc="On death, spawns invisible Rage Ghost that attacks while inside Rage effect.",
    ),
    CardType.HUNTER: EvolutionDef(
        card_type=CardType.HUNTER,
        cycles=2, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Net Gun",
        ability_desc="First attack fires a net disabling target's movement/attacks for 4s. Pulls flying units to ground.",
    ),
    CardType.EXECUTIONER: EvolutionDef(
        card_type=CardType.EXECUTIONER,
        cycles=1, total_cost=10,
        stat_boost="Identical Stats",
        ability_name="Boomerang Blitz",
        ability_desc="Axe returns faster and hits harder on the way back.",
    ),
    CardType.WITCH: EvolutionDef(
        card_type=CardType.WITCH,
        cycles=1, total_cost=10,
        stat_boost="Identical Stats",
        ability_name="Skeletal Surge",
        ability_desc="Spawns more Skeletons that inherit splash damage ability.",
    ),
    CardType.CANNON: EvolutionDef(
        card_type=CardType.CANNON,
        cycles=2, total_cost=9,
        stat_boost="Identical Stats",
        ability_name="Cannonball Barrage",
        ability_desc="Periodically fires a barrage of cannonballs in an area.",
    ),
    CardType.FURNACE: EvolutionDef(
        card_type=CardType.FURNACE,
        cycles=2, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Blazing Furnace",
        ability_desc="Spawned Fire Spirits deal increased damage and leave a burn area.",
    ),
    CardType.GIANT_SNOWBALL: EvolutionDef(
        card_type=CardType.GIANT_SNOWBALL,
        cycles=2, total_cost=6,
        stat_boost="+1 second Slowdown",
        ability_name="Avalanche",
        ability_desc="Extended slow duration and increased knockback effect.",
    ),
    CardType.INFERNO_DRAGON: EvolutionDef(
        card_type=CardType.INFERNO_DRAGON,
        cycles=2, total_cost=12,
        stat_boost="Identical Stats",
        ability_name="Inferno Overdrive",
        ability_desc="After reaching max damage tier, beam splits to hit nearby targets.",
    ),
    CardType.ROYAL_HOGS: EvolutionDef(
        card_type=CardType.ROYAL_HOGS,
        cycles=2, total_cost=10,
        stat_boost="Identical Stats",
        ability_name="Royal Rush",
        ability_desc="Hogs gain speed boost and damage resistance while charging together.",
    ),
}
