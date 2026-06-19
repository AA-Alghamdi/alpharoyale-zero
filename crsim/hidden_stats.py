"""Hidden card stats for Clash Royale simulation accuracy.

Sources:
- https://royaleapi.com/blog/secret-stats (Hit Time / Load Time mechanics)
- https://clashroyale.fandom.com/wiki/User_blog:AesDragon/Hidden_card_stats
- https://guidescroll.com/2016/08/clash-royale-hidden-card-stats-list/

Key mechanic: First Hit = Hit Time - Load Time
Troops pre-load their first attack while moving. Load Time is the max
time that can be "skipped" from the first hit timer.
"""

from __future__ import annotations

# Movement speed tiers (internal CR units → tiles per second)
# Conversion factor: tiles_per_sec = internal_speed / 30
# Based on measured game data:
#   Hog Rider (120) crosses ~14 tiles in ~3.5s = 4.0 tiles/sec
#   Knight (60) crosses ~14 tiles in ~7s = 2.0 tiles/sec
#   Giant (45) crosses ~14 tiles in ~9.3s = 1.5 tiles/sec
SPEED_TIERS = {
    "Slow": 45,        # 1.5 tiles/sec (Golem, PEKKA, Giant)
    "Medium": 60,      # 2.0 tiles/sec (Knight, Valkyrie, Witch)
    "Fast": 90,        # 3.0 tiles/sec (Baby Dragon, Mini PEKKA, Minions)
    "Very Fast": 120,  # 4.0 tiles/sec (Hog Rider, Goblins, Elite Barbs)
}

# Our cards.py uses tiles/sec directly:
SPEED_TO_TILES_PER_SEC = {
    45: 1.5,
    60: 2.0,
    90: 3.0,
    120: 4.0,
}

# Projectile speeds (tiles per minute → rough tiles per second)
# Source: Fandom wiki blog post
PROJECTILE_SPEEDS = {
    "Archers": 600,       # 10 tiles/sec
    "Musketeer": 800,     # 13.3 tiles/sec
    "Spear Goblins": 500, # 8.3 tiles/sec
    "Wizard": 600,        # 10 tiles/sec
    "Magic Archer": 1000, # 16.7 tiles/sec
    "Cannon": 1000,       # 16.7 tiles/sec
    "Mega Minion": 1000,  # 16.7 tiles/sec
    "Flying Machine": 800,# 13.3 tiles/sec
    "Dart Goblin": 800,   # 13.3 tiles/sec
    "Princess": 400,      # 6.7 tiles/sec
    "Hunter": 550,        # 9.2 tiles/sec
    "Mortar": 300,        # 5.0 tiles/sec
    "X-Bow": 1200,        # 20 tiles/sec
    "Baby Dragon": 600,   # 10 tiles/sec
    "Electro Wizard": 800,# 13.3 tiles/sec
    "Ice Wizard": 600,    # 10 tiles/sec
    "Executioner": 400,   # 6.7 tiles/sec (slower for boomerang)
    "Bowler": 300,        # 5.0 tiles/sec (boulder rolls)
    "Firecracker": 600,   # 10 tiles/sec
    "Lava Hound": 400,    # 6.7 tiles/sec
}

# Spell travel speeds (tiles per minute)
SPELL_TRAVEL_SPEEDS = {
    "Arrows": 1100,       # 18.3 tiles/sec
    "Fireball": 900,      # 15 tiles/sec
    "Rocket": 350,        # 5.8 tiles/sec
    "Lightning": 500,     # 8.3 tiles/sec
    "Goblin Barrel": 400, # 6.7 tiles/sec
    "Miner": 650,         # 10.8 tiles/sec (burrow)
    "The Log": 350,       # 5.8 tiles/sec
    "Barbarian Barrel": 200,  # 3.3 tiles/sec
    "Royal Delivery": 300,    # 5.0 tiles/sec
}

# Mass values (affects knockback/push interactions)
# Higher mass = less knockback received
TROOP_MASS = {
    "Knight": 6,
    "Giant": 18,
    "PEKKA": 18,
    "Golem": 20,
    "Golemite": 6,
    "Giant Skeleton": 15,
    "Royal Giant": 18,
    "Mega Knight": 15,
    "Bowler": 18,
    "Sparky": 18,
    "Lava Hound": 5,
    "Prince": 6,
    "Dark Prince": 6,
    "Valkyrie": 5,
    "Hog Rider": 4,
    "Miner": 6,
    "Musketeer": 5,
    "Wizard": 5,
    "Witch": 4,
    "Baby Dragon": 5,
    "Mini Pekka": 4,
    "Barbarian": 4,
    "Bomber": 4,
    "Archer": 3,
    "Goblin": 2,
    "Skeleton": 1,
    "Minion": 2,
    "Spear Goblin": 1,
    "Ice Spirit": 1,
    "Fire Spirit": 1,
    "Electro Spirit": 1,
    "Lumberjack": 4,
}

# Buff/debuff multipliers (for spell effects)
RAGE_MULTIPLIER = 1.4  # 40% boost to attack/move speed (confirmed from CR data)
FREEZE_DURATION_BASE = 4.0  # seconds at tournament standard
POISON_SLOW = 0.0  # poison no longer slows (removed in 2019)
SLOW_MULTIPLIER = 0.65  # 35% slow (Ice Wizard, Giant Snowball)
GRAVEYARD_SPAWN_INTERVAL = 0.5  # seconds between skeleton spawns
GRAVEYARD_DURATION = 10.0  # total duration
GRAVEYARD_MAX_SKELETONS = 20  # maximum spawned

# Sight range (tiles) - how far a troop can "see" an enemy before engaging
DEFAULT_SIGHT_RANGE = 5.5
BUILDING_SIGHT_RANGE = 6.0  # most defensive buildings
MORTAR_SIGHT_RANGE = 12.0  # buildings with long range
MORTAR_BLIND_SPOT = 4.5  # mortar can't hit within this range

# Deploy time
DEFAULT_DEPLOY_TIME = 1.0  # seconds
HEAVY_DEPLOY_TIME = 3.0  # Golem, Lava Hound, Elixir Golem
