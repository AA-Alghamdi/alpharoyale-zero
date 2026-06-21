"""Global constants for the Clash Royale simulator."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Arena geometry
# ---------------------------------------------------------------------------
ARENA_W: int = 18  # tiles wide
ARENA_H: int = 32  # tiles tall

# River spans rows 15–16; bridges at columns 3–4 and 13–14
RIVER_ROW_LO: int = 15
RIVER_ROW_HI: int = 16
BRIDGE_LEFT_COLS: tuple[int, int] = (3, 4)
BRIDGE_RIGHT_COLS: tuple[int, int] = (13, 14)

# ---------------------------------------------------------------------------
# Tower positions  (x, y)
# ---------------------------------------------------------------------------
# Player 0 (bottom)
P0_KING_POS: tuple[float, float] = (9.0, 1.0)
P0_PRINCESS_L_POS: tuple[float, float] = (4.0, 4.0)
P0_PRINCESS_R_POS: tuple[float, float] = (13.0, 4.0)

# Player 1 (top)
P1_KING_POS: tuple[float, float] = (9.0, 30.0)
P1_PRINCESS_L_POS: tuple[float, float] = (4.0, 27.0)
P1_PRINCESS_R_POS: tuple[float, float] = (13.0, 27.0)

KING_TOWER_HP: float = 4008.0
PRINCESS_TOWER_HP: float = 2534.0
KING_TOWER_DAMAGE: float = 109.0
PRINCESS_TOWER_DAMAGE: float = 109.0
KING_TOWER_RANGE: float = 7.0
PRINCESS_TOWER_RANGE: float = 7.5
TOWER_ATTACK_INTERVAL: float = 1.0  # seconds between attacks

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
# The real game uses ~20 ticks/sec (50ms) internally for deterministic sim.
# We match this for physics accuracy.
TICK_DURATION: float = 0.05  # 50ms per tick (matches real game internal rate)
TICKS_PER_SECOND: int = 20   # 1.0 / TICK_DURATION

# For RL: the agent only makes decisions every DECISION_INTERVAL ticks.
# This reduces MCTS calls from 7200 to 720 per game (10× faster training).
# Between decisions, the game advances with the current action (wait).
DECISION_INTERVAL: int = 10  # agent decides every 0.5 seconds (10 ticks × 50ms)

REGULAR_TIME_TICKS: int = 3600  # 3 minutes = 180s / 0.05
OVERTIME_TICKS: int = 2400       # 2 minutes = 120s / 0.05 (triple elixir)
SUDDEN_DEATH_TICKS: int = 1200   # 1 minute = 60s / 0.05 (triple elixir)

# Double Elixir begins in the last 60s of regulation (the 2:00 mark of a 3:00
# match), matching the real game: 1x for 0:00-2:00, 2x for 2:00-3:00.
DOUBLE_ELIXIR_START_TICKS: int = REGULAR_TIME_TICKS - 1200  # tick 2400 = 120s

TOTAL_MAX_TICKS: int = REGULAR_TIME_TICKS + OVERTIME_TICKS + SUDDEN_DEATH_TICKS

# Elixir regeneration: 1 elixir per 2.8 s in normal time. The real game ramps
# 1x -> 2x (last minute of regulation) -> 3x (overtime / sudden death).
ELIXIR_REGEN_NORMAL: float = TICK_DURATION / 2.8  # ~0.1786 per tick
ELIXIR_REGEN_DOUBLE: float = ELIXIR_REGEN_NORMAL * 2.0
ELIXIR_REGEN_OVERTIME: float = ELIXIR_REGEN_NORMAL * 3.0
ELIXIR_REGEN_SUDDEN: float = ELIXIR_REGEN_NORMAL * 3.0

STARTING_ELIXIR: float = 5.0
MAX_ELIXIR: float = 10.0

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
MAX_ENTITIES: int = 128  # per game (both sides combined)
NUM_CARD_TYPES: int = 125  # 0-124 inclusive (all CR cards as of June 2026)

# Movement speed tiers
# Internal CR values: Slow=45, Medium=60, Fast=90, Very Fast=120
# Conversion: tiles_per_sec = internal_speed / 30
# But arena is 18 tiles wide, 32 tiles tall
# Real speeds measured as tiles/sec:
SPEED_SLOW: float = 1.5        # 45/30 = Golem, PEKKA, Giant, Royal Giant
SPEED_MEDIUM: float = 2.0      # 60/30 = Knight, Valkyrie, Witch, Musketeer, Prince
SPEED_FAST: float = 3.0        # 90/30 = Baby Dragon, Mini PEKKA, Minions, Guards
SPEED_VERY_FAST: float = 4.0   # 120/30 = Hog Rider, Goblins, Elite Barbarians

# Rage multiplier (confirmed from CR data: 1.4x, NOT 1.35x)
RAGE_MULTIPLIER: float = 1.4
RAGE_DURATION: float = 7.5  # seconds (at Tournament Standard, Rage spell duration is level-dependent but base ~7.5)

# Slowdown multiplier (Ice Wizard, Snowball, etc.)
SLOW_MULTIPLIER: float = 0.65  # 35% speed reduction
SLOW_DURATION_ICE_WIZ: float = 2.0  # seconds
SLOW_DURATION_SNOWBALL: float = 2.5

# Freeze duration (spell)
FREEZE_DURATION: float = 4.0  # seconds at Tournament Standard

# Poison duration and tick
POISON_DURATION: float = 8.0  # seconds

# Tornado pull speed
TORNADO_PULL_SPEED: float = 2.0  # tiles/sec toward center
TORNADO_DURATION: float = 1.0  # seconds

# Deploy time (standard for all troops)
DEPLOY_TIME_STANDARD: float = 1.0  # seconds

# Sight range (aggro range) defaults
SIGHT_RANGE_DEFAULT: float = 5.5  # most troops
SIGHT_RANGE_BUILDING_TARGETER: float = 7.5  # Giant, Golem, Hog Rider
SIGHT_RANGE_PEKKA: float = 5.0
SIGHT_RANGE_MUSKETEER: float = 6.0
SIGHT_RANGE_HOG: float = 9.5
SIGHT_RANGE_PRINCESS: float = 9.5

# Pushback values (tiles)
PUSHBACK_FIREBALL: float = 1.8
PUSHBACK_LOG: float = 1.5
PUSHBACK_BOWLER: float = 1.0

# Tower damage reduction for spells
SPELL_TOWER_DAMAGE_FACTOR: float = 0.35  # spells do 35% damage to towers

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
NUM_HAND_SLOTS: int = 4
_PLACEMENT_ACTION_COUNT: int = NUM_HAND_SLOTS * ARENA_W * ARENA_H
# One extra discrete action activates a deployed champion's ability (CR allows at
# most one champion per deck and one on the field, so a single action suffices).
ABILITY_ACTION: int = _PLACEMENT_ACTION_COUNT
WAIT_ACTION: int = _PLACEMENT_ACTION_COUNT + 1
ACTION_SPACE_SIZE: int = _PLACEMENT_ACTION_COUNT + 2  # placements + ABILITY + WAIT

# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
# Spatial planes are *semantic*, not one-per-card. Per-card identity is carried
# by the entity-transformer's type embedding, so the spatial map only needs the
# aggregate information a conv net benefits from (threat/HP/type density per
# side + map). This replaces the old 254-plane (one density plane per card per
# side) encoding, which was ~14x larger, mostly empty, and slow to build.
#
# Plane indices (per current-player perspective; board is flipped so the
# current player is always at the bottom):
CH_FRIENDLY_HP: int = 0       # HP-weighted unit density (friendly)
CH_ENEMY_HP: int = 1          # HP-weighted unit density (enemy)
CH_FRIENDLY_DPS: int = 2      # DPS-weighted density / threat (friendly)
CH_ENEMY_DPS: int = 3         # DPS-weighted density / threat (enemy)
CH_FRIENDLY_GROUND: int = 4   # ground-unit count density (friendly)
CH_ENEMY_GROUND: int = 5      # ground-unit count density (enemy)
CH_FRIENDLY_AIR: int = 6      # air-unit count density (friendly)
CH_ENEMY_AIR: int = 7         # air-unit count density (enemy)
CH_FRIENDLY_BUILDING: int = 8  # non-tower building HP (friendly)
CH_ENEMY_BUILDING: int = 9     # non-tower building HP (enemy)
CH_FRIENDLY_TOWER_HP: int = 10  # tower HP at tower cell (friendly)
CH_ENEMY_TOWER_HP: int = 11     # tower HP at tower cell (enemy)
CH_PLACEMENT_MASK: int = 12     # valid placement cells for current player
CH_STATIC_MAP: int = 13         # river=1.0, bridge=0.5
CH_FRIENDLY_WINCON: int = 14    # building-targeting unit density (friendly)
CH_ENEMY_WINCON: int = 15       # building-targeting unit density (enemy)
CH_FRIENDLY_READY: int = 16     # attack-ready unit density (friendly)
CH_ENEMY_READY: int = 17        # attack-ready unit density (enemy)

SPATIAL_CHANNELS: int = 18
SCALAR_FEATURES: int = 2 + 5 * NUM_CARD_TYPES + 14  # 641 for 125 cards

# ---------------------------------------------------------------------------
# Reward shaping (potential-based; Ng et al. 1999 — policy-invariant)
# ---------------------------------------------------------------------------
# Terminal crown/win reward stays dominant (TERMINAL_CROWN_SCALE = 1.0). The
# dense budget is small relative to it so the optimum stays anchored to winning.
#   GAMMA per 50ms tick; 0.997^1200 (~60 s) ~= 0.027 -> ~3% of terminal.
GAMMA: float = 0.997
W_TOWER: float = 0.30          # tower-HP differential (in potential)
W_ELIXIR: float = 0.05         # elixir advantage (in potential)
W_LEAK: float = 0.10           # leak penalty (the one NON-potential term)
TERMINAL_CROWN_SCALE: float = 1.0
W_NAKED_PUSH: float = 0.05     # win-con deployed deep with no support nearby
