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
TICK_DURATION: float = 0.5  # seconds per simulation tick

REGULAR_TIME_TICKS: int = 360  # 3 minutes = 180s / 0.5
OVERTIME_TICKS: int = 360
SUDDEN_DEATH_TICKS: int = 360

TOTAL_MAX_TICKS: int = REGULAR_TIME_TICKS + OVERTIME_TICKS + SUDDEN_DEATH_TICKS

# Elixir regeneration: 1 elixir per 2.8 s in normal time
ELIXIR_REGEN_NORMAL: float = TICK_DURATION / 2.8  # ~0.1786 per tick
ELIXIR_REGEN_OVERTIME: float = ELIXIR_REGEN_NORMAL * 2.0
ELIXIR_REGEN_SUDDEN: float = ELIXIR_REGEN_NORMAL * 3.0

STARTING_ELIXIR: float = 5.0
MAX_ELIXIR: float = 10.0

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
MAX_ENTITIES: int = 128  # per game (both sides combined)
NUM_CARD_TYPES: int = 125  # 0-124 inclusive (all CR cards as of June 2026)

# Movement speed tiers (tiles per second)
SPEED_SLOW: float = 1.0
SPEED_MEDIUM: float = 1.5
SPEED_FAST: float = 2.5
SPEED_VERY_FAST: float = 3.0

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
NUM_HAND_SLOTS: int = 4
ACTION_SPACE_SIZE: int = NUM_HAND_SLOTS * ARENA_W * ARENA_H + 1  # +1 for WAIT
WAIT_ACTION: int = ACTION_SPACE_SIZE - 1

# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
SPATIAL_CHANNELS: int = 2 * NUM_CARD_TYPES + 4  # 254 for 125 cards
SCALAR_FEATURES: int = 2 + 5 * NUM_CARD_TYPES + 14  # 641 for 125 cards
