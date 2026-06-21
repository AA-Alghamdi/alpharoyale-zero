"""Structured schema for the Clash Royale gameplay dataset.

The schema is designed to capture enough state/action information per match to
support imitation learning and AlphaStar-style RL. Any field that cannot be read
reliably from a frame is allowed to be ``None`` and carries a confidence score,
so downstream training can filter on quality.

Coordinate system
------------------
The Clash Royale arena is modelled as an 18 (wide) x 32 (tall) tile grid.
``x`` increases to the right, ``y`` increases from the player's (bottom) side
toward the opponent's (top) side. The river sits around ``y == 16``. Positions
are floats so sub-tile precision from frame annotation is preserved.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

ARENA_WIDTH_TILES = 18
ARENA_HEIGHT_TILES = 32
SCHEMA_VERSION = "0.1.0"


class Owner(str, Enum):
    PLAYER = "player"  # the camera-bottom player whose elixir bar is visible
    OPPONENT = "opponent"
    UNKNOWN = "unknown"


class GameMode(str, Enum):
    LADDER = "ladder"
    PATH_OF_LEGENDS = "path_of_legends"
    CHALLENGE = "challenge"
    TOURNAMENT = "tournament"
    FRIENDLY = "friendly"
    DUEL = "duel"
    TWO_V_TWO = "2v2"
    UNKNOWN = "unknown"


class ElixirPhase(str, Enum):
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    OVERTIME = "overtime"
    UNKNOWN = "unknown"


class Result(str, Enum):
    PLAYER_WIN = "player_win"
    OPPONENT_WIN = "opponent_win"
    DRAW = "draw"
    UNKNOWN = "unknown"


class StyleTag(str, Enum):
    BEATDOWN = "beatdown"
    CONTROL = "control"
    CYCLE = "cycle"
    BRIDGE_SPAM = "bridge_spam"
    SIEGE = "siege"
    BAIT = "bait"
    SPELL_BAIT = "spell_bait"
    LAVALOON = "lavaloon"
    GRAVEYARD = "graveyard"
    UNKNOWN = "unknown"


class TilePosition(BaseModel):
    x: float = Field(..., ge=-1, le=ARENA_WIDTH_TILES + 1)
    y: float = Field(..., ge=-1, le=ARENA_HEIGHT_TILES + 1)


class TowerState(BaseModel):
    king_hp: int | None = None
    left_princess_hp: int | None = None
    right_princess_hp: int | None = None
    king_activated: bool | None = None


class Unit(BaseModel):
    """A troop/building visible on the arena at a given timestep."""

    owner: Owner = Owner.UNKNOWN
    card: str = "unknown"
    position: TilePosition | None = None
    count: int = 1  # e.g. a Skeleton Army spawns many; model may group them
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class Action(BaseModel):
    """A card placement detected between the previous frame and this one."""

    owner: Owner = Owner.UNKNOWN
    card: str = "unknown"
    position: TilePosition | None = None
    elixir_cost: int | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class Timestep(BaseModel):
    """Full observable game state at one sampled instant."""

    game_time_s: float | None = None  # seconds since match start (in-game clock)
    video_time_s: float  # seconds into the source video (always known)
    frame_path: str | None = None

    elixir_player: float | None = Field(None, ge=0, le=10)
    elixir_opponent: float | None = Field(None, ge=0, le=10)
    phase: ElixirPhase = ElixirPhase.UNKNOWN

    player_towers: TowerState = Field(default_factory=TowerState)
    opponent_towers: TowerState = Field(default_factory=TowerState)
    crowns_player: int | None = Field(None, ge=0, le=3)
    crowns_opponent: int | None = Field(None, ge=0, le=3)

    units: list[Unit] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)

    is_gameplay: bool = True  # False for menu/intro/replay/non-arena frames
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    raw_annotation: dict | None = None  # raw annotation output for debugging/audit


class PlayerProfile(BaseModel):
    name: str | None = None
    clan: str | None = None
    trophies: int | None = None
    king_level: int | None = None
    deck: list[str] = Field(default_factory=list)  # up to 8 cards, inferred over game
    style_tags: list[StyleTag] = Field(default_factory=list)


class GameRecord(BaseModel):
    """One full match extracted from a source video."""

    schema_version: str = SCHEMA_VERSION

    # provenance
    video_id: str
    video_url: str
    channel: str | None = None
    upload_date: str | None = None  # YYYYMMDD
    game_index_in_video: int = 0
    segment_start_s: float = 0.0
    segment_end_s: float = 0.0

    mode: GameMode = GameMode.UNKNOWN
    arena: str | None = None
    trophy_range: str | None = None

    player: PlayerProfile = Field(default_factory=PlayerProfile)
    opponent: PlayerProfile = Field(default_factory=PlayerProfile)

    timeline: list[Timestep] = Field(default_factory=list)

    result: Result = Result.UNKNOWN
    final_crowns_player: int | None = None
    final_crowns_opponent: int | None = None

    # extraction metadata
    annotation_model: str | None = None
    sample_fps: float | None = None
    prompt_version: str | None = None
    mean_confidence: float | None = None
    notes: str | None = None


if __name__ == "__main__":
    import json

    # Emit the JSON Schema so it can be reviewed / used to validate exports.
    print(json.dumps(GameRecord.model_json_schema(), indent=2))
