"""CRSim — fast Clash Royale simulator for RL training."""

from crsim.constants import (
    ARENA_H,
    ARENA_W,
    MAX_ELIXIR,
    MAX_ENTITIES,
    NUM_CARD_TYPES,
    TICK_DURATION,
)
from crsim.game import CRGame, GamePhase, GameResult

__all__ = [
    "CRGame",
    "GamePhase",
    "GameResult",
    "ARENA_W",
    "ARENA_H",
    "TICK_DURATION",
    "MAX_ELIXIR",
    "MAX_ENTITIES",
    "NUM_CARD_TYPES",
]
