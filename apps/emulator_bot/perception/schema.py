"""Unified perception output schema.

Every backend returns a :class:`PerceptionResult`. Positions are reported BOTH
in source-image pixels (``bbox_px`` / ``center_px``, scaled to the input frame's
real dimensions) and in game-logical tile coordinates (``tile``), so consumers
can use whichever they need. ``to_json()`` gives a stable JSON contract.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Unit:
    name: str           # canonical card name (or raw label if unmapped)
    card_id: int        # 0..124 canonical id, or -1 if unmapped
    owner: str          # "ally" | "enemy"
    confidence: float
    bbox_px: list[int]  # [x, y, w, h] in source-image pixels
    center_px: list[int]
    tile: list[int]     # [tile_x, tile_y] game grid (18 x 32)


@dataclass
class HandCard:
    slot: int           # 0..3
    name: str | None
    card_id: int
    ready: bool         # affordable / playable now
    confidence: float
    bbox_px: list[int]


@dataclass
class Tower:
    side: str           # "ally" | "enemy"
    kind: str           # "princess_left" | "princess_right" | "king"
    hp: int


@dataclass
class PerceptionResult:
    backend: str
    image_size: list[int]          # [W, H] of the source frame
    screen: str                    # "in_game" | "lobby" | ...
    elixir: float
    hand: list[HandCard]
    allies: list[Unit]
    enemies: list[Unit]
    towers: list[Tower]
    opponent_cards_seen: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)
