"""Pluggable Clash Royale perception infra.

One model-agnostic layer: any frame -> structured JSON (the 4 hand cards, all
units identified against the canonical 125-card vocabulary, with pixel + tile
positions, tower HP, elixir, screen). Plug in new detection models via
perception.base.register.
"""
from .base import Perceptor, available, get_perceptor, register
from .schema import HandCard, PerceptionResult, Tower, Unit

__all__ = [
    "Perceptor", "available", "get_perceptor", "register",
    "PerceptionResult", "Unit", "HandCard", "Tower",
]
