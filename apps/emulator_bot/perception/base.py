"""Pluggable perception interface + backend registry.

A backend is any object with ``name`` and ``detect(image) -> PerceptionResult``.
New models (KataCR YOLOv8, a custom detector, a cloud API) plug in by
subclassing/duck-typing and registering with ``@register("name")``. Consumers
never depend on a specific model:

    from perception import get_perceptor
    p = get_perceptor("buildabot")
    result = p.detect("frame.png")
    print(result.to_json(indent=2))
"""
from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from .schema import PerceptionResult


@runtime_checkable
class Perceptor(Protocol):
    name: str

    def detect(self, image) -> PerceptionResult:
        """Identify cards + units in a frame (path, PIL.Image, or ndarray)."""
        ...


REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


def available() -> list[str]:
    return sorted(REGISTRY)


def get_perceptor(name: str, **kwargs) -> Perceptor:
    if name not in REGISTRY:
        # lazy-import the backend module (registers on import)
        importlib.import_module(f"perception.backends.{name}")
    if name not in REGISTRY:
        raise KeyError(f"unknown perceptor '{name}'; available: {available()}")
    return REGISTRY[name](**kwargs)
