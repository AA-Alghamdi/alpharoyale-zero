"""Bridge: re-export the compiled Rust extension module."""
from cr_engine.cr_engine import CREngine  # type: ignore[import]

__all__ = ["CREngine"]
