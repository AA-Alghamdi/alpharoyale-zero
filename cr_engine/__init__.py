"""Bridge: re-export the compiled Rust extension module.

Build the native extension with::

    cd cr_engine && maturin develop --release --features python

which installs the standalone `cr_engine_native` module that this package
re-exports.
"""

from cr_engine_native import CREngine  # type: ignore[import-not-found]

__all__ = ["CREngine"]
