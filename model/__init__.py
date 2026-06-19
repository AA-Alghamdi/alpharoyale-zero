"""Neural network and feature extraction for ClashRoyale-Zero."""

from model.features import encode_state
from model.network import CRZeroNet

__all__ = ["CRZeroNet", "encode_state"]
