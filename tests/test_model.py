"""Tests for the neural network and feature extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from crsim.constants import (
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
)
from crsim.game import CRGame
from model.features import encode_batch, encode_state
from model.network import CRZeroNet


def test_encode_state_shape():
    """Feature encoding produces correct shapes."""
    game = CRGame(seed=0)
    spatial, scalar = encode_state(game, player=0)
    assert spatial.shape == (SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    assert scalar.shape == (SCALAR_FEATURES,)
    assert spatial.dtype == np.float32
    assert scalar.dtype == np.float32


def test_encode_state_flipped():
    """Encoding from player 1's perspective flips the board."""
    game = CRGame(seed=0)
    sp0, sc0 = encode_state(game, player=0)
    sp1, sc1 = encode_state(game, player=1)
    # Spatial channels should differ (flipped)
    assert not np.allclose(sp0, sp1)


def test_encode_batch():
    """Batch encoding works correctly."""
    games = [CRGame(seed=i) for i in range(4)]
    players = [0, 1, 0, 1]
    spatials, scalars = encode_batch(games, players)
    assert spatials.shape == (4, SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    assert scalars.shape == (4, SCALAR_FEATURES)


def test_network_forward():
    """Network produces correct output shapes."""
    model = CRZeroNet(n_res_blocks=2, n_filters=32)  # small for testing
    bsz = 4

    spatial = torch.randn(bsz, SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    scalar = torch.randn(bsz, SCALAR_FEATURES)
    mask = torch.ones(bsz, ACTION_SPACE_SIZE, dtype=torch.bool)

    logits, value = model(spatial, scalar, mask)
    assert logits.shape == (bsz, ACTION_SPACE_SIZE)
    assert value.shape == (bsz,)

    # Value should be in [-1, 1] (tanh)
    assert (value >= -1.0).all() and (value <= 1.0).all()


def test_network_predict():
    """Predict returns probabilities that sum to ~1."""
    model = CRZeroNet(n_res_blocks=2, n_filters=32)
    model.eval()

    spatial = torch.randn(1, SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    scalar = torch.randn(1, SCALAR_FEATURES)
    mask = torch.ones(1, ACTION_SPACE_SIZE, dtype=torch.bool)

    policy, value = model.predict(spatial, scalar, mask)
    assert policy.shape == (1, ACTION_SPACE_SIZE)
    assert abs(policy.sum().item() - 1.0) < 1e-4


def test_network_masking():
    """Masked actions get near-zero probability."""
    model = CRZeroNet(n_res_blocks=2, n_filters=32)
    model.eval()

    spatial = torch.randn(1, SPATIAL_CHANNELS, ARENA_H, ARENA_W)
    scalar = torch.randn(1, SCALAR_FEATURES)

    # Only WAIT action is valid
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, ACTION_SPACE_SIZE - 1] = True

    policy, _ = model.predict(spatial, scalar, mask)
    # WAIT action should have ~100% probability
    assert policy[0, ACTION_SPACE_SIZE - 1].item() > 0.99


def test_network_param_count():
    """Full-size network has reasonable param count."""
    model = CRZeroNet(n_res_blocks=20, n_filters=256)
    total = sum(p.numel() for p in model.parameters())
    # Should be ~25–35M parameters (similar to AlphaZero)
    assert 10_000_000 < total < 100_000_000, f"Unexpected param count: {total:,}"
    print(f"Total parameters: {total:,}")


if __name__ == "__main__":
    test_encode_state_shape()
    test_encode_state_flipped()
    test_encode_batch()
    test_network_forward()
    test_network_predict()
    test_network_masking()
    test_network_param_count()
    print("All model tests passed!")
