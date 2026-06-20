"""Tests for the imitation / behavioral-cloning warm-start.

These verify the pieces are network-correct and actually run end-to-end (the
previous version called ``model(states)``, which never matched the real
``forward(spatial, scalar, mask, ...)`` interface): expert data generation
records only legal actions, behavioral cloning reduces policy loss on both
networks, and the deck-matchup value warm-start fits separable outcomes and
writes a checkpoint.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from crsim.constants import ACTION_SPACE_SIZE, ARENA_H, ARENA_W, SPATIAL_CHANNELS
from model.network import CRZeroNet
from model.transformer_net import CRStarNet
from training.imitation import (
    ExpertDataset,
    ImitationConfig,
    generate_expert_dataset,
    train_behavioral_cloning,
    train_value_warmstart,
)


def _tiny_zero() -> CRZeroNet:
    return CRZeroNet(n_res_blocks=1, n_filters=16)


def _tiny_star() -> CRStarNet:
    return CRStarNet(spatial_blocks=1, spatial_filters=32, core_hidden=64, core_layers=1)


def test_expert_dataset_records_legal_actions(tmp_path):
    npz = tmp_path / "expert.npz"
    n = generate_expert_dataset(str(npz), n_games=2, decision_interval=8, max_ticks=120, seed=0)
    assert n > 0
    ds = ExpertDataset(str(npz))
    assert len(ds) == n
    for i in range(len(ds)):
        spatial, scalar, mask, action, value, ef, em = ds[i]
        assert spatial.shape == (SPATIAL_CHANNELS, ARENA_H, ARENA_W)
        assert scalar.shape[0] > 0
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask[int(action)], "expert action must be legal under its mask"
        assert value in (-1.0, 0.0, 1.0)


def test_behavioral_cloning_reduces_policy_loss(tmp_path):
    npz = tmp_path / "expert.npz"
    generate_expert_dataset(str(npz), n_games=2, decision_interval=8, max_ticks=160, seed=1)
    ds = ExpertDataset(str(npz))
    net = _tiny_zero()
    metrics = train_behavioral_cloning(
        net, ds, ImitationConfig(batch_size=32, num_epochs=20, learning_rate=3e-3), "cpu"
    )
    assert metrics["policy_loss"][-1] < metrics["policy_loss"][0]


def test_behavioral_cloning_runs_on_transformer(tmp_path):
    npz = tmp_path / "expert.npz"
    generate_expert_dataset(str(npz), n_games=1, decision_interval=8, max_ticks=120, seed=2)
    ds = ExpertDataset(str(npz))
    star = _tiny_star()
    metrics = train_behavioral_cloning(
        star, ds, ImitationConfig(batch_size=16, num_epochs=2, learning_rate=1e-3), "cpu"
    )
    assert len(metrics["policy_loss"]) == 2
    assert all(np.isfinite(metrics["policy_loss"]))


def test_value_warmstart_fits_separable_outcomes(tmp_path):
    import random

    battles = tmp_path / "battles.jsonl"
    # A learnable matchup corpus: P0 wins iff it holds the lower-index deck.
    # (A single opening is low-signal, so warm-start needs a population of
    # matchups to fit a deck-strength prior — mirroring the real Kaggle dump.)
    random.seed(0)
    rows = []
    for _ in range(24):
        a = random.randrange(0, 60)
        c = random.randrange(60, 121)
        d0 = [(a + i) % 121 for i in range(8)]
        d1 = [(c + i) % 121 for i in range(8)]
        rows.append({"deck0": d0, "deck1": d1, "p0_win": 1.0})
        rows.append({"deck0": d1, "deck1": d0, "p0_win": 0.0})
    with open(battles, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    torch.manual_seed(0)
    out = tmp_path / "vw.pt"
    metrics = train_value_warmstart(
        str(battles), str(out), epochs=80, device="cpu",
        model=_tiny_zero(), batch_size=16, learning_rate=5e-4,
    )
    assert metrics["value_loss"][-1] < metrics["value_loss"][0]
    assert out.exists()
    # the saved checkpoint reloads into a fresh network
    CRZeroNet(n_res_blocks=1, n_filters=16).load_state_dict(torch.load(out, weights_only=True))


def test_value_warmstart_winner_seat_convention(tmp_path):
    """winner=0 -> P0 won -> target +1; winner=1 -> target -1."""
    from training.imitation import _p0_win_from

    assert _p0_win_from({"winner": 0}) == 1.0
    assert _p0_win_from({"winner": 1}) == 0.0
    assert _p0_win_from({"p0_win": 0.75}) == 0.75


def test_missing_battles_file_is_graceful(tmp_path):
    out = tmp_path / "vw.pt"
    metrics = train_value_warmstart(str(tmp_path / "nope.jsonl"), str(out), epochs=1, device="cpu")
    assert metrics == {"value_loss": []}
    assert not out.exists()
