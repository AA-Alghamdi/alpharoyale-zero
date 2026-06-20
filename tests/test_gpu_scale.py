"""GPU-scale training path: launch-readiness tests (run on CPU, tiny scale).

These guard the primitives that make large-scale self-play training tractable:

  * batched NN evaluation is numerically identical to per-state evaluation,
  * batched Gumbel search returns well-formed, action-masked policies,
  * the vectorized self-play worker produces the *same* full ``ReplayEntry``
    format (entity features + auxiliary targets) as the sequential worker,
  * the replay buffer round-trips through ``save``/``load`` (warm-start) and
    ``drain_all`` is exhaustive (the distributed collector relies on this).

Everything is kept tiny so the whole file runs on CPU in well under a minute.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crsim.game import Action, CRGame
from mcts.batched_gumbel import BatchedGumbelSearch
from mcts.gumbel_search import GumbelConfig, GumbelMuZeroSearch
from model.batched_eval import evaluate_games
from model.transformer_net import CRStarNet
from training.replay_buffer import ReplayBuffer, ReplayEntry
from training.self_play_v2 import SelfPlayV2Config
from training.vectorized_self_play import VectorizedSelfPlayWorker

DEVICE = torch.device("cpu")


def _tiny_model() -> CRStarNet:
    model = CRStarNet(
        spatial_blocks=1, spatial_filters=16, core_hidden=32, core_layers=1
    )
    model.eval()
    return model


def _stepped_games(n: int, ticks: int = 20) -> list[CRGame]:
    """A few games advanced a little so they contain live units."""
    games = [CRGame(seed=i) for i in range(n)]
    for g in games:
        for _ in range(ticks):
            g.step([Action(0, -1), Action(1, -1)])
    return games


def _tiny_sp_config() -> SelfPlayV2Config:
    return SelfPlayV2Config(
        gumbel_config=GumbelConfig(
            n_simulations=4,
            max_considered_actions=4,
            playout_cap_randomization=False,
        ),
        decision_interval_ticks=25,
        max_ticks=400,
        augment_flip=False,
    )


def test_batched_eval_matches_sequential() -> None:
    """One batched forward pass == N sequential ``_evaluate`` calls."""
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    np.random.seed(0)

    model = _tiny_model()
    games = _stepped_games(5)
    requests = [(g, p) for g in games for p in (0, 1)]

    seq = GumbelMuZeroSearch(model=model, config=GumbelConfig(), device=DEVICE)
    batched = evaluate_games(model, requests, DEVICE)

    assert len(batched) == len(requests)
    for (g, p), (pol_b, val_b) in zip(requests, batched, strict=True):
        pol_s, val_s = seq._evaluate(g, p)
        assert np.allclose(pol_b, pol_s, atol=1e-5)
        assert abs(val_b - val_s) < 1e-4


def test_batched_gumbel_returns_valid_policies() -> None:
    """``search_many`` returns one normalized, action-masked policy per input."""
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    np.random.seed(0)

    model = _tiny_model()
    games = _stepped_games(4)
    cfg = GumbelConfig(
        n_simulations=4, max_considered_actions=4, playout_cap_randomization=False
    )
    search = BatchedGumbelSearch(model=model, config=cfg, device=DEVICE)

    instances = [(g, 0) for g in games] + [(g, 1) for g in games]
    policies = search.search_many(instances)

    assert len(policies) == len(instances)
    for (game, player), policy in zip(instances, policies, strict=True):
        assert policy.shape[0] > 0
        assert abs(float(policy.sum()) - 1.0) < 1e-4
        assert np.all(policy >= 0.0)
        # Probability mass only on legal actions.
        mask = game.get_valid_actions_mask(player)
        assert not np.any(policy[~mask] > 1e-6)


def test_batched_gumbel_empty_input() -> None:
    model = _tiny_model()
    search = BatchedGumbelSearch(
        model=model, config=GumbelConfig(n_simulations=2), device=DEVICE
    )
    assert search.search_many([]) == []


def _make_entry(i: int) -> ReplayEntry:
    rng = np.random.default_rng(i)
    from crsim.constants import (
        ACTION_SPACE_SIZE,
        ARENA_H,
        ARENA_W,
        SCALAR_FEATURES,
        SPATIAL_CHANNELS,
    )
    from training.replay_buffer import ENTITY_DIM, ENTITY_SLOTS

    policy = rng.random(ACTION_SPACE_SIZE).astype(np.float32)
    policy /= policy.sum()
    return ReplayEntry(
        spatial=rng.random((SPATIAL_CHANNELS, ARENA_H, ARENA_W)).astype(np.float32),
        scalar=rng.random(SCALAR_FEATURES).astype(np.float32),
        policy=policy,
        value=float(rng.uniform(-1, 1)),
        entity_features=rng.random((ENTITY_SLOTS, ENTITY_DIM)).astype(np.float32),
        entity_mask=(rng.random(ENTITY_SLOTS) > 0.5),
        crown_target=int(rng.integers(0, 7)),
        tower_hp_target=rng.random(6).astype(np.float32),
        game_length_target=float(rng.uniform(0, 1)),
    )


def test_replay_buffer_save_load_roundtrip(tmp_path: Path) -> None:
    """``save`` then ``load`` preserves every field in chronological order."""
    buf = ReplayBuffer(max_size=100)
    entries = [_make_entry(i) for i in range(10)]
    for e in entries:
        buf.push(e)

    path = str(tmp_path / "buf.npz")
    buf.save(path)

    fresh = ReplayBuffer(max_size=100)
    n = fresh.load(path)
    assert n == 10
    assert len(fresh) == 10

    a = buf.drain_all()
    b = fresh.drain_all()
    assert set(a.keys()) == set(b.keys())
    # float16 storage means a small tolerance on the big arrays.
    assert np.allclose(a["spatial"], b["spatial"], atol=1e-2)
    assert np.allclose(a["scalar"], b["scalar"], atol=1e-5)
    assert np.allclose(a["policy"], b["policy"], atol=1e-5)
    assert np.allclose(a["value"], b["value"], atol=1e-5)
    assert np.array_equal(a["entity_mask"], b["entity_mask"])
    assert np.array_equal(a["crown_target"], b["crown_target"])
    assert np.allclose(a["game_length_target"], b["game_length_target"], atol=1e-5)


def test_replay_buffer_save_load_empty(tmp_path: Path) -> None:
    buf = ReplayBuffer(max_size=10)
    path = str(tmp_path / "empty.npz")
    buf.save(path)
    fresh = ReplayBuffer(max_size=10)
    assert fresh.load(path) == 0


def test_replay_buffer_load_respects_capacity(tmp_path: Path) -> None:
    """Loading more positions than capacity keeps the most recent ones."""
    big = ReplayBuffer(max_size=100)
    for i in range(20):
        big.push(_make_entry(i))
    path = str(tmp_path / "buf.npz")
    big.save(path)

    small = ReplayBuffer(max_size=5)
    n = small.load(path)
    assert n == 5
    assert len(small) == 5


def test_replay_buffer_drain_all_is_exhaustive() -> None:
    """``drain_all`` returns every position once and empties the buffer."""
    buf = ReplayBuffer(max_size=100)
    for i in range(8):
        buf.push(_make_entry(i))
    drained = buf.drain_all()
    assert len(drained["value"]) == 8
    assert len(buf) == 0
    # A second drain on the empty buffer is a no-op.
    assert buf.drain_all() == {}


def test_vectorized_worker_produces_full_entries() -> None:
    """The vectorized worker fills the buffer with entity + aux signal."""
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    np.random.seed(0)

    model = _tiny_model()
    buf = ReplayBuffer(max_size=5000)
    worker = VectorizedSelfPlayWorker(
        model=model,
        replay_buffer=buf,
        config=_tiny_sp_config(),
        device=DEVICE,
        n_envs=4,
    )
    n_positions = worker.run_batch(n_games=4)
    assert n_positions > 0
    assert len(buf) == n_positions

    batch = buf.sample_full(min(8, len(buf)))
    # Entity features and aux targets must be present and non-trivial.
    assert batch["entity_features"].shape[1:] == (64, 40)
    assert batch["entity_mask"].shape[1] == 64
    assert batch["tower_hp_target"].shape[1] == 6
    # Policies are valid distributions.
    assert np.allclose(batch["policy"].sum(axis=1), 1.0, atol=1e-3)
