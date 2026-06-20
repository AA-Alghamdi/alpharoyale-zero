"""Tests for vectorized, GPU-batched self-play.

The core correctness claim is that batching the network evaluation across many
concurrent games is numerically identical to evaluating each position on its own
(``batched_eval`` == per-sample forward), so the throughput path produces the
same data the per-game path would. The rest verify trajectory validity, the
terminal-value sign, backend selection, replay-buffer compatibility, and
determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from crsim.constants import ACTION_SPACE_SIZE, ARENA_H, ARENA_W, SPATIAL_CHANNELS
from crsim.game import CRGame
from model.network import CRZeroNet
from training.replay_buffer import ReplayBuffer
from training.vectorized_selfplay import (
    VectorizedSelfPlay,
    VectorizedSelfPlayConfig,
    make_game,
    rust_backend_available,
)


def _tiny_net() -> CRZeroNet:
    net = CRZeroNet(n_res_blocks=1, n_filters=16)
    net.eval()
    return net


def _runner(net, **cfg_kwargs) -> VectorizedSelfPlay:
    cfg = VectorizedSelfPlayConfig(**cfg_kwargs)
    return VectorizedSelfPlay(net, cfg, device="cpu", seed=1)


def test_batched_eval_matches_per_sample():
    """One batched forward == stacking N independent single forwards."""
    net = _tiny_net()
    runner = _runner(net, n_envs=6)
    # Six distinct positions: advance fresh games a few ticks so states differ.
    games = [CRGame(seed=s) for s in range(6)]
    for k, g in enumerate(games):
        from crsim.game import Action

        for _ in range(k):
            g.step([Action(0, -1), Action(1, -1)])
    items = [(g, p) for g, p in zip(games, [0, 1, 0, 1, 0, 1])]

    batched_policies, batched_values = runner.batched_eval(items)

    for i, (game, player) in enumerate(items):
        single_policies, single_values = runner.batched_eval([(game, player)])
        assert np.allclose(batched_policies[i], single_policies[0], atol=1e-5)
        assert np.allclose(batched_values[i], single_values[0], atol=1e-5)


def test_generate_produces_valid_entries():
    net = _tiny_net()
    runner = _runner(net, n_envs=4, decision_interval_ticks=8, max_ticks=48)
    entries = runner.generate(4)

    assert entries, "expected at least one replay entry"
    for e in entries:
        assert e.spatial.shape == (SPATIAL_CHANNELS, ARENA_H, ARENA_W)
        assert e.policy.shape == (ACTION_SPACE_SIZE,)
        assert e.value in (-1.0, 0.0, 1.0)
        assert np.isfinite(e.policy).all()
        assert abs(float(e.policy.sum()) - 1.0) < 1e-3


def test_terminal_value_sign_matches_winner():
    """Winner's positions get +1, loser's get -1 (Meta sweeps Wait -> P0 wins)."""
    from crsim.actions import action_id_to_action
    from crsim.game import Action
    from eval.baseline_agents import MetaAgent, WaitAgent
    from model.features import encode_state, extract_entity_features

    game = CRGame(seed=2)
    a0, a1 = MetaAgent(), WaitAgent()
    steps = 0
    while not game.done and steps < 8000:
        if steps % 8 == 0:
            act0 = action_id_to_action(a0.select_action(game, 0), 0)
            act1 = action_id_to_action(a1.select_action(game, 1), 1)
            game.step([act0, act1])
        else:
            game.step([Action(0, -1), Action(1, -1)])
        steps += 1
    assert game.done and game.get_reward(0) == 1.0

    # Build a one-step-per-player trajectory and finalize it.
    trajectory = []
    for p in (0, 1):
        spatial, scalar = encode_state(game, p)
        ent_feats, ent_mask = extract_entity_features(game, p)
        trajectory.append(
            {
                "spatial": spatial,
                "scalar": scalar,
                "entity_features": ent_feats,
                "entity_mask": ent_mask,
                "policy": np.zeros(ACTION_SPACE_SIZE, dtype=np.float32),
                "player": p,
            }
        )

    runner = _runner(_tiny_net(), augment_flip=False)
    entries = runner._finalize(game, trajectory)
    by_player = {e.value for e in entries[:2]}
    assert entries[0].value == 1.0  # player 0 (winner)
    assert entries[1].value == -1.0  # player 1 (loser)
    assert by_player == {1.0, -1.0}


def test_make_game_backend_selection():
    assert isinstance(make_game("python"), CRGame)
    with pytest.raises(ValueError):
        make_game("nonsense")

    if rust_backend_available():
        from crsim.game import Action
        from crsim.rust_adapter import CRGameRust

        g = make_game("rust", seed=0)
        assert isinstance(g, CRGameRust)
        g.step([Action(0, -1), Action(1, -1)])  # runs without error
    else:
        # Documented: the Rust path is opt-in and absent when unbuilt.
        assert not rust_backend_available()


def test_entries_push_to_replay_buffer():
    net = _tiny_net()
    runner = _runner(net, n_envs=2, max_ticks=32)
    entries = runner.generate(2)
    buf = ReplayBuffer(capacity=10_000)
    for e in entries:
        buf.push(e)
    assert len(buf) == len(entries)
    batch = buf.sample(min(4, len(buf)))
    assert len(batch) == min(4, len(entries))


def test_generation_is_deterministic_under_seed():
    net = _tiny_net()
    a = _runner(net, n_envs=3, max_ticks=40).generate(3)
    b = _runner(net, n_envs=3, max_ticks=40).generate(3)
    assert len(a) == len(b)
    assert [e.value for e in a] == [e.value for e in b]
    assert np.allclose(a[0].spatial, b[0].spatial)


def test_parallel_trainer_wrapper_collects_entries():
    from training.parallel_sim import ParallelSimConfig, ParallelTrainer

    net = _tiny_net()
    cfg = ParallelSimConfig(batch_size=2, max_ticks=32)
    trainer = ParallelTrainer(net, cfg, device="cpu")
    entries = trainer.collect_games(2)
    assert entries
    assert trainer.stats["total_games_completed"] == 2
    assert cfg.to_vectorized().n_envs == 2
