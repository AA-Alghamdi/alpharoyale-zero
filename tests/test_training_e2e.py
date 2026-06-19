"""End-to-end training smoke test.

Guards the full pipeline that previously never actually ran to completion:
    self-play (decision-interval Gumbel search) -> replay buffer ->
    TrainerV2 gradient steps -> checkpoint save/load.

Kept intentionally small (tiny model, truncated games, few steps) so it runs on
CPU in well under a minute, but it exercises every real component.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcts.gumbel_search import GumbelConfig
from model.transformer_net import CRStarNet
from training.replay_buffer import ReplayBuffer
from training.self_play_v2 import SelfPlayV2Config, SelfPlayWorkerV2
from training.trainer_v2 import TrainerV2


def _tiny_model() -> CRStarNet:
    return CRStarNet(
        spatial_blocks=1, spatial_filters=16, core_hidden=32, core_layers=1
    )


def test_self_play_game_is_tractable() -> None:
    """A self-play game must finish quickly with the decision interval."""
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    np.random.seed(0)

    model = _tiny_model()
    buf = ReplayBuffer(max_size=1000)
    cfg = SelfPlayV2Config(
        gumbel_config=GumbelConfig(n_simulations=2, max_considered_actions=2),
        decision_interval_ticks=25,
        max_ticks=400,
        augment_flip=False,
    )
    worker = SelfPlayWorkerV2(
        model=model, replay_buffer=buf, config=cfg, device=torch.device("cpu")
    )
    entries = worker.play_game()
    assert len(entries) > 0


def test_training_pipeline_end_to_end(tmp_path: Path) -> None:
    """Self-play -> buffer -> train -> checkpoint must work and reduce loss."""
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")

    model = _tiny_model()
    buf = ReplayBuffer(max_size=2000)
    cfg = SelfPlayV2Config(
        gumbel_config=GumbelConfig(n_simulations=2, max_considered_actions=2),
        decision_interval_ticks=25,
        max_ticks=400,
        augment_flip=True,
    )
    worker = SelfPlayWorkerV2(
        model=model, replay_buffer=buf, config=cfg, device=device
    )

    while len(buf) < 64:
        worker.run_batch(n_games=1)

    trainer = TrainerV2(
        model=model,
        replay_buffer=buf,
        device=device,
        batch_size=32,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "runs"),
    )

    losses: list[float] = []
    for _ in range(30):
        metrics = trainer.train_step()
        if metrics:
            losses.append(metrics["loss/total"])

    assert len(losses) >= 10
    assert all(np.isfinite(losses)), "loss became NaN/Inf"
    # Loss should trend down over the run.
    assert np.mean(losses[-5:]) < np.mean(losses[:5])

    # Checkpoint round-trip.
    path = trainer.save_checkpoint(tag="e2e")
    fresh = _tiny_model()
    trainer2 = TrainerV2(
        model=fresh,
        replay_buffer=buf,
        device=device,
        batch_size=32,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    trainer2.load_checkpoint(path)
    assert trainer2.step_count == trainer.step_count
