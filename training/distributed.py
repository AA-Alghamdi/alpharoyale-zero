"""Distributed training orchestration using Ray.

Layout for 8× A100:
  - GPUs 0–5: Self-play workers (6 workers, each running games sequentially with MCTS)
  - GPUs 6–7: Training (DDP across 2 GPUs, or single GPU with larger batch)

The coordinator:
  1. Starts self-play workers and the trainer
  2. Periodically broadcasts new model weights from trainer → workers
  3. Monitors throughput and saves checkpoints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

from mcts.search import MCTSConfig
from model.network import CRZeroNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig, SelfPlayWorker
from training.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Top-level configuration for the full training run."""

    # --- Hardware ---
    n_self_play_gpus: int = 6
    n_training_gpus: int = 2  # DDP across these

    # --- Self-play ---
    games_per_worker_batch: int = 4  # games per run_batch call
    mcts_simulations: int = 800
    mcts_c_puct: float = 2.5
    dirichlet_alpha: float = 0.15
    temperature_threshold: int = 20
    random_decks: bool = True

    # --- Training ---
    batch_size: int = 2048
    lr: float = 2e-4
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    max_steps: int = 500_000
    min_buffer_before_train: int = 5_000

    # --- Replay buffer ---
    buffer_capacity: int = 500_000

    # --- Model ---
    n_res_blocks: int = 20
    n_filters: int = 256

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    weight_broadcast_interval: int = 500  # training steps between weight syncs
    checkpoint_interval: int = 5_000

    # --- Schedule ---
    # Phase transitions (in training steps)
    phase1_end: int = 50_000    # starter decks, 200 MCTS sims
    phase2_end: int = 200_000   # random decks, 400 MCTS sims
    # phase3: full MCTS (800 sims) until max_steps


def _self_play_process(
    worker_id: int,
    gpu_id: int,
    model_state_dict: dict,
    shared_buffer_arrays: dict,
    buffer_lock: mp.Lock,
    config: DistributedConfig,
    stop_event: mp.Event,
    weight_version: mp.Value,
    latest_weights: mp.Queue,
) -> None:
    """Self-play worker process (runs on a single GPU)."""
    device = torch.device(f"cuda:{gpu_id}")
    model = CRZeroNet(
        n_res_blocks=config.n_res_blocks,
        n_filters=config.n_filters,
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # Create a local replay buffer that writes into the shared memory
    replay_buffer = ReplayBuffer(capacity=config.buffer_capacity)
    # Point internal arrays to shared memory
    replay_buffer._spatial = shared_buffer_arrays["spatial"]
    replay_buffer._scalar = shared_buffer_arrays["scalar"]
    replay_buffer._policy = shared_buffer_arrays["policy"]
    replay_buffer._value = shared_buffer_arrays["value"]
    replay_buffer._lock = buffer_lock

    mcts_cfg = MCTSConfig(
        n_simulations=config.mcts_simulations,
        c_puct=config.mcts_c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
    )
    sp_cfg = SelfPlayConfig(
        n_games=config.games_per_worker_batch,
        mcts_config=mcts_cfg,
        temperature_threshold=config.temperature_threshold,
        random_decks=config.random_decks,
    )

    worker = SelfPlayWorker(
        model=model,
        replay_buffer=replay_buffer,
        config=sp_cfg,
        device=device,
        worker_id=worker_id,
    )

    current_version = 0
    while not stop_event.is_set():
        worker.run_batch()

        # Check for new weights
        try:
            while not latest_weights.empty():
                new_state = latest_weights.get_nowait()
                if new_state is not None:
                    worker.update_model(new_state)
                    current_version += 1
        except Exception:
            pass


class Coordinator:
    """Orchestrates the full distributed training run.

    For simplicity, this implementation uses a single-process approach with
    sequential self-play and training interleaved. For actual multi-GPU
    training, use the Ray-based launcher (scripts/train_distributed.py).
    """

    def __init__(self, config: DistributedConfig) -> None:
        self.config = config

    def run_single_gpu(self, device: torch.device | None = None) -> None:
        """Single-GPU training loop (for development/testing).

        Alternates between self-play and training steps.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Starting single-GPU training on %s", device)

        # Initialize model
        model = CRZeroNet(
            n_res_blocks=self.config.n_res_blocks,
            n_filters=self.config.n_filters,
        ).to(device)

        # Replay buffer
        replay_buffer = ReplayBuffer(capacity=self.config.buffer_capacity)

        # Self-play worker
        mcts_cfg = MCTSConfig(
            n_simulations=min(self.config.mcts_simulations, 100),  # reduced for dev
            c_puct=self.config.mcts_c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
        )
        sp_cfg = SelfPlayConfig(
            n_games=2,
            mcts_config=mcts_cfg,
            temperature_threshold=self.config.temperature_threshold,
            random_decks=self.config.random_decks,
        )
        worker = SelfPlayWorker(
            model=model,
            replay_buffer=replay_buffer,
            config=sp_cfg,
            device=device,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            replay_buffer=replay_buffer,
            device=device,
            lr=self.config.lr,
            lr_min=self.config.lr_min,
            weight_decay=self.config.weight_decay,
            batch_size=min(self.config.batch_size, 256),  # reduced for dev
            max_steps=self.config.max_steps,
            checkpoint_dir=self.config.checkpoint_dir,
            log_dir=self.config.log_dir,
        )

        # Main loop: alternate self-play and training
        games_per_cycle = 2
        train_steps_per_cycle = 10

        logger.info("Filling replay buffer with initial games...")
        while len(replay_buffer) < self.config.min_buffer_before_train:
            worker.run_batch(games_per_cycle)
            logger.info(
                "Buffer size: %d / %d",
                len(replay_buffer), self.config.min_buffer_before_train,
            )

        logger.info("Starting training loop")
        while trainer.step_count < self.config.max_steps:
            # Self-play phase
            worker.run_batch(games_per_cycle)

            # Training phase
            for _ in range(train_steps_per_cycle):
                trainer.train_step()

            # Checkpoint
            if trainer.step_count % self.config.checkpoint_interval == 0:
                trainer.save_checkpoint()

    def run_multi_gpu(self) -> None:
        """Multi-GPU training using Ray for orchestration.

        This is the production training mode for 8× A100s.
        """
        try:
            import ray
        except ImportError:
            logger.error("Ray is required for multi-GPU training: pip install 'ray[default]'")
            raise

        ray.init(num_gpus=self.config.n_self_play_gpus + self.config.n_training_gpus)

        # The Ray-based implementation would use Ray actors for:
        # 1. Self-play workers (remote actors on GPUs 0-5)
        # 2. Trainer (remote actor on GPUs 6-7)
        # 3. Replay buffer (shared via Ray object store)
        # 4. Parameter server (broadcasts weights)
        #
        # For now, we provide the single-GPU version above and the
        # multi-GPU script in scripts/train_distributed.py using Ray actors.

        logger.info(
            "Multi-GPU mode: %d self-play GPUs, %d training GPUs",
            self.config.n_self_play_gpus,
            self.config.n_training_gpus,
        )

        # This would be implemented with Ray actors - see scripts/train_distributed.py
        raise NotImplementedError(
            "Use scripts/train_distributed.py for multi-GPU training with Ray"
        )
