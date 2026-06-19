#!/usr/bin/env python3
"""Distributed training with Ray — production mode for 8× A100.

Architecture:
  - 6 Ray actors as self-play workers (GPUs 0–5)
  - 1 Ray actor as trainer (GPU 6, or DDP across 6–7)
  - 1 Ray actor as parameter server (CPU)
  - Shared replay buffer via Ray object store

Usage:
    # On a machine with 8 A100s:
    ray start --head --num-gpus=8
    python scripts/train_distributed.py

    # Or let Ray auto-detect:
    python scripts/train_distributed.py --auto-ray
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("distributed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-ray", action="store_true",
                        help="Auto-initialize Ray (otherwise expect an existing cluster)")
    parser.add_argument("--n-sp-workers", type=int, default=6)
    parser.add_argument("--n-train-gpus", type=int, default=2)
    parser.add_argument("--mcts-sims", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--buffer-capacity", type=int, default=500_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-interval", type=int, default=10_000,
                        help="Evaluate model every N training steps")
    parser.add_argument("--eval-games", type=int, default=50)
    args = parser.parse_args()

    try:
        import ray
    except ImportError:
        logger.error("Ray is required: pip install 'ray[default]'")
        sys.exit(1)

    if args.auto_ray:
        ray.init(num_gpus=args.n_sp_workers + args.n_train_gpus)
    else:
        ray.init(address="auto")

    logger.info("Ray cluster: %s", ray.cluster_resources())

    from eval.evaluator import Evaluator
    from mcts.search import MCTSConfig
    from model.network import CRZeroNet
    from training.replay_buffer import ReplayBuffer, ReplayEntry
    from training.self_play import SelfPlayConfig, SelfPlayWorker
    from training.trainer import Trainer

    # --- Ray Actors ---

    @ray.remote(num_gpus=1)
    class SelfPlayActor:
        """Remote self-play worker."""

        def __init__(self, worker_id: int, model_state: dict, mcts_sims: int):
            self.device = torch.device("cuda")
            self.model = CRZeroNet().to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()
            self.buffer = ReplayBuffer(capacity=50_000)
            mcts_cfg = MCTSConfig(n_simulations=mcts_sims)
            sp_cfg = SelfPlayConfig(n_games=2, mcts_config=mcts_cfg)
            self.worker = SelfPlayWorker(
                model=self.model,
                replay_buffer=self.buffer,
                config=sp_cfg,
                device=self.device,
                worker_id=worker_id,
            )
            self.games_played = 0

        def generate_games(self, n_games: int = 2) -> list[dict]:
            """Play games and return serialized experience."""
            old_size = len(self.buffer)
            self.worker.run_batch(n_games)
            self.games_played += n_games

            # Extract new entries
            new_size = len(self.buffer)
            entries = []
            for i in range(old_size, new_size):
                idx = i % self.buffer.capacity
                entries.append({
                    "spatial": self.buffer._spatial[idx].copy(),
                    "scalar": self.buffer._scalar[idx].copy(),
                    "policy": self.buffer._policy[idx].copy(),
                    "value": float(self.buffer._value[idx]),
                })
            return entries

        def update_weights(self, state_dict: dict) -> None:
            self.model.load_state_dict(state_dict)
            self.model.eval()

        def get_stats(self) -> dict:
            return {"games_played": self.games_played}

    @ray.remote(num_gpus=args.n_train_gpus)
    class TrainerActor:
        """Remote trainer."""

        def __init__(self, model_state: dict, batch_size: int, max_steps: int, checkpoint_dir: str):
            self.device = torch.device("cuda")
            self.model = CRZeroNet().to(self.device)
            self.model.load_state_dict(model_state)
            self.buffer = ReplayBuffer(capacity=args.buffer_capacity)
            self.trainer = Trainer(
                model=self.model,
                replay_buffer=self.buffer,
                device=self.device,
                batch_size=batch_size,
                max_steps=max_steps,
                checkpoint_dir=checkpoint_dir,
            )

        def add_experience(self, entries: list[dict]) -> None:
            for e in entries:
                self.buffer.push(ReplayEntry(
                    spatial=e["spatial"],
                    scalar=e["scalar"],
                    policy=e["policy"],
                    value=e["value"],
                ))

        def train_steps(self, n_steps: int) -> dict:
            metrics = {}
            for _ in range(n_steps):
                m = self.trainer.train_step()
                if m:
                    metrics = m
            return metrics

        def get_weights(self) -> dict:
            return self.trainer.get_model_state()

        def get_step(self) -> int:
            return self.trainer.step_count

        def save_checkpoint(self, tag: str | None = None) -> str:
            return self.trainer.save_checkpoint(tag)

        def buffer_size(self) -> int:
            return len(self.buffer)

    # --- Initialize ---

    init_model = CRZeroNet()
    init_state = init_model.state_dict()

    # Start actors
    sp_actors = [
        SelfPlayActor.remote(i, init_state, args.mcts_sims)
        for i in range(args.n_sp_workers)
    ]
    trainer_actor = TrainerActor.remote(
        init_state, args.batch_size, args.max_steps, args.checkpoint_dir
    )

    # Best model for evaluation
    best_state = {k: v.clone() for k, v in init_state.items()}
    evaluator = Evaluator(mcts_simulations=100)

    logger.info("Started %d self-play workers and 1 trainer", args.n_sp_workers)

    # --- Main Loop ---

    train_step = 0
    last_eval_step = 0

    while train_step < args.max_steps:
        # 1. Self-play: generate games in parallel
        game_futures = [actor.generate_games.remote(2) for actor in sp_actors]
        all_entries = ray.get(game_futures)

        # 2. Feed experience to trainer
        for entries in all_entries:
            if entries:
                ray.get(trainer_actor.add_experience.remote(entries))

        # 3. Train
        buf_size = ray.get(trainer_actor.buffer_size.remote())
        if buf_size >= 5000:
            metrics = ray.get(trainer_actor.train_steps.remote(50))
            train_step = ray.get(trainer_actor.get_step.remote())

            if train_step % 1000 == 0:
                logger.info(
                    "Step %d | buffer=%d | metrics=%s",
                    train_step, buf_size, metrics,
                )

        # 4. Broadcast weights to workers
        if train_step % 500 == 0 and train_step > 0:
            new_weights = ray.get(trainer_actor.get_weights.remote())
            ray.get([actor.update_weights.remote(new_weights) for actor in sp_actors])

        # 5. Periodic evaluation
        if train_step - last_eval_step >= args.eval_interval and train_step > 0:
            new_weights = ray.get(trainer_actor.get_weights.remote())
            new_model = CRZeroNet()
            new_model.load_state_dict(new_weights)

            best_model = CRZeroNet()
            best_model.load_state_dict(best_state)

            promoted = evaluator.evaluate_checkpoint(
                new_model, best_model, train_step, n_games=args.eval_games
            )
            if promoted:
                best_state = {k: v.clone() for k, v in new_weights.items()}
                ray.get(trainer_actor.save_checkpoint.remote(f"best_step_{train_step}"))

            last_eval_step = train_step

        # 6. Periodic checkpoint
        if train_step % 5000 == 0 and train_step > 0:
            ray.get(trainer_actor.save_checkpoint.remote())

    # Final save
    ray.get(trainer_actor.save_checkpoint.remote("final"))
    logger.info("Training complete at step %d", train_step)
    ray.shutdown()


if __name__ == "__main__":
    main()
