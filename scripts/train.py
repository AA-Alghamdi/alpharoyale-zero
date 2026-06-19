#!/usr/bin/env python3
"""Main training script for ClashRoyale-Zero.

Usage:
    # Single GPU (development / testing)
    python scripts/train.py --mode single --device cuda:0

    # Multi-GPU with Ray (production, 8× A100)
    python scripts/train.py --mode multi
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from training.distributed import Coordinator, DistributedConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClashRoyale-Zero Training")

    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="Training mode: single GPU or multi-GPU with Ray")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for single-GPU mode")

    # Model
    parser.add_argument("--n-res-blocks", type=int, default=20)
    parser.add_argument("--n-filters", type=int, default=256)

    # Training
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Self-play
    parser.add_argument("--mcts-sims", type=int, default=800)
    parser.add_argument("--games-per-batch", type=int, default=4)
    parser.add_argument("--random-decks", action="store_true", default=True)

    # Buffer
    parser.add_argument("--buffer-capacity", type=int, default=500_000)
    parser.add_argument("--min-buffer", type=int, default=5_000)

    # Multi-GPU
    parser.add_argument("--n-sp-gpus", type=int, default=6,
                        help="Number of self-play GPUs")
    parser.add_argument("--n-train-gpus", type=int, default=2,
                        help="Number of training GPUs")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = DistributedConfig(
        n_self_play_gpus=args.n_sp_gpus,
        n_training_gpus=args.n_train_gpus,
        games_per_worker_batch=args.games_per_batch,
        mcts_simulations=args.mcts_sims,
        random_decks=args.random_decks,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=1e-5,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        min_buffer_before_train=args.min_buffer,
        buffer_capacity=args.buffer_capacity,
        n_res_blocks=args.n_res_blocks,
        n_filters=args.n_filters,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    coordinator = Coordinator(config)

    if args.mode == "single":
        device = torch.device(args.device)
        logger.info("Starting single-GPU training on %s", device)
        coordinator.run_single_gpu(device=device)
    else:
        logger.info("Starting multi-GPU training (%d SP + %d train GPUs)",
                     args.n_sp_gpus, args.n_train_gpus)
        coordinator.run_multi_gpu()


if __name__ == "__main__":
    main()
