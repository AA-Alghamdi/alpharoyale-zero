"""Neural network trainer with AMP and optional DDP.

Handles:
  - Mini-batch training from the replay buffer
  - Mixed-precision (FP16) training for A100 throughput
  - Learning rate scheduling (cosine decay)
  - Gradient clipping
  - TensorBoard logging
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as f_nn  # noqa: N812
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.network import CRZeroNet
from training.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class Trainer:
    """Trains the CRZeroNet from replay buffer data."""

    def __init__(
        self,
        model: CRZeroNet,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        lr: float = 2e-4,
        lr_min: float = 1e-5,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        max_steps: int = 500_000,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        value_loss_weight: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.replay_buffer = replay_buffer
        self.device = device
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.value_loss_weight = value_loss_weight

        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr_min
        )
        self.scaler = GradScaler()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.step_count: int = 0
        self._tb_writer = None

        # Try to import tensorboard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logger.warning("TensorBoard not available; logging to stdout only")

    def train_step(self) -> dict[str, float]:
        """Run one training step (sample batch → forward → backward → update).

        Returns dict of loss components for logging.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}  # not enough data yet

        self.model.train()

        # Sample
        spatial_np, scalar_np, policy_np, value_np = self.replay_buffer.sample(
            self.batch_size
        )

        spatial = torch.from_numpy(spatial_np).to(self.device)
        scalar = torch.from_numpy(scalar_np).to(self.device)
        target_policy = torch.from_numpy(policy_np).to(self.device)
        target_value = torch.from_numpy(value_np).to(self.device)

        # Forward
        with autocast():
            pred_logits, pred_value = self.model(spatial, scalar)

            # Policy loss: cross-entropy with MCTS target distribution
            log_probs = f_nn.log_softmax(pred_logits, dim=-1)
            policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

            # Value loss: MSE
            value_loss = f_nn.mse_loss(pred_value, target_value)

            loss = policy_loss + self.value_loss_weight * value_loss

        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.step_count += 1

        # Logging
        metrics = {
            "loss/total": float(loss),
            "loss/policy": float(policy_loss),
            "loss/value": float(value_loss),
            "lr": self.optimizer.param_groups[0]["lr"],
            "buffer_size": len(self.replay_buffer),
        }

        if self._tb_writer is not None and self.step_count % 100 == 0:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, self.step_count)

        if self.step_count % 500 == 0:
            logger.info(
                "Step %d | loss=%.4f (policy=%.4f value=%.4f) | lr=%.2e | buf=%d",
                self.step_count,
                metrics["loss/total"],
                metrics["loss/policy"],
                metrics["loss/value"],
                metrics["lr"],
                metrics["buffer_size"],
            )

        return metrics

    def train_loop(
        self,
        min_buffer_size: int = 10_000,
        steps_between_checkpoints: int = 5_000,
    ) -> None:
        """Run the training loop until max_steps."""
        logger.info("Waiting for replay buffer to fill (min=%d)...", min_buffer_size)
        while len(self.replay_buffer) < min_buffer_size:
            time.sleep(1.0)

        logger.info("Starting training loop (max_steps=%d)", self.max_steps)

        while self.step_count < self.max_steps:
            self.train_step()

            if self.step_count % steps_between_checkpoints == 0:
                self.save_checkpoint()

        self.save_checkpoint(tag="final")
        logger.info("Training complete: %d steps", self.step_count)

    def save_checkpoint(self, tag: str | None = None) -> str:
        """Save model checkpoint. Returns the path."""
        name = tag or f"step_{self.step_count:07d}"
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "step": self.step_count,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
        return str(path)

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.step_count = ckpt["step"]
        logger.info("Resumed from checkpoint: %s (step %d)", path, self.step_count)

    def get_model_state(self) -> dict:
        """Return the current model state dict (for broadcasting to workers)."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
