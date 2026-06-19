"""Imitation Learning warm-start from human data.

Instead of starting self-play from random weights (which takes days
to learn basic CR strategy), we warm-start from human data:

1. Download Kaggle 37.9M matches — extract value estimates per matchup
2. Use KataCR replay dataset — extract state->action pairs
3. Train supervised policy for 1-2 hours
4. Switch to self-play from a competent baseline

Both AlphaGo and AlphaStar used supervised warm-start before self-play.

Data sources:
  - Kaggle: kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset
  - KataCR: github.com/wty-yy/Clash-Royale-Replay-Dataset
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class ImitationConfig:
    """Configuration for imitation learning warm-start."""

    data_dir: str = "data/imitation"
    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 10
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    max_samples: int | None = None  # limit for debugging


class MatchupDataset(Dataset):
    """Dataset of deck matchup outcomes from Kaggle data.

    Each sample: (deck_0_cards, deck_1_cards, winner)
    Used to warm-start the value network:
      V(state) ≈ P(deck_0 wins | decks, initial state)
    """

    def __init__(self, csv_path: str, num_card_types: int = 121, max_samples: int | None = None) -> None:
        self.data: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.num_card_types = num_card_types
        self._load(csv_path, max_samples)

    def _load(self, csv_path: str, max_samples: int | None) -> None:
        """Load matchup data from CSV.

        Expected columns: player0_card0..7, player1_card0..7, winner
        """
        if not os.path.exists(csv_path):
            logger.warning(f"Matchup data not found at {csv_path}")
            return

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if max_samples and count >= max_samples:
                    break

                # Encode decks as multi-hot vectors
                deck0 = np.zeros(self.num_card_types, dtype=np.float32)
                deck1 = np.zeros(self.num_card_types, dtype=np.float32)

                for i in range(8):
                    c0 = int(row.get(f"player0_card{i}", 0))
                    c1 = int(row.get(f"player1_card{i}", 0))
                    if 0 <= c0 < self.num_card_types:
                        deck0[c0] = 1.0
                    if 0 <= c1 < self.num_card_types:
                        deck1[c1] = 1.0

                winner = float(row.get("winner", 0.5))
                self.data.append((deck0, deck1, winner))
                count += 1

        logger.info(f"Loaded {len(self.data)} matchup samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        deck0, deck1, winner = self.data[idx]
        return deck0, deck1, np.float32(winner)


class ReplayDataset(Dataset):
    """Dataset of expert replay trajectories for policy warm-start.

    Each sample: (state, action, value)
    - state: game state encoding at decision point
    - action: expert's chosen action
    - value: game outcome from this state (1.0 win, 0.0 loss)
    """

    def __init__(self, replay_dir: str, max_samples: int | None = None) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.values: list[float] = []
        self._load(replay_dir, max_samples)

    def _load(self, replay_dir: str, max_samples: int | None) -> None:
        if not os.path.exists(replay_dir):
            logger.warning(f"Replay data not found at {replay_dir}")
            return

        replay_files = sorted(Path(replay_dir).glob("*.npz"))
        count = 0
        for fp in replay_files:
            if max_samples and count >= max_samples:
                break
            try:
                data = np.load(fp)
                self.states.extend(data["states"])
                self.actions.extend(data["actions"])
                self.values.extend(data["values"])
                count += len(data["states"])
            except Exception:
                logger.warning(f"Failed to load {fp}")

        logger.info(f"Loaded {len(self.states)} replay samples from {replay_dir}")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            self.states[idx].astype(np.float32),
            np.int64(self.actions[idx]),
            np.float32(self.values[idx]),
        )


def train_imitation(
    model: nn.Module,
    config: ImitationConfig,
    device: str = "cuda",
) -> dict:
    """Run imitation learning warm-start.

    Phase 1: Train value network on matchup data
    Phase 2: Train policy + value on replay data (if available)

    Returns training metrics.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    metrics = {"value_loss": [], "policy_loss": [], "total_loss": []}

    # Phase 1: Matchup value training
    matchup_path = os.path.join(config.data_dir, "matchups.csv")
    if os.path.exists(matchup_path):
        logger.info("Phase 1: Training value network on matchup data")
        matchup_dataset = MatchupDataset(matchup_path, max_samples=config.max_samples)
        if len(matchup_dataset) > 0:
            loader = DataLoader(matchup_dataset, batch_size=config.batch_size, shuffle=True)
            for epoch in range(config.num_epochs):
                epoch_loss = _train_matchup_epoch(model, loader, optimizer, device)
                metrics["value_loss"].append(epoch_loss)
                logger.info(f"Matchup epoch {epoch}: loss={epoch_loss:.4f}")

    # Phase 2: Replay policy training
    replay_dir = os.path.join(config.data_dir, "replays")
    if os.path.exists(replay_dir):
        logger.info("Phase 2: Training policy on replay data")
        replay_dataset = ReplayDataset(replay_dir, max_samples=config.max_samples)
        if len(replay_dataset) > 0:
            loader = DataLoader(replay_dataset, batch_size=config.batch_size, shuffle=True)
            for epoch in range(config.num_epochs):
                p_loss, v_loss = _train_replay_epoch(
                    model, loader, optimizer, device, config
                )
                metrics["policy_loss"].append(p_loss)
                metrics["total_loss"].append(p_loss + v_loss)
                logger.info(f"Replay epoch {epoch}: policy_loss={p_loss:.4f}, value_loss={v_loss:.4f}")

    return metrics


def _train_matchup_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for deck0, deck1, winner in loader:
        deck0 = deck0.to(device)
        deck1 = deck1.to(device)
        winner = winner.to(device)

        # Concatenate decks as input features
        features = torch.cat([deck0, deck1], dim=-1)

        # Forward pass — expect model to have a value head
        # This is a simplified interface; actual model may need adaptation
        optimizer.zero_grad()

        if hasattr(model, "predict_matchup_value"):
            value_pred = model.predict_matchup_value(features)
        else:
            # Generic: use the model's value head with dummy state
            _, value_pred = model(features)

        loss = F.mse_loss(value_pred.squeeze(-1), winner)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def _train_replay_epoch(model, loader, optimizer, device, config) -> tuple[float, float]:
    model.train()
    total_p_loss = 0.0
    total_v_loss = 0.0
    n_batches = 0

    for states, actions, values in loader:
        states = states.to(device)
        actions = actions.to(device)
        values = values.to(device)

        optimizer.zero_grad()

        policy_logits, value_pred = model(states)

        p_loss = F.cross_entropy(policy_logits, actions)
        v_loss = F.mse_loss(value_pred.squeeze(-1), values)

        loss = config.policy_loss_weight * p_loss + config.value_loss_weight * v_loss
        loss.backward()
        optimizer.step()

        total_p_loss += p_loss.item()
        total_v_loss += v_loss.item()
        n_batches += 1

    return total_p_loss / max(1, n_batches), total_v_loss / max(1, n_batches)
