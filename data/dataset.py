"""PyTorch datasets for CR training data.

Supports:
  1. Battle outcome dataset (from API scraping / Kaggle) — for imitation learning
  2. Self-play trajectory dataset — for RL training
  3. Card stats loader — from cr-csv decoded game data
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Card name → global ID mapping (from Supercell API, common subset)
# This is a condensed mapping; full mapping loaded from card_stats.json at runtime
CARD_NAME_TO_IDX: dict[str, int] = {}
CARD_IDX_TO_NAME: dict[int, str] = {}


def load_card_stats(path: str = "data/card_stats.json") -> dict:
    """Load card stats from JSON file (cr-csv format or API format).

    Returns dict mapping card_id → {name, elixir, rarity, hitpoints, damage, ...}
    """
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


class BattleOutcomeDataset(Dataset):
    """Dataset of battle outcomes for imitation/warm-start learning.

    Each sample encodes:
    - p0_deck: one-hot over card vocabulary (sparse)
    - p1_deck: one-hot over card vocabulary
    - p0_trophies, p1_trophies: normalized trophy counts
    - winner: 0 or 1 (target)

    This is used for warm-starting the value network — learning which decks
    beat which other decks at what trophy levels.
    """

    def __init__(
        self,
        battles_path: str,
        card_vocab_size: int = 200,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.card_vocab_size = card_vocab_size
        self.samples: list[dict] = []

        with open(battles_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                record = json.loads(line.strip())
                if record.get("winner", -1) in (0, 1):
                    self.samples.append(record)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.samples[idx]

        # Encode decks as multi-hot vectors
        p0_deck = torch.zeros(self.card_vocab_size)
        p1_deck = torch.zeros(self.card_vocab_size)

        for card_id in record["deck_p0"]:
            if 0 <= card_id < self.card_vocab_size:
                p0_deck[card_id] = 1.0

        for card_id in record["deck_p1"]:
            if 0 <= card_id < self.card_vocab_size:
                p1_deck[card_id] = 1.0

        # Normalize trophies (0-9000 range)
        p0_trophies = min(record.get("trophies_p0", 5000), 9000) / 9000.0
        p1_trophies = min(record.get("trophies_p1", 5000), 9000) / 9000.0

        # Features: [p0_deck, p1_deck, p0_trophies, p1_trophies]
        features = torch.cat([
            p0_deck, p1_deck,
            torch.tensor([p0_trophies, p1_trophies]),
        ])

        winner = torch.tensor(record["winner"], dtype=torch.float32)

        return {"features": features, "winner": winner}


class TrajectoryDataset(Dataset):
    """Dataset of self-play trajectories for RL training.

    Each sample is a (spatial, scalar, policy, value) tuple from a game position.
    Supports loading from numpy archives or the ReplayBuffer.
    """

    def __init__(self, data_dir: str, max_samples: int | None = None) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self._spatial: np.ndarray | None = None
        self._scalar: np.ndarray | None = None
        self._policy: np.ndarray | None = None
        self._value: np.ndarray | None = None
        self._size = 0

        # Load from .npz files
        npz_files = sorted(self.data_dir.glob("trajectory_*.npz"))
        if npz_files:
            spatials, scalars, policies, values = [], [], [], []
            total = 0
            for npz_path in npz_files:
                data = np.load(npz_path)
                n = len(data["spatial"])
                if max_samples and total + n > max_samples:
                    n = max_samples - total
                spatials.append(data["spatial"][:n])
                scalars.append(data["scalar"][:n])
                policies.append(data["policy"][:n])
                values.append(data["value"][:n])
                total += n
                if max_samples and total >= max_samples:
                    break
            if spatials:
                self._spatial = np.concatenate(spatials)
                self._scalar = np.concatenate(scalars)
                self._policy = np.concatenate(policies)
                self._value = np.concatenate(values)
                self._size = len(self._spatial)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "spatial": torch.from_numpy(self._spatial[idx]),
            "scalar": torch.from_numpy(self._scalar[idx]),
            "policy": torch.from_numpy(self._policy[idx]),
            "value": torch.tensor(self._value[idx], dtype=torch.float32),
        }


class KaggleBattleDataset(Dataset):
    """Loader for Kaggle's 37.9M match dataset.

    Expects CSV files with columns like:
    player.deck.card1.name, ..., player.deck.card8.name,
    opponent.deck.card1.name, ..., opponent.deck.card8.name,
    player.crownsEarned, opponent.crownsEarned,
    player.startingTrophies, opponent.startingTrophies
    """

    def __init__(
        self,
        csv_path: str,
        card_vocab_size: int = 200,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.card_vocab_size = card_vocab_size
        self.samples: list[tuple[list[str], list[str], int, int, int]] = []

        import csv

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break

                # Extract deck card names
                p_deck = []
                o_deck = []
                for j in range(1, 9):
                    p_key = f"player.deck.card{j}.name"
                    o_key = f"opponent.deck.card{j}.name"
                    if p_key in row and o_key in row:
                        p_deck.append(row[p_key])
                        o_deck.append(row[o_key])

                if len(p_deck) != 8 or len(o_deck) != 8:
                    continue

                p_crowns = int(row.get("player.crownsEarned", 0))
                o_crowns = int(row.get("opponent.crownsEarned", 0))

                if p_crowns > o_crowns:
                    winner = 0
                elif o_crowns > p_crowns:
                    winner = 1
                else:
                    continue  # skip draws

                p_trophies = int(row.get("player.startingTrophies", 5000))
                o_trophies = int(row.get("opponent.startingTrophies", 5000))

                self.samples.append((p_deck, o_deck, winner, p_trophies, o_trophies))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p_deck, o_deck, winner, p_trophies, o_trophies = self.samples[idx]

        # Encode card names as hashed indices
        p_onehot = torch.zeros(self.card_vocab_size)
        o_onehot = torch.zeros(self.card_vocab_size)

        for name in p_deck:
            h = hash(name) % self.card_vocab_size
            p_onehot[h] = 1.0
        for name in o_deck:
            h = hash(name) % self.card_vocab_size
            o_onehot[h] = 1.0

        features = torch.cat([
            p_onehot, o_onehot,
            torch.tensor([p_trophies / 9000.0, o_trophies / 9000.0]),
        ])

        return {
            "features": features,
            "winner": torch.tensor(winner, dtype=torch.float32),
        }


def save_trajectories(
    spatials: np.ndarray,
    scalars: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    output_dir: str,
    chunk_name: str = "trajectory_000",
) -> None:
    """Save trajectory data as a compressed numpy archive."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{chunk_name}.npz",
        spatial=spatials,
        scalar=scalars,
        policy=policies,
        value=values,
    )
