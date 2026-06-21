"""Imitation / behavioral-cloning warm-start.

Self-play from random weights spends a long time discovering even basic Clash
Royale strategy. Both AlphaGo and AlphaStar warm-started from human play before
switching to RL. We provide the same capability with a network-correct,
end-to-end-runnable implementation:

1. **Behavioral cloning** (`generate_expert_dataset` + `train_behavioral_cloning`):
   record ``(spatial, scalar, valid_mask, expert_action, outcome)`` from a strong
   reference bot (e.g. :class:`eval.baseline_agents.MetaAgent`) on the verified
   engine, then supervise the policy (cross-entropy to the expert action) and
   value (MSE to the game outcome). This needs no external download and produces
   a competent baseline to seed self-play.
2. **Value warm-start** (`train_value_warmstart`): regress the value head on deck
   matchup outcomes (encode each matchup's *initial* state and fit
   ``V ≈ outcome``). This is what ``scripts/train_v2.py --warmstart-data`` calls.

External human datasets (KataCR replays, the Kaggle 37.9M-match dump) are
drop-in: convert them to the same npz schema (BC) or matchup file (value) and the
same trainers consume them. The earlier version of this module called
``model(states)`` / ``model(features)``, which never matched the real
``forward(spatial, scalar, action_mask, ...)`` interface and could not run.

Data sources:
  - Kaggle: kaggle.com/datasets/bwandowando/clash-royale-season-18-dec-0320-dataset
  - KataCR: github.com/wty-yy/Clash-Royale-Replay-Dataset
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

from crsim.cards import CardType
from crsim.game import Action, CRGame
from model.features import encode_state, extract_entity_features
from model.network import CRZeroNet
from model.transformer_net import CRStarNet

logger = logging.getLogger(__name__)

DECK_SIZE = 8


@dataclass
class ImitationConfig:
    """Configuration for imitation / behavioral-cloning warm-start."""

    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 10
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    max_samples: int | None = None  # limit for debugging


def _forward_policy_value(
    model: nn.Module,
    spatial: torch.Tensor,
    scalar: torch.Tensor,
    mask: torch.Tensor,
    entity_features: torch.Tensor | None = None,
    entity_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call either network through its real ``(spatial, scalar, mask, ...)`` API.

    ``CRZeroNet`` returns ``(logits, value)``; ``CRStarNet`` additionally accepts
    entity tokens and returns ``(logits, value, extras)``. Returns
    ``(logits, value)`` with ``value`` flattened to ``(B,)``.
    """
    if isinstance(model, CRStarNet):
        logits, value, _ = model(
            spatial, scalar, mask,
            entity_features=entity_features, entity_mask=entity_mask,
        )
    else:
        logits, value = model(spatial, scalar, mask)
    return logits, value.reshape(-1)


def _resolve_deck(cards: list) -> list[CardType]:
    """Resolve a deck of card names or indices to ``CardType`` values."""
    pool = list(CardType)
    by_name = {c.name: c for c in pool}
    resolved: list[CardType] = []
    for c in cards:
        if isinstance(c, CardType):
            resolved.append(c)
        elif isinstance(c, str) and c in by_name:
            resolved.append(by_name[c])
        elif isinstance(c, (int, np.integer)) and 0 <= int(c) < len(pool):
            resolved.append(pool[int(c)])
        else:
            raise ValueError(f"unresolvable card identifier: {c!r}")
    return resolved


# --------------------------------------------------------------------------- #
# Behavioral cloning from a reference bot
# --------------------------------------------------------------------------- #

def generate_expert_dataset(
    out_path: str,
    n_games: int = 16,
    expert=None,
    opponent=None,
    backend: str = "python",
    decision_interval: int = 8,
    max_ticks: int | None = None,
    seed: int = 0,
) -> int:
    """Record expert ``(state, action, outcome)`` samples to an npz file.

    Both seats are driven by reference bots and every decision is recorded from
    the acting player's perspective with the terminal game outcome as the value
    target. Returns the number of samples written.
    """
    from crsim.actions import action_id_to_action
    from eval.baseline_agents import MetaAgent
    from training.vectorized_selfplay import make_game

    expert = expert or MetaAgent()
    opponent = opponent or MetaAgent()
    rng = np.random.default_rng(seed)

    spatials, scalars, masks, actions = [], [], [], []
    players, ent_feats, ent_masks, game_ids = [], [], [], []
    rewards_per_game: list[list[float]] = []  # [reward_p0, reward_p1] per game

    for g_idx in range(n_games):
        game = make_game(backend, seed=int(rng.integers(0, 2**31)))
        agents = [expert, opponent]
        steps = 0
        limit = max_ticks if max_ticks is not None else 7200
        while not game.done and steps < limit:
            if steps % decision_interval == 0:
                step_actions = []
                for p in (0, 1):
                    action_id = agents[p].select_action(game, p)
                    spatial, scalar = encode_state(game, p)
                    ef, em = extract_entity_features(game, p)
                    spatials.append(spatial)
                    scalars.append(scalar)
                    masks.append(game.get_valid_actions_mask(p))
                    actions.append(int(action_id))
                    players.append(p)
                    ent_feats.append(ef)
                    ent_masks.append(em)
                    game_ids.append(g_idx)
                    step_actions.append(action_id_to_action(action_id, p))
                game.step(step_actions)
            else:
                game.step([Action(0, -1), Action(1, -1)])
            steps += 1
        rewards_per_game.append([game.get_reward(0), game.get_reward(1)])

    # Value target per sample = the acting player's terminal game outcome.
    values = np.array(
        [rewards_per_game[game_ids[i]][players[i]] for i in range(len(players))],
        dtype=np.float32,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spatial=np.asarray(spatials, dtype=np.float32),
        scalar=np.asarray(scalars, dtype=np.float32),
        mask=np.asarray(masks, dtype=bool),
        action=np.asarray(actions, dtype=np.int64),
        value=values,
        entity_features=np.asarray(ent_feats, dtype=np.float32),
        entity_mask=np.asarray(ent_masks, dtype=bool),
    )
    return len(actions)


class ExpertDataset(Dataset):
    """Behavioral-cloning dataset of expert ``(state, action, value)`` samples."""

    def __init__(self, npz_path: str, max_samples: int | None = None) -> None:
        data = np.load(npz_path)
        n = len(data["action"])
        if max_samples is not None:
            n = min(n, max_samples)
        self.spatial = data["spatial"][:n].astype(np.float32)
        self.scalar = data["scalar"][:n].astype(np.float32)
        self.mask = data["mask"][:n].astype(bool)
        self.action = data["action"][:n].astype(np.int64)
        self.value = data["value"][:n].astype(np.float32)
        self.entity_features = data["entity_features"][:n].astype(np.float32)
        self.entity_mask = data["entity_mask"][:n].astype(bool)

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, idx: int):
        return (
            self.spatial[idx],
            self.scalar[idx],
            self.mask[idx],
            self.action[idx],
            self.value[idx],
            self.entity_features[idx],
            self.entity_mask[idx],
        )


def train_behavioral_cloning(
    model: nn.Module,
    dataset: ExpertDataset,
    config: ImitationConfig | None = None,
    device: str = "cpu",
) -> dict:
    """Supervise policy (CE to expert action) + value (MSE to outcome)."""
    config = config or ImitationConfig()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "total_loss": []}
    for epoch in range(config.num_epochs):
        model.train()
        p_sum = v_sum = t_sum = 0.0
        n_batches = 0
        for spatial, scalar, mask, action, value, ef, em in loader:
            spatial = spatial.to(device)
            scalar = scalar.to(device)
            mask = mask.to(device)
            action = action.to(device)
            value = value.to(device)
            ef = ef.to(device)
            em = em.to(device)

            optimizer.zero_grad()
            logits, value_pred = _forward_policy_value(model, spatial, scalar, mask, ef, em)
            p_loss = F.cross_entropy(logits, action)
            v_loss = F.mse_loss(value_pred, value)
            loss = config.policy_loss_weight * p_loss + config.value_loss_weight * v_loss
            loss.backward()
            optimizer.step()

            p_sum += p_loss.item()
            v_sum += v_loss.item()
            t_sum += loss.item()
            n_batches += 1

        nb = max(1, n_batches)
        metrics["policy_loss"].append(p_sum / nb)
        metrics["value_loss"].append(v_sum / nb)
        metrics["total_loss"].append(t_sum / nb)
        logger.info(
            "BC epoch %d: policy=%.4f value=%.4f", epoch, p_sum / nb, v_sum / nb
        )
    return metrics


def train_imitation(
    model: nn.Module,
    config: ImitationConfig | None = None,
    device: str = "cpu",
    npz_path: str = "data/imitation/expert.npz",
) -> dict:
    """Convenience entrypoint: behavioral cloning from a saved expert dataset."""
    config = config or ImitationConfig()
    if not os.path.exists(npz_path):
        logger.warning("Expert dataset not found at %s", npz_path)
        return {}
    dataset = ExpertDataset(npz_path, max_samples=config.max_samples)
    if len(dataset) == 0:
        return {}
    return train_behavioral_cloning(model, dataset, config, device)


# --------------------------------------------------------------------------- #
# Value warm-start from deck-matchup outcomes
# --------------------------------------------------------------------------- #

def _p0_win_from(obj: dict) -> float:
    """Extract player-0 win signal in ``[0, 1]`` (1.0 = player 0 won).

    Accepts ``p0_win`` (probability, preferred) or ``winner`` (seat 0/1).
    """
    if "p0_win" in obj and obj["p0_win"] is not None:
        return float(obj["p0_win"])
    return 1.0 if int(obj["winner"]) == 0 else 0.0


def _load_battles(path: str, max_samples: int | None) -> list[tuple[list, list, float]]:
    """Load deck-matchup outcomes from JSONL or CSV.

    JSONL: ``{"deck0": [...8], "deck1": [...8], "p0_win": 0..1}`` (or ``"winner": 0|1``
    for the winning seat). CSV: columns ``player0_card0..7, player1_card0..7`` plus
    ``p0_win`` or ``winner``. Cards may be ``CardType`` names or indices. The
    returned outcome is always player-0's win signal in ``[0, 1]``.
    """
    rows: list[tuple[list, list, float]] = []
    if not os.path.exists(path):
        logger.warning("Battle data not found at %s", path)
        return rows

    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append((obj["deck0"], obj["deck1"], _p0_win_from(obj)))
                if max_samples and len(rows) >= max_samples:
                    break
    else:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                d0 = [row[f"player0_card{i}"] for i in range(DECK_SIZE)]
                d1 = [row[f"player1_card{i}"] for i in range(DECK_SIZE)]
                rows.append((d0, d1, _p0_win_from(row)))
                if max_samples and len(rows) >= max_samples:
                    break
    logger.info("Loaded %d battle matchups from %s", len(rows), path)
    return rows


def train_value_warmstart(
    battles_path: str,
    output_path: str,
    epochs: int = 5,
    device: str = "cpu",
    model: nn.Module | None = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_samples: int | None = None,
) -> dict:
    """Warm-start the value head on deck-matchup outcomes.

    Each matchup's *initial* state is encoded and the value head is regressed
    toward the outcome (``+1`` P0 win, ``-1`` P0 loss). Saves the (warm-started)
    model state_dict to ``output_path``. Returns training metrics.
    """
    battles = _load_battles(battles_path, max_samples)
    if not battles:
        logger.warning("No battles to train on; skipping value warm-start")
        return {"value_loss": []}

    # Encode initial states once.
    spatials, scalars, masks, ent_feats, ent_masks, targets = [], [], [], [], [], []
    for deck0, deck1, p0_win in battles:
        try:
            d0 = _resolve_deck(deck0)
            d1 = _resolve_deck(deck1)
        except ValueError as e:
            logger.warning("skipping matchup: %s", e)
            continue
        game = CRGame(deck_p0=d0, deck_p1=d1, seed=0)
        spatial, scalar = encode_state(game, 0)
        ef, em = extract_entity_features(game, 0)
        spatials.append(spatial)
        scalars.append(scalar)
        masks.append(game.get_valid_actions_mask(0))
        ent_feats.append(ef)
        ent_masks.append(em)
        # p0_win in [0,1] -> value target in [-1,+1] (1.0 win -> +1)
        targets.append(2.0 * p0_win - 1.0)

    if not targets:
        return {"value_loss": []}

    spatial_t = torch.from_numpy(np.stack(spatials)).float()
    scalar_t = torch.from_numpy(np.stack(scalars)).float()
    mask_t = torch.from_numpy(np.stack(masks)).bool()
    ef_t = torch.from_numpy(np.stack(ent_feats)).float()
    em_t = torch.from_numpy(np.stack(ent_masks)).bool()
    target_t = torch.tensor(targets, dtype=torch.float32)

    model = model or CRZeroNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n = len(target_t)
    metrics: dict[str, list[float]] = {"value_loss": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            _, value_pred = _forward_policy_value(
                model,
                spatial_t[idx].to(device),
                scalar_t[idx].to(device),
                mask_t[idx].to(device),
                ef_t[idx].to(device),
                em_t[idx].to(device),
            )
            loss = F.mse_loss(value_pred, target_t[idx].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        metrics["value_loss"].append(epoch_loss / max(1, n_batches))
        logger.info("Value warm-start epoch %d: loss=%.4f", epoch, metrics["value_loss"][-1])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info("Saved value-warm-started model to %s", output_path)
    return metrics
