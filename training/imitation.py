"""Imitation learning pre-training — warm-start the network from battle data.

Two modes:
  1. Value warm-start: Train the value head on deck matchup → outcome data
     (from Kaggle 37.9M matches or scraped API data). This teaches the network
     "which decks beat which" before any self-play.

  2. Policy warm-start: Train the policy head from expert replay data
     (from KataCR replay dataset or TV Royale frames with labeled actions).

This dramatically reduces the cold-start problem — instead of 30% of training
time learning that "deploying troops is good", the network starts with a
competent baseline from human data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f_nn  # noqa: N812
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.dataset import BattleOutcomeDataset, KaggleBattleDataset

logger = logging.getLogger(__name__)


class DeckValueNetwork(nn.Module):
    """Lightweight network for predicting battle outcomes from decks.

    Takes concatenated [deck_p0, deck_p1, trophies] and predicts P(p0 wins).
    Used for warm-starting the value head / learning card embeddings.
    """

    def __init__(self, input_dim: int = 402, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CardEmbeddingModel(nn.Module):
    """Learn card embeddings from battle outcome data.

    Each card gets a learned embedding. Deck = average of card embeddings.
    Matchup = interaction of deck embeddings → predicted outcome.

    The learned card embeddings can then be transferred to the main network
    to provide a meaningful initialization for card representations.
    """

    def __init__(self, vocab_size: int = 200, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.interaction = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, 256),  # +2 for trophies
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def embed_deck(self, deck_ids: torch.Tensor) -> torch.Tensor:
        """Embed a deck (B, 8) → (B, embed_dim) via mean pooling."""
        embs = self.embed(deck_ids)  # (B, 8, D)
        return embs.mean(dim=1)  # (B, D)

    def forward(
        self,
        deck_p0: torch.Tensor,
        deck_p1: torch.Tensor,
        trophies: torch.Tensor,
    ) -> torch.Tensor:
        e0 = self.embed_deck(deck_p0)
        e1 = self.embed_deck(deck_p1)
        combined = torch.cat([e0, e1, trophies], dim=-1)
        return self.interaction(combined).squeeze(-1)


def train_value_warmstart(
    battles_path: str,
    output_path: str = "checkpoints/value_warmstart.pt",
    card_vocab_size: int = 200,
    batch_size: int = 2048,
    epochs: int = 10,
    lr: float = 1e-3,
    max_samples: int | None = None,
    device: str = "cuda",
) -> DeckValueNetwork:
    """Train a value warm-start model from battle outcome data.

    This can be from API-scraped data (JSONL) or Kaggle CSV.
    """
    logger.info("Loading battle data from %s", battles_path)

    if battles_path.endswith(".csv"):
        dataset = KaggleBattleDataset(battles_path, card_vocab_size, max_samples)
    else:
        dataset = BattleOutcomeDataset(battles_path, card_vocab_size, max_samples)

    logger.info("Loaded %d battles", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    input_dim = card_vocab_size * 2 + 2
    model = DeckValueNetwork(input_dim=input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["winner"].to(device)

            preds = model(features)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            correct += ((preds > 0.5).float() == targets).sum().item()
            total += len(targets)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        logger.info(
            "Epoch %d/%d: loss=%.4f accuracy=%.4f",
            epoch + 1, epochs, avg_loss, accuracy,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            logger.info("Saved best model to %s", output_path)

    return model


def transfer_value_weights(
    warmstart_model: DeckValueNetwork,
    main_model: nn.Module,
) -> None:
    """Transfer learned weights from the warm-start model to the main network's value head.

    Copies the final hidden→output layers from warmstart to the main model's
    value head, adapting for dimension mismatches when necessary.
    """
    with torch.no_grad():
        # DeckValueNetwork layers: L(in,512) → L(512,512) → L(512,256) → L(256,1)
        ws_layers = [m for m in warmstart_model.net if isinstance(m, nn.Linear)]

        if not hasattr(main_model, "value_head"):
            logger.warning("Main model has no value_head — skipping transfer")
            return

        # Find linear layers in the main model's value head
        main_layers: list[nn.Linear] = []
        for m in main_model.value_head.modules():
            if isinstance(m, nn.Linear):
                main_layers.append(m)

        if not main_layers or not ws_layers:
            logger.warning("Could not find layers to transfer")
            return

        # Transfer final layer (256→1) if shapes match
        ws_final = ws_layers[-1]
        main_final = main_layers[-1]
        if ws_final.weight.shape == main_final.weight.shape:
            main_final.weight.copy_(ws_final.weight)
            if ws_final.bias is not None and main_final.bias is not None:
                main_final.bias.copy_(ws_final.bias)
            logger.info(
                "Transferred final layer (%s → %s)",
                ws_final.weight.shape, main_final.weight.shape,
            )

        # Transfer second-to-last layer (512→256) if shapes match
        if len(ws_layers) >= 2 and len(main_layers) >= 2:
            ws_prev = ws_layers[-2]
            main_prev = main_layers[-2]
            if ws_prev.weight.shape == main_prev.weight.shape:
                main_prev.weight.copy_(ws_prev.weight)
                if ws_prev.bias is not None and main_prev.bias is not None:
                    main_prev.bias.copy_(ws_prev.bias)
                logger.info(
                    "Transferred penultimate layer (%s → %s)",
                    ws_prev.weight.shape, main_prev.weight.shape,
                )
            else:
                # Partial transfer: copy overlapping dimensions
                min_out = min(ws_prev.weight.shape[0], main_prev.weight.shape[0])
                min_in = min(ws_prev.weight.shape[1], main_prev.weight.shape[1])
                main_prev.weight[:min_out, :min_in].copy_(
                    ws_prev.weight[:min_out, :min_in]
                )
                logger.info(
                    "Partial transfer penultimate layer (%dx%d of %s)",
                    min_out, min_in, main_prev.weight.shape,
                )

        logger.info("Value warm-start weight transfer complete")


class PolicyWarmstart:
    """Warm-start the policy network from expert replay data.

    Uses the KataCR replay dataset format: episodes of (state, action) pairs
    from expert play, training the policy via behavior cloning.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 3e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train one epoch of behavior cloning."""
        self.model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            spatial = batch["spatial"].to(self.device)
            scalar = batch["scalar"].to(self.device)
            target_policy = batch["policy"].to(self.device)
            target_value = batch["value"].to(self.device)

            # Forward
            policy_logits, value, _ = self.model(spatial, scalar)

            # Policy loss (KL divergence from expert policy)
            log_probs = f_nn.log_softmax(policy_logits, dim=-1)
            policy_loss = f_nn.kl_div(log_probs, target_policy, reduction="batchmean")

            # Value loss (MSE)
            value_loss = f_nn.mse_loss(value, target_value)

            # Combined loss
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "policy_loss": total_policy_loss / max(n_batches, 1),
            "value_loss": total_value_loss / max(n_batches, 1),
        }
