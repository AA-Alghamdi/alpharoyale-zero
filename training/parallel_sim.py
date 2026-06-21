"""Parallel self-play game generation (compatibility wrapper).

Historically this module held a thread/queue prototype
(``GPUBatcher`` + ``GameWorker``) that targeted a single-action turn-based
``CRGame`` interface which never existed — it could not run. The working,
tested implementation now lives in :mod:`training.vectorized_selfplay`, which
advances many games in lockstep and evaluates them in a single batched forward
pass (the actual GPU-utilization win).

This module is kept as a thin, backward-compatible facade: ``ParallelSimConfig``
and ``ParallelTrainer`` delegate to :class:`~training.vectorized_selfplay.VectorizedSelfPlay`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from crsim.cards import CardType
from training.replay_buffer import ReplayEntry
from training.vectorized_selfplay import (
    VectorizedSelfPlay,
    VectorizedSelfPlayConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ParallelSimConfig:
    """Configuration for parallel game generation.

    ``n_cpu_workers`` is retained for backward compatibility but no longer maps
    to OS threads; throughput now comes from batching ``batch_size`` positions
    per GPU forward pass across ``batch_size`` concurrent games.
    """

    n_cpu_workers: int = 8
    batch_size: int = 256  # games advanced in lockstep / positions per batch
    decision_interval_ticks: int = 8
    max_ticks: int | None = None
    random_decks: bool = True
    card_pool: list[CardType] = field(default_factory=lambda: list(CardType))
    use_rust_engine: bool = False

    def to_vectorized(self) -> VectorizedSelfPlayConfig:
        return VectorizedSelfPlayConfig(
            n_envs=self.batch_size,
            decision_interval_ticks=self.decision_interval_ticks,
            max_ticks=self.max_ticks,
            random_decks=self.random_decks,
            card_pool=self.card_pool,
            backend="rust" if self.use_rust_engine else "python",
        )


class ParallelTrainer:
    """Generate self-play data with batched GPU inference.

    Usage::

        trainer = ParallelTrainer(model, config)
        entries = trainer.collect_games(n_games=1000)   # list[ReplayEntry]
        ...                                             # train on entries
        trainer.update_model(model.state_dict())
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: ParallelSimConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 0,
    ) -> None:
        self.model = model
        self.config = config or ParallelSimConfig()
        self.device = device
        self._engine = VectorizedSelfPlay(
            model=model,
            config=self.config.to_vectorized(),
            device=device,
            seed=seed,
        )
        self._total_games = 0

    def collect_games(self, n_games: int) -> list[ReplayEntry]:
        """Play ``n_games`` self-play games and return replay entries."""
        entries = self._engine.generate(n_games)
        self._total_games += n_games
        return entries

    def update_model(self, new_state_dict: dict) -> None:
        self.model.load_state_dict(new_state_dict)

    @property
    def stats(self) -> dict:
        return {"total_games_completed": self._total_games}
