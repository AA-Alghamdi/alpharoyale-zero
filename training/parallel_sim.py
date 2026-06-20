"""Parallel simulation architecture for self-play training.

Architecture:
    [CPU Workers x N] → [State Queue] → [GPU Batch Inference x M]
                                               ↓
                       [Policy/Value Queue] ← [Results]
                                 ↓
                       [Replay Buffer] → [Training Loop]

This is the standard architecture used by AlphaZero, MuZero, and
EfficientZero. Our Rust engine makes the CPU side fast enough that
GPUs become the bottleneck (which maximizes GPU utilization).
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ParallelSimConfig:
    """Configuration for parallel game generation."""

    n_cpu_workers: int = 8  # game simulation workers
    n_gpu_workers: int = 1  # neural net inference workers
    batch_size: int = 256  # states per GPU batch
    max_queue_size: int = 4096  # max pending states
    games_per_worker_batch: int = 16  # games to play before yielding
    use_rust_engine: bool = True


@dataclass
class GameResult:
    """Result of a completed game for the replay buffer."""

    states: list[np.ndarray]
    policies: list[np.ndarray]
    values: list[float]
    rewards: list[float]
    player_0_deck: list[int]
    player_1_deck: list[int]
    winner: int  # 0, 1, or -1 for draw
    game_length: int  # in ticks
    metadata: dict = field(default_factory=dict)


class InferenceRequest:
    """A state waiting for neural net evaluation."""

    __slots__ = (
        "worker_id", "game_id", "spatial", "scalar",
        "entity_features", "entity_mask", "valid_actions",
        "result_event", "policy", "value",
    )

    def __init__(
        self,
        worker_id: int,
        game_id: int,
        spatial: np.ndarray,
        scalar: np.ndarray,
        entity_features: np.ndarray,
        entity_mask: np.ndarray,
        valid_actions: np.ndarray,
    ) -> None:
        self.worker_id = worker_id
        self.game_id = game_id
        self.spatial = spatial
        self.scalar = scalar
        self.entity_features = entity_features
        self.entity_mask = entity_mask
        self.valid_actions = valid_actions
        self.result_event = threading.Event()
        self.policy: np.ndarray | None = None
        self.value: float | None = None


class GPUBatcher:
    """Collects inference requests and batches them for GPU evaluation.

    Runs as a daemon thread. Waits for batch_size requests or a timeout,
    then evaluates them all in one GPU forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        request_queue: queue.Queue,
        batch_size: int = 256,
        timeout: float = 0.005,  # 5ms — flush partial batches quickly
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.request_queue = request_queue
        self.batch_size = batch_size
        self.timeout = timeout
        self.device = device
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._batches_processed = 0
        self._total_inferences = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        self.model.eval()
        while not self._stop.is_set():
            batch: list[InferenceRequest] = []

            # Collect up to batch_size requests
            try:
                req = self.request_queue.get(timeout=self.timeout)
                batch.append(req)
            except queue.Empty:
                continue

            # Try to fill the batch
            while len(batch) < self.batch_size:
                try:
                    req = self.request_queue.get_nowait()
                    batch.append(req)
                except queue.Empty:
                    break

            if not batch:
                continue

            # Stack structured inputs into batch tensors
            spatial_batch = torch.from_numpy(
                np.stack([r.spatial for r in batch])
            ).float().to(self.device)
            scalar_batch = torch.from_numpy(
                np.stack([r.scalar for r in batch])
            ).float().to(self.device)
            mask_batch = torch.from_numpy(
                np.stack([r.valid_actions for r in batch])
            ).float().to(self.device)
            entity_batch = torch.from_numpy(
                np.stack([r.entity_features for r in batch])
            ).float().to(self.device)
            entity_mask_batch = torch.from_numpy(
                np.stack([r.entity_mask for r in batch])
            ).bool().to(self.device)

            # Forward pass with full CRStarNet signature
            with torch.no_grad():
                result = self.model(
                    spatial_batch, scalar_batch, mask_batch,
                    entity_batch, entity_mask_batch,
                )
                policy_logits = result[0]
                values = result[1]
                policies = torch.softmax(policy_logits, dim=-1).cpu().numpy()
                values = values.squeeze(-1).cpu().numpy()

            # Distribute results back to workers
            for i, req in enumerate(batch):
                # Mask invalid actions
                policy = policies[i]
                if req.valid_actions is not None:
                    policy = policy * req.valid_actions
                    policy_sum = policy.sum()
                    if policy_sum > 0:
                        policy /= policy_sum
                    else:
                        policy = req.valid_actions / req.valid_actions.sum()

                req.policy = policy
                req.value = float(values[i])
                req.result_event.set()

            self._batches_processed += 1
            self._total_inferences += len(batch)

    @property
    def stats(self) -> dict:
        return {
            "batches_processed": self._batches_processed,
            "total_inferences": self._total_inferences,
            "avg_batch_size": (
                self._total_inferences / max(1, self._batches_processed)
            ),
        }


class GameWorker:
    """Runs game simulations on a CPU thread.

    Each worker manages multiple concurrent games. When a game needs
    a neural net evaluation, it submits a request to the GPU batcher
    and waits for the result.
    """

    def __init__(
        self,
        worker_id: int,
        request_queue: queue.Queue,
        result_queue: queue.Queue,
        config: ParallelSimConfig,
        deck_sampler=None,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.config = config
        self.deck_sampler = deck_sampler
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._games_completed = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._play_one_game()
                self.result_queue.put(result)
                self._games_completed += 1
            except Exception:
                logger.exception(f"Worker {self.worker_id} game error")

    def _play_one_game(self) -> GameResult:
        """Play a single game using MCTS + neural net evaluation."""
        # Import here to avoid circular imports
        if self.config.use_rust_engine:
            from crsim.rust_adapter import CRGameRust as GameClass
        else:
            from crsim.game import CRGame as GameClass

        # Sample decks
        if self.deck_sampler:
            deck0, deck1 = self.deck_sampler()
        else:
            deck0, deck1 = self._random_decks()

        game = GameClass(deck_p0=deck0, deck_p1=deck1)

        states = []
        policies = []
        values = []
        rewards = []
        game_id = self._games_completed

        while not game.done:
            # Get state encoding
            state = game.encode_state(player=game.current_player)
            valid_mask = game.get_valid_actions_mask()

            # Request neural net evaluation
            req = InferenceRequest(
                worker_id=self.worker_id,
                game_id=game_id,
                state=state,
                valid_actions=valid_mask,
            )
            self.request_queue.put(req)
            req.result_event.wait(timeout=5.0)

            if req.policy is None:
                # Timeout — play random valid action
                valid_indices = np.where(valid_mask > 0)[0]
                action = np.random.choice(valid_indices)
            else:
                # Sample action from policy
                action = np.random.choice(len(req.policy), p=req.policy)

            states.append(state)
            policies.append(req.policy if req.policy is not None else valid_mask / valid_mask.sum())
            values.append(req.value if req.value is not None else 0.0)

            game.step(action)
            rewards.append(0.0)  # sparse reward at end

        # Set final reward
        winner = game.winner
        if rewards:
            rewards[-1] = 1.0 if winner == 0 else (-1.0 if winner == 1 else 0.0)

        return GameResult(
            states=states,
            policies=policies,
            values=values,
            rewards=rewards,
            player_0_deck=[c.value for c in deck0],
            player_1_deck=[c.value for c in deck1],
            winner=winner,
            game_length=game.tick,
        )

    def _random_decks(self):
        from crsim.cards import CardType
        all_cards = list(CardType)
        np.random.shuffle(all_cards)
        return list(all_cards[:8]), list(all_cards[8:16])


class ParallelTrainer:
    """Orchestrates parallel game generation + GPU inference + training.

    Usage:
        trainer = ParallelTrainer(model, config)
        trainer.start()

        for batch in trainer.generate_games(n_games=10000):
            loss = train_on_batch(model, batch)
            trainer.update_model(model)

        trainer.stop()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: ParallelSimConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model
        self.config = config or ParallelSimConfig()
        self.device = device

        self._request_queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._result_queue: queue.Queue = queue.Queue()

        self._gpu_batcher = GPUBatcher(
            model=model,
            request_queue=self._request_queue,
            batch_size=self.config.batch_size,
            device=device,
        )

        self._workers: list[GameWorker] = []
        for i in range(self.config.n_cpu_workers):
            worker = GameWorker(
                worker_id=i,
                request_queue=self._request_queue,
                result_queue=self._result_queue,
                config=self.config,
            )
            self._workers.append(worker)

    def start(self) -> None:
        """Start all workers and the GPU batcher."""
        self._gpu_batcher.start()
        for worker in self._workers:
            worker.start()
        logger.info(
            f"Started {len(self._workers)} CPU workers + GPU batcher "
            f"(batch_size={self.config.batch_size})"
        )

    def stop(self) -> None:
        """Stop all workers and the GPU batcher."""
        for worker in self._workers:
            worker.stop()
        self._gpu_batcher.stop()

    def collect_games(self, n_games: int) -> list[GameResult]:
        """Collect n_games completed games."""
        results = []
        while len(results) < n_games:
            try:
                result = self._result_queue.get(timeout=30.0)
                results.append(result)
            except queue.Empty:
                logger.warning(f"Timeout waiting for games ({len(results)}/{n_games})")
        return results

    def update_model(self, new_state_dict: dict) -> None:
        """Update the model weights used for inference."""
        self.model.load_state_dict(new_state_dict)

    @property
    def stats(self) -> dict:
        total_games = sum(w._games_completed for w in self._workers)
        return {
            "total_games_completed": total_games,
            "gpu_batcher": self._gpu_batcher.stats,
            "result_queue_size": self._result_queue.qsize(),
            "request_queue_size": self._request_queue.qsize(),
        }
