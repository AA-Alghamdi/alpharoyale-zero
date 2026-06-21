"""Vectorized, GPU-batched self-play generation.

This is the throughput substrate for self-play: instead of evaluating one
position at a time (``self_play_v2.SelfPlayWorkerV2``, which calls the network
once per move per game), it advances ``n_envs`` games **in lockstep** and
evaluates every active position in a **single batched forward pass**. The
network forward is the GPU bottleneck in AlphaZero-style pipelines, so batching
across environments is what keeps a GPU saturated. The exact same code runs on
CPU (just slower), so the path is correct and testable without a GPU and
launch-ready for one (pass ``device="cuda"``).

Two axes are pluggable:

* **engine backend** — ``make_game(backend=...)`` returns either the verified
  Python ``crsim.game.CRGame`` (default) or the fast Rust ``CRGameRust`` adapter
  when the native extension is built. Both expose the same interface, so the
  generator is backend-agnostic.
* **device** — all tensors live on ``device``; CPU and CUDA are identical apart
  from speed.

Policy targets here are the network's masked policy (an expert-iteration-lite /
behaviour signal). Wiring the batched leaf evaluation into a parallel MCTS to
produce visit-count targets is the documented next step; this module is the
batching infrastructure that step builds on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from crsim.actions import action_id_to_action
from crsim.cards import CardType
from crsim.game import Action, CRGame
from model.features import (
    encode_state,
    extract_auxiliary_targets,
    extract_entity_features,
)
from training.replay_buffer import ReplayEntry

logger = logging.getLogger(__name__)

DECK_SIZE = 8


def make_game(
    backend: str = "python",
    deck_p0: list[CardType] | None = None,
    deck_p1: list[CardType] | None = None,
    seed: int | None = None,
):
    """Construct a game on the requested engine backend.

    Both backends share the ``(deck_p0, deck_p1, seed)`` constructor and the
    ``step([Action, Action]) / done / result / get_reward / get_valid_actions_mask``
    interface, so callers are backend-agnostic.

    Parameters
    ----------
    backend : {"python", "rust"}
        ``"python"`` uses the cross-engine-verified ``crsim.game.CRGame``.
        ``"rust"`` uses ``crsim.rust_adapter.CRGameRust`` (requires the
        ``cr_engine_native`` extension built via ``maturin``). The Rust engine
        is fast but its combat is not yet fidelity-verified against the oracle,
        so ``"python"`` is the default.
    """
    if backend == "python":
        return CRGame(deck_p0=deck_p0, deck_p1=deck_p1, seed=seed)
    if backend == "rust":
        from crsim.rust_adapter import CRGameRust

        return CRGameRust(deck_p0=deck_p0, deck_p1=deck_p1, seed=seed)
    raise ValueError(f"unknown backend {backend!r} (expected 'python' or 'rust')")


def rust_backend_available() -> bool:
    """True if the native Rust engine extension can be imported."""
    try:
        import cr_engine_native  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class VectorizedSelfPlayConfig:
    n_envs: int = 32  # games advanced in lockstep / batched per forward pass
    decision_interval_ticks: int = 8
    temperature: float = 1.0  # sampling temperature before the threshold
    temperature_threshold: int = 30  # switch to greedy-ish after N decisions
    final_temperature: float = 0.1
    max_ticks: int | None = None  # truncate long games (terminal value at cut)
    random_decks: bool = True
    card_pool: list[CardType] = field(default_factory=lambda: list(CardType))
    augment_flip: bool = True
    backend: str = "python"


def _sample_action(probs: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Sample an action id from a masked probability vector at a temperature.

    ``probs`` already has ~zero mass on illegal actions (the network masks
    them), so re-tempering preserves legality. ``temperature <= 0`` is greedy.
    """
    if temperature <= 1e-6:
        return int(np.argmax(probs))
    logits = np.log(np.maximum(probs, 1e-12)) / temperature
    logits -= logits.max()
    p = np.exp(logits)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(probs))
    p /= total
    return int(rng.choice(len(p), p=p))


class VectorizedSelfPlay:
    """Generate self-play trajectories with batched network evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: VectorizedSelfPlayConfig | None = None,
        device: torch.device | str | None = None,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.config = config or VectorizedSelfPlayConfig()
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.rng = np.random.default_rng(seed)

    def _random_deck(self) -> list[CardType]:
        pool = self.config.card_pool
        idx = self.rng.choice(len(pool), size=DECK_SIZE, replace=False)
        return [pool[int(i)] for i in idx]

    def batched_eval(
        self, items: list[tuple[object, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``(game, player)`` positions in a single forward pass.

        Returns ``(policies, values)`` where ``policies`` is ``(N, A)`` masked
        softmax probabilities and ``values`` is ``(N,)``. This is the one place
        the GPU is used; everything else is CPU bookkeeping.
        """
        spatials, scalars, masks = [], [], []
        for game, player in items:
            spatial, scalar = encode_state(game, player)
            spatials.append(spatial)
            scalars.append(scalar)
            masks.append(game.get_valid_actions_mask(player))

        spatial_t = torch.from_numpy(np.stack(spatials)).float().to(self.device)
        scalar_t = torch.from_numpy(np.stack(scalars)).float().to(self.device)
        mask_t = torch.from_numpy(np.stack(masks)).bool().to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits, values = self.model(spatial_t, scalar_t, mask_t)
            policies = torch.softmax(logits, dim=-1)
        return policies.cpu().numpy(), values.cpu().numpy()

    def generate(self, n_games: int) -> list[ReplayEntry]:
        """Play ``n_games`` self-play games and return replay entries."""
        entries: list[ReplayEntry] = []
        played = 0
        while played < n_games:
            wave = min(self.config.n_envs, n_games - played)
            entries.extend(self._play_wave(wave))
            played += wave
        return entries

    def _play_wave(self, n_envs: int) -> list[ReplayEntry]:
        cfg = self.config
        games = []
        for _ in range(n_envs):
            if cfg.random_decks:
                d0, d1 = self._random_deck(), self._random_deck()
            else:
                d0 = d1 = None
            seed = int(self.rng.integers(0, 2**31))
            games.append(make_game(cfg.backend, d0, d1, seed))

        trajectories: list[list[dict]] = [[] for _ in range(n_envs)]
        move_counts = [0] * n_envs
        interval = max(1, cfg.decision_interval_ticks)
        active = list(range(n_envs))
        tick = 0
        wave_entries: list[ReplayEntry] = []

        while active:
            is_decision = (tick % interval) == 0
            actions: dict[int, list[Action]] = {
                e: [Action(player=0, hand_slot=-1), Action(player=1, hand_slot=-1)]
                for e in active
            }

            if is_decision:
                items = [(games[e], p) for e in active for p in (0, 1)]
                policies, values = self.batched_eval(items)
                for i, e in enumerate(active):
                    for p in (0, 1):
                        idx = 2 * i + p
                        probs = policies[idx]
                        temp = (
                            cfg.temperature
                            if move_counts[e] < cfg.temperature_threshold
                            else cfg.final_temperature
                        )
                        action_id = _sample_action(probs, temp, self.rng)
                        spatial, scalar = encode_state(games[e], p)
                        ent_feats, ent_mask = extract_entity_features(games[e], p)
                        trajectories[e].append(
                            {
                                "spatial": spatial,
                                "scalar": scalar,
                                "entity_features": ent_feats,
                                "entity_mask": ent_mask,
                                "policy": probs.astype(np.float32),
                                "player": p,
                            }
                        )
                        actions[e][p] = action_id_to_action(action_id, p)
                    move_counts[e] += 1

            for e in active:
                games[e].step(actions[e])

            tick += 1
            still_active = []
            for e in active:
                truncated = (
                    cfg.max_ticks is not None and games[e].tick_count >= cfg.max_ticks
                )
                if games[e].done or truncated:
                    wave_entries.extend(self._finalize(games[e], trajectories[e]))
                else:
                    still_active.append(e)
            active = still_active

        return wave_entries

    def _finalize(self, game, trajectory: list[dict]) -> list[ReplayEntry]:
        final_aux = [extract_auxiliary_targets(game, p) for p in (0, 1)]
        rewards = [game.get_reward(0), game.get_reward(1)]
        entries: list[ReplayEntry] = []
        for step in trajectory:
            p = step["player"]
            aux = final_aux[p]
            entries.append(
                ReplayEntry(
                    spatial=step["spatial"],
                    scalar=step["scalar"],
                    policy=step["policy"],
                    value=rewards[p],
                    entity_features=step["entity_features"],
                    entity_mask=step["entity_mask"],
                    crown_target=int(aux["crown_target"]),
                    tower_hp_target=aux["tower_hp_target"],
                    game_length_target=float(aux["game_length_target"]),
                )
            )

        # flip-augmentation removed: it stored the opponent's reward with the
        # original player's tensors and flipped the wrong axis (see
        # self_play_v2 for the full note). Both players are already recorded.
        return entries
