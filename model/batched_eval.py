"""Batched neural-network evaluation — the core GPU throughput lever.

The single-game search in :mod:`mcts.gumbel_search` evaluates one
``(game, player)`` state per forward pass. On a GPU that wastes almost all of
the available parallelism: a forward pass for a batch of 1 costs nearly the
same wall-clock time as a batch of 512, so running ``N`` games sequentially is
roughly ``N`` times slower than it needs to be.

This module provides the primitive that fixes that: encode many states, stack
them into a single tensor, run **one** forward pass, then split the results
back out. It is intentionally tiny and dependency-free so it can be reused by
both the vectorized self-play worker and any batched evaluation harness.

The numerical result is identical (up to floating-point batching differences)
to calling ``GumbelMuZeroSearch._evaluate`` once per state — this is verified
in ``tests/test_gpu_scale.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from crsim.game import CRGame
from model.features import encode_state, extract_entity_features


def model_has_entity_support(model: torch.nn.Module) -> bool:
    """Whether the model consumes entity-transformer features.

    ``CRStarNet`` does; the legacy ``CRZeroNet`` does not.
    """
    return hasattr(model, "entity_encoder")


def encode_request(
    game: CRGame,
    player: int,
    has_entity: bool,
) -> dict[str, np.ndarray]:
    """Encode a single ``(game, player)`` state into model-ready arrays.

    Returns a dict with ``spatial``/``scalar``/``valid_mask`` always present and
    ``entity_features``/``entity_mask`` present iff ``has_entity``.
    """
    spatial, scalar = encode_state(game, player)
    valid_mask = game.get_valid_actions_mask(player)
    out: dict[str, np.ndarray] = {
        "spatial": spatial,
        "scalar": scalar,
        "valid_mask": valid_mask,
    }
    if has_entity:
        entity_feats, entity_mask = extract_entity_features(game, player)
        out["entity_features"] = entity_feats
        out["entity_mask"] = entity_mask
    return out


@torch.no_grad()
def forward_encoded(
    model: torch.nn.Module,
    encoded: list[dict[str, np.ndarray]],
    device: torch.device,
    has_entity: bool,
    max_batch: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference over a list of pre-encoded states.

    The work is split into chunks of at most ``max_batch`` so GPU memory stays
    bounded regardless of how many states are queued.

    Returns
    -------
    policies : ndarray (N, ACTION_SPACE_SIZE) float32  (softmax, action-masked)
    values   : ndarray (N,) float32
    """
    n = len(encoded)
    if n == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    policies: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for start in range(0, n, max_batch):
        chunk = encoded[start : start + max_batch]
        sp = torch.from_numpy(
            np.stack([e["spatial"] for e in chunk])
        ).to(device)
        sc = torch.from_numpy(
            np.stack([e["scalar"] for e in chunk])
        ).to(device)
        vm = torch.from_numpy(
            np.stack([e["valid_mask"] for e in chunk])
        ).to(device)

        if has_entity:
            ef = torch.from_numpy(
                np.stack([e["entity_features"] for e in chunk])
            ).to(device)
            em = torch.from_numpy(
                np.stack([e["entity_mask"] for e in chunk])
            ).to(device)
            policy_t, value_t = model.predict(
                sp, sc, vm, entity_features=ef, entity_mask=em,
            )
        else:
            policy_t, value_t = model.predict(sp, sc, vm)

        policy_np = policy_t.detach().cpu().numpy()
        value_np = value_t.detach().cpu().numpy()
        if value_np.ndim == 2:
            value_np = value_np[:, 0]
        policies.append(policy_np.astype(np.float32, copy=False))
        values.append(value_np.astype(np.float32, copy=False))

    return np.concatenate(policies, axis=0), np.concatenate(values, axis=0)


@torch.no_grad()
def evaluate_games(
    model: torch.nn.Module,
    requests: list[tuple[CRGame, int]],
    device: torch.device,
    max_batch: int = 512,
) -> list[tuple[np.ndarray, float]]:
    """Convenience wrapper: encode + batch-evaluate a list of game states.

    Equivalent to calling ``GumbelMuZeroSearch._evaluate`` for each request, but
    in a single (chunked) forward pass.

    Returns a list aligned with ``requests`` of ``(policy, value)`` pairs.
    """
    if not requests:
        return []
    has_entity = model_has_entity_support(model)
    encoded = [encode_request(g, p, has_entity) for (g, p) in requests]
    policies, values = forward_encoded(
        model, encoded, device, has_entity, max_batch=max_batch,
    )
    return [(policies[i], float(values[i])) for i in range(len(requests))]
