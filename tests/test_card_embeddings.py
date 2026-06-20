"""Tests for the stat-derived card embedding module.

These exercise the public API of ``model/card_embeddings.py`` and, crucially,
the near-zero-shot generalization path that lets an unseen card be embedded from
its stats alone. CPU-only and fast.
"""

from __future__ import annotations

import numpy as np
import torch

from crsim.cards import CARD_DEFS, CardType
from crsim.constants import NUM_CARD_TYPES
from model.card_embeddings import (
    STAT_DIM,
    StatBasedCardEmbedding,
    build_stat_matrix,
    stat_vector,
)


def test_build_stat_matrix_shape_finite_float():
    matrix = build_stat_matrix()
    assert matrix.shape == (NUM_CARD_TYPES, STAT_DIM)
    assert np.issubdtype(matrix.dtype, np.floating)
    assert np.all(np.isfinite(matrix))


def test_stat_vector_matches_matrix_rows():
    matrix = build_stat_matrix()
    for ct in (CardType.KNIGHT, CardType.GOLEM, CardType.CANNON, CardType.FIREBALL):
        assert np.allclose(matrix[int(ct)], stat_vector(CARD_DEFS[ct]))


def test_forward_shape_and_deterministic_in_eval():
    model = StatBasedCardEmbedding(embed_dim=64)
    model.eval()
    card_ids = torch.tensor([0, 3, 9, 21, 99], dtype=torch.long)
    with torch.no_grad():
        out1 = model(card_ids)
        out2 = model(card_ids)
    assert out1.shape == (5, 64)
    assert torch.allclose(out1, out2)


def test_all_embeddings_shape():
    model = StatBasedCardEmbedding(embed_dim=32)
    emb = model.all_embeddings()
    assert emb.shape == (NUM_CARD_TYPES, 32)


def test_forward_from_stats_unseen_card_runs():
    model = StatBasedCardEmbedding(embed_dim=64)
    model.eval()
    # A synthetic normalized stat vector for a hypothetical/unseen card.
    synthetic = torch.rand(4, STAT_DIM)
    with torch.no_grad():
        out = model.forward_from_stats(synthetic)
    assert out.shape == (4, 64)
    assert torch.all(torch.isfinite(out))


def test_different_cards_get_different_embeddings():
    model = StatBasedCardEmbedding(embed_dim=64)
    model.eval()
    ids = torch.tensor(
        [int(CardType.SKELETONS), int(CardType.GOLEM), int(CardType.PRINCESS)],
        dtype=torch.long,
    )
    with torch.no_grad():
        emb = model(ids)
    # Distinct cards should not collapse to the same vector.
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[0], emb[2])
    assert not torch.allclose(emb[1], emb[2])


def test_forward_vs_forward_from_stats_differ_by_id_component():
    model = StatBasedCardEmbedding(embed_dim=64)
    model.eval()
    card_id = int(CardType.MUSKETEER)
    ids = torch.tensor([card_id], dtype=torch.long)
    stat_row = torch.from_numpy(build_stat_matrix()[card_id]).unsqueeze(0)
    with torch.no_grad():
        full = model(ids)
        stat_only = model.forward_from_stats(stat_row)
        id_only = model.id_table(ids)
    # forward(id) = stat_mlp(stats) + id_table[id];  forward_from_stats = stat_mlp(stats).
    # Hence the difference must equal the id-table component.
    assert torch.allclose(full - stat_only, id_only, atol=1e-5)


def test_gradients_flow_to_both_paths():
    model = StatBasedCardEmbedding(embed_dim=32)
    model.train()
    card_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    loss = model(card_ids).pow(2).sum()
    loss.backward()

    id_grad = model.id_table.weight.grad
    assert id_grad is not None
    assert torch.any(id_grad != 0)

    mlp_grads = [p.grad for p in model.stat_mlp.parameters()]
    assert all(g is not None for g in mlp_grads)
    assert any(torch.any(g != 0) for g in mlp_grads)
