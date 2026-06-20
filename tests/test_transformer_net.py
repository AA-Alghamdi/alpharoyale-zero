"""Tests for the CRStarNet (Entity Transformer + autoregressive policy)."""

from __future__ import annotations

import pytest
import torch

from crsim.constants import (
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
)
from model.transformer_net import (
    AutoregressivePolicyHead,
    AuxiliaryHeads,
    CRStarNet,
    DynamicsModel,
    EntityEncoder,
    LSTMCore,
    ScalarEncoder,
    SpatialEncoder,
)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestEntityEncoder:
    def test_output_shapes(self, device):
        enc = EntityEncoder(
            entity_dim=40, embed_dim=64, n_heads=4, n_layers=2,
        ).to(device)
        bsz, n_ent = 4, 32
        features = torch.randn(bsz, n_ent, 40, device=device)
        mask = torch.ones(bsz, n_ent, dtype=torch.bool, device=device)
        mask[:, 20:] = False

        entity_emb, summary = enc(features, mask)
        assert entity_emb.shape == (bsz, n_ent, 64)
        assert summary.shape == (bsz, 64)

    def test_handles_all_padding(self, device):
        enc = EntityEncoder(entity_dim=40, embed_dim=64).to(device)
        bsz, n_ent = 2, 16
        features = torch.zeros(bsz, n_ent, 40, device=device)
        mask = torch.zeros(bsz, n_ent, dtype=torch.bool, device=device)

        entity_emb, summary = enc(features, mask)
        assert entity_emb.shape == (bsz, n_ent, 64)
        assert summary.shape == (bsz, 64)


class TestSpatialEncoder:
    def test_output_shapes(self, device):
        enc = SpatialEncoder(n_filters=64, n_blocks=3).to(device)
        spatial = torch.randn(2, SPATIAL_CHANNELS, ARENA_H, ARENA_W, device=device)

        feat_map, global_feat = enc(spatial)
        assert feat_map.shape == (2, 64, ARENA_H, ARENA_W)
        assert global_feat.shape == (2, 64)


class TestScalarEncoder:
    def test_output_shape(self, device):
        enc = ScalarEncoder(scalar_dim=SCALAR_FEATURES, out_dim=128).to(device)
        x = torch.randn(4, SCALAR_FEATURES, device=device)
        out = enc(x)
        assert out.shape == (4, 128)


class TestLSTMCore:
    def test_output_shapes(self, device):
        core = LSTMCore(input_dim=256, hidden_dim=256, n_layers=2).to(device)
        x = torch.randn(4, 256, device=device)

        out, hidden = core(x)
        assert out.shape == (4, 256)
        assert hidden[0].shape == (2, 4, 256)
        assert hidden[1].shape == (2, 4, 256)

    def test_with_hidden_state(self, device):
        core = LSTMCore(input_dim=256, hidden_dim=256, n_layers=2).to(device)
        x = torch.randn(4, 256, device=device)
        h = core.init_hidden(4, device)

        out1, h1 = core(x, h)
        out2, _h2 = core(x, h1)
        assert out2.shape == (4, 256)
        assert not torch.allclose(out1, out2)


class TestAutoregressivePolicyHead:
    def test_output_shapes(self, device):
        head = AutoregressivePolicyHead(hidden_dim=256).to(device)
        core_out = torch.randn(4, 256, device=device)

        card_logits, x_logits, y_logits = head(core_out)
        assert card_logits.shape == (4, 5)
        assert x_logits.shape == (4, ARENA_W)
        assert y_logits.shape == (4, ARENA_H)

    def test_with_teacher_forcing(self, device):
        head = AutoregressivePolicyHead(hidden_dim=256).to(device)
        core_out = torch.randn(4, 256, device=device)
        card_idx = torch.tensor([0, 1, 2, 3], device=device)
        x_pos = torch.tensor([5, 10, 3, 15], device=device)

        card_logits, x_logits, y_logits = head(core_out, card_idx, x_pos)
        assert card_logits.shape == (4, 5)
        assert x_logits.shape == (4, ARENA_W)
        assert y_logits.shape == (4, ARENA_H)

    def test_sample(self, device):
        head = AutoregressivePolicyHead(hidden_dim=256).to(device)
        core_out = torch.randn(1, 256, device=device)

        card_idx, x_pos, y_pos, log_prob = head.sample(core_out)
        assert 0 <= card_idx.item() <= 4
        assert 0 <= x_pos.item() < ARENA_W
        assert 0 <= y_pos.item() < ARENA_H
        assert log_prob.shape == (1,)


class TestAuxiliaryHeads:
    def test_output_shapes(self, device):
        heads = AuxiliaryHeads(hidden_dim=256).to(device)
        x = torch.randn(4, 256, device=device)

        crown_logits, tower_hp, game_length = heads(x)
        assert crown_logits.shape == (4, 7)
        assert tower_hp.shape == (4, 6)
        assert game_length.shape == (4,)


class TestDynamicsModel:
    def test_output_shapes(self, device):
        dyn = DynamicsModel(hidden_dim=256, action_dim=32).to(device)
        hidden_state = torch.randn(4, 256, device=device)
        action_oh = torch.randn(4, 5 + ARENA_W + ARENA_H, device=device)

        next_state, reward = dyn(hidden_state, action_oh)
        assert next_state.shape == (4, 256)
        assert reward.shape == (4,)

    def test_residual_connection(self, device):
        dyn = DynamicsModel(hidden_dim=256).to(device)
        hidden_state = torch.randn(4, 256, device=device)
        action_oh = torch.randn(4, 5 + ARENA_W + ARENA_H, device=device)

        next_state, _ = dyn(hidden_state, action_oh)
        assert not torch.allclose(next_state, hidden_state)


class TestCRStarNet:
    def test_forward_basic(self, device):
        model = CRStarNet(
            spatial_filters=32,
            spatial_blocks=2,
            entity_embed_dim=32,
            core_hidden=64,
            core_layers=1,
        ).to(device)

        bsz = 2
        spatial = torch.randn(
            bsz, SPATIAL_CHANNELS, ARENA_H, ARENA_W, device=device,
        )
        scalar = torch.randn(bsz, SCALAR_FEATURES, device=device)

        policy_logits, value, extras = model(spatial, scalar)
        assert policy_logits.shape == (bsz, ACTION_SPACE_SIZE)
        assert value.shape == (bsz,)
        assert "card_logits" in extras
        assert "crown_logits" in extras
        assert extras["card_logits"].shape == (bsz, 5)

    def test_with_action_mask(self, device):
        model = CRStarNet(
            spatial_filters=32, spatial_blocks=2,
            entity_embed_dim=32, core_hidden=64, core_layers=1,
        ).to(device)

        bsz = 2
        spatial = torch.randn(
            bsz, SPATIAL_CHANNELS, ARENA_H, ARENA_W, device=device,
        )
        scalar = torch.randn(bsz, SCALAR_FEATURES, device=device)
        mask = torch.zeros(
            bsz, ACTION_SPACE_SIZE, dtype=torch.bool, device=device,
        )
        mask[:, :10] = True

        policy_logits, value, _ = model(spatial, scalar, action_mask=mask)
        assert policy_logits[:, 10:].max() < -1e8

    def test_with_entity_features(self, device):
        model = CRStarNet(
            spatial_filters=32, spatial_blocks=2,
            entity_embed_dim=32, core_hidden=64, core_layers=1,
        ).to(device)

        bsz, n_ent = 2, 16
        spatial = torch.randn(
            bsz, SPATIAL_CHANNELS, ARENA_H, ARENA_W, device=device,
        )
        scalar = torch.randn(bsz, SCALAR_FEATURES, device=device)
        entity_feat = torch.randn(bsz, n_ent, 40, device=device)
        entity_mask = torch.ones(
            bsz, n_ent, dtype=torch.bool, device=device,
        )

        policy_logits, value, extras = model(
            spatial, scalar,
            entity_features=entity_feat,
            entity_mask=entity_mask,
        )
        assert policy_logits.shape == (bsz, ACTION_SPACE_SIZE)
        assert value.shape == (bsz,)

    def test_predict_compatibility(self, device):
        model = CRStarNet(
            spatial_filters=32, spatial_blocks=2,
            entity_embed_dim=32, core_hidden=64, core_layers=1,
        ).to(device)

        bsz = 1
        spatial = torch.randn(
            bsz, SPATIAL_CHANNELS, ARENA_H, ARENA_W, device=device,
        )
        scalar = torch.randn(bsz, SCALAR_FEATURES, device=device)

        policy, value = model.predict(spatial, scalar)
        assert policy.shape == (bsz, ACTION_SPACE_SIZE)
        assert value.shape == (bsz,)
        assert abs(policy.sum().item() - 1.0) < 0.01

    def test_encode(self, device):
        model = CRStarNet(
            spatial_filters=32, spatial_blocks=2,
            entity_embed_dim=32, core_hidden=64, core_layers=1,
        ).to(device)

        bsz = 2
        spatial = torch.randn(
            bsz, SPATIAL_CHANNELS, ARENA_H, ARENA_W, device=device,
        )
        scalar = torch.randn(bsz, SCALAR_FEATURES, device=device)

        core_out, hidden = model.encode(spatial, scalar)
        assert core_out.shape == (bsz, 64)
        assert hidden[0].shape[0] == 1
        assert hidden[0].shape[1] == bsz

    def test_parameter_count(self):
        model = CRStarNet(
            spatial_filters=128, spatial_blocks=10,
            entity_embed_dim=128, core_hidden=512, core_layers=2,
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < n_params < 200_000_000
