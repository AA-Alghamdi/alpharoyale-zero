"""Next-generation network: Entity Transformer + Spatial ResNet + LSTM core.

Inspired by AlphaStar's architecture adapted for Clash Royale:
  - Entity Transformer: self-attention over variable-length entity lists
  - Spatial ResNet: conv tower for map/placement features
  - LSTM core: temporal memory across game ticks
  - Autoregressive policy head: card → x → y (not flat 2304-way)
  - Auxiliary heads: crown prediction, tower HP, game length (KataGo-style)
  - Dynamics model head: predict next latent state (EfficientZero-style)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f_nn  # noqa: N812

from crsim.constants import (
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    NUM_HAND_SLOTS,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
)

# Entity feature dim: [type_embed(32) + owner(1) + pos(2) + hp(2) + flags(3)]
ENTITY_FEATURE_DIM = 40
MAX_ENTITIES = 64
NUM_ENTITY_TYPES = 128  # card/building/spell type IDs


class EntityEncoder(nn.Module):
    """Transformer encoder for variable-length entity lists.

    Each entity is represented as a fixed-size feature vector. The encoder
    produces a fixed-size summary via mean-pooling + a learned [CLS] token.
    """

    def __init__(
        self,
        entity_dim: int = ENTITY_FEATURE_DIM,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Type embedding
        self.type_embed = nn.Embedding(NUM_ENTITY_TYPES, 32)

        # Project raw entity features to embed_dim
        self.input_proj = nn.Linear(entity_dim, embed_dim)

        # Positional encoding (learned, up to MAX_ENTITIES)
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_ENTITIES + 1, embed_dim) * 0.02)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode entities.

        Parameters
        ----------
        entity_features : (B, N, entity_dim) — raw entity features
        entity_mask : (B, N) bool — True for real entities, False for padding

        Returns
        -------
        entity_embeddings : (B, N, embed_dim)
        summary : (B, embed_dim) — aggregated entity representation
        """
        b, n, _ = entity_features.shape

        # Project to embed_dim
        x = self.input_proj(entity_features)  # (b, n, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(b, -1, -1)  # (b, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)  # (b, n+1, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed[:, : n + 1, :]

        # Build attention mask (True = ignore)
        cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
        attn_mask = torch.cat([cls_mask, ~entity_mask], dim=1)  # (b, n+1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = self.norm(x)

        summary = x[:, 0, :]  # CLS token output
        entity_embeddings = x[:, 1:, :]  # (B, N, embed_dim)

        return entity_embeddings, summary


class SpatialEncoder(nn.Module):
    """Residual conv tower for spatial features with global pooling (KataGo-style)."""

    def __init__(
        self,
        in_channels: int = SPATIAL_CHANNELS,
        n_filters: int = 128,
        n_blocks: int = 10,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            _SpatialResBlock(n_filters) for _ in range(n_blocks)
        ])
        # Global pooling branch (KataGo-style)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_filters, n_filters),
            nn.ReLU(),
        )

    def forward(self, spatial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode spatial features.

        Returns
        -------
        feature_map : (B, F, H, W)
        global_features : (B, F) — KataGo-style global context
        """
        x = self.input_conv(spatial)
        for block in self.blocks:
            x = block(x)
        global_feat = self.global_pool(x)
        return x, global_feat


class _SpatialResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # SE block
        mid = max(channels // 8, 1)
        self.se_fc1 = nn.Linear(channels, mid)
        self.se_fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = f_nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # SE
        se = out.mean(dim=(2, 3))
        se = f_nn.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se.unsqueeze(-1).unsqueeze(-1)
        return f_nn.relu(out + residual)


class ScalarEncoder(nn.Module):
    """Encode scalar game features (elixir, time, hand cards, tower status)."""

    def __init__(self, scalar_dim: int = SCALAR_FEATURES, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scalar_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        return self.net(scalar)


class LSTMCore(nn.Module):
    """Recurrent core for temporal context across game ticks."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, n_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x : (B, input_dim) — current timestep features
        hidden : (h, c) each (n_layers, B, hidden_dim)

        Returns
        -------
        output : (B, hidden_dim)
        new_hidden : (h, c)
        """
        x = x.unsqueeze(1)  # (B, 1, input_dim)
        if hidden is None:
            out, new_hidden = self.lstm(x)
        else:
            out, new_hidden = self.lstm(x, hidden)
        out = out.squeeze(1)  # (B, hidden_dim)
        out = self.norm(out)
        return out, new_hidden

    def init_hidden(
        self, batch_size: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return h, c


class AutoregressivePolicyHead(nn.Module):
    """Decomposed action head: card_slot → x_position → y_position.

    Instead of a flat 2305-way softmax, decomposes the action into three
    sequential decisions. This is much easier to learn and scales better.
    """

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()

        # Card selection (0-3 = play card, 4 = wait)
        self.card_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_HAND_SLOTS + 1),  # 5-way: 4 cards + wait
        )

        # X position head (conditioned on card embedding)
        self.card_embed = nn.Embedding(NUM_HAND_SLOTS + 1, 32)
        self.x_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, ARENA_W),  # 18-way
        )

        # Y position head (conditioned on card + x)
        self.x_embed = nn.Embedding(ARENA_W, 16)
        self.y_head = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, ARENA_H),  # 32-way
        )

    def forward(
        self,
        core_out: torch.Tensor,
        card_idx: torch.Tensor | None = None,
        x_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all three logit distributions.

        During training, card_idx and x_pos are provided (teacher forcing).
        During inference, sample sequentially.

        Returns
        -------
        card_logits : (B, 5)
        x_logits : (B, 18)
        y_logits : (B, 32)
        """
        card_logits = self.card_head(core_out)  # (B, 5)

        # For x/y heads, use teacher forcing if available
        if card_idx is None:
            card_idx = card_logits.argmax(dim=-1)  # (B,)

        card_emb = self.card_embed(card_idx)  # (B, 32)
        x_input = torch.cat([core_out, card_emb], dim=-1)
        x_logits = self.x_head(x_input)  # (B, 18)

        if x_pos is None:
            x_pos = x_logits.argmax(dim=-1)  # (B,)

        x_emb = self.x_embed(x_pos)  # (B, 16)
        y_input = torch.cat([core_out, card_emb, x_emb], dim=-1)
        y_logits = self.y_head(y_input)  # (B, 32)

        return card_logits, x_logits, y_logits

    def sample(
        self,
        core_out: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[int, int, int, torch.Tensor]:
        """Sample an action autoregressively.

        Returns (card_idx, x, y, log_prob).
        """
        card_logits = self.card_head(core_out)
        card_probs = f_nn.softmax(card_logits / temperature, dim=-1)
        card_idx = torch.multinomial(card_probs, 1).squeeze(-1)

        card_emb = self.card_embed(card_idx)
        x_logits = self.x_head(torch.cat([core_out, card_emb], dim=-1))
        x_probs = f_nn.softmax(x_logits / temperature, dim=-1)
        x_pos = torch.multinomial(x_probs, 1).squeeze(-1)

        x_emb = self.x_embed(x_pos)
        y_logits = self.y_head(torch.cat([core_out, card_emb, x_emb], dim=-1))
        y_probs = f_nn.softmax(y_logits / temperature, dim=-1)
        y_pos = torch.multinomial(y_probs, 1).squeeze(-1)

        # Total log probability
        log_prob = (
            torch.log(card_probs.gather(1, card_idx.unsqueeze(1)) + 1e-8)
            + torch.log(x_probs.gather(1, x_pos.unsqueeze(1)) + 1e-8)
            + torch.log(y_probs.gather(1, y_pos.unsqueeze(1)) + 1e-8)
        ).squeeze(-1)

        return card_idx, x_pos, y_pos, log_prob


class ValueHead(nn.Module):
    """Scalar value head: win probability in [-1, 1]."""

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AuxiliaryHeads(nn.Module):
    """KataGo-style auxiliary prediction targets for richer gradients.

    Predicts:
    - Crown difference (-3 to +3)
    - Tower HP fractions (6 values: our king, our L princess, our R princess, ...)
    - Game length remaining (0-1)
    """

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        # Crown prediction: 7-class classification (-3 to +3)
        self.crown_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )
        # Tower HP regression (6 values)
        self.tower_hp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid(),
        )
        # Game length remaining (scalar 0-1)
        self.length_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crown_logits = self.crown_head(x)
        tower_hp = self.tower_hp_head(x)
        game_length = self.length_head(x).squeeze(-1)
        return crown_logits, tower_hp, game_length


class DynamicsModel(nn.Module):
    """EfficientZero-style dynamics model for latent-space planning.

    Given current latent state + action → predicts next latent state + reward.
    """

    def __init__(self, hidden_dim: int = 512, action_dim: int = 64) -> None:
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(NUM_HAND_SLOTS + 1 + ARENA_W + ARENA_H, action_dim),
            nn.ReLU(),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward.

        Parameters
        ----------
        hidden_state : (B, hidden_dim)
        action_onehot : (B, card_oh + x_oh + y_oh)

        Returns
        -------
        next_state : (B, hidden_dim)
        reward : (B,)
        """
        action_emb = self.action_encoder(action_onehot)
        combined = torch.cat([hidden_state, action_emb], dim=-1)
        next_state = self.norm(self.dynamics(combined) + hidden_state)  # residual
        reward = self.reward_head(next_state).squeeze(-1)
        return next_state, reward


BELIEF_FEATURE_DIM = 128


class OpponentModelHead(nn.Module):
    """Predict opponent's likely next action / deck composition from belief state."""

    def __init__(self, hidden_dim: int = 512, n_card_types: int = 42) -> None:
        super().__init__()
        self.deck_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_card_types),
            nn.Sigmoid(),
        )
        self.next_card_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_card_types),
        )
        self.elixir_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, core_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deck_probs = self.deck_predictor(core_out)
        next_card_logits = self.next_card_predictor(core_out)
        elixir_est = self.elixir_predictor(core_out).squeeze(-1)
        return deck_probs, next_card_logits, elixir_est


class CRStarNet(nn.Module):
    """Full next-gen ClashRoyale network combining all components.

    Architecture overview:
      Entity list → EntityEncoder → entity_summary (128-d)
      Spatial map  → SpatialEncoder → spatial_summary (128-d) + feature_map
      Scalars      → ScalarEncoder → scalar_emb (128-d)
      Belief state → BeliefEncoder → belief_emb (64-d)
      [entity_summary ‖ spatial_summary ‖ scalar_emb ‖ belief_emb] → LSTMCore → core_out (512-d)
      core_out → AutoregressivePolicyHead → (card, x, y)
      core_out → ValueHead → win probability
      core_out → AuxiliaryHeads → crowns, tower HP, game length
      core_out → OpponentModelHead → opponent deck/card/elixir predictions
      core_out + action → DynamicsModel → next_state, reward
    """

    def __init__(
        self,
        entity_dim: int = ENTITY_FEATURE_DIM,
        entity_embed_dim: int = 128,
        spatial_filters: int = 128,
        spatial_blocks: int = 10,
        scalar_dim: int = SCALAR_FEATURES,
        core_hidden: int = 512,
        core_layers: int = 2,
        belief_dim: int = BELIEF_FEATURE_DIM,
    ) -> None:
        super().__init__()

        self.entity_encoder = EntityEncoder(
            entity_dim=entity_dim,
            embed_dim=entity_embed_dim,
        )
        self.spatial_encoder = SpatialEncoder(
            n_filters=spatial_filters,
            n_blocks=spatial_blocks,
        )
        self.scalar_encoder = ScalarEncoder(
            scalar_dim=scalar_dim,
            out_dim=entity_embed_dim,
        )

        # Belief state encoder for opponent modeling
        belief_out = 64
        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_dim, 128),
            nn.ReLU(),
            nn.Linear(128, belief_out),
            nn.ReLU(),
        )

        # Fusion: entity(128) + spatial_global(128) + scalar(128) + belief(64) = 448
        fusion_dim = entity_embed_dim + spatial_filters + entity_embed_dim + belief_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, core_hidden),
            nn.ReLU(),
        )

        self.core = LSTMCore(
            input_dim=core_hidden,
            hidden_dim=core_hidden,
            n_layers=core_layers,
        )

        self.policy_head = AutoregressivePolicyHead(hidden_dim=core_hidden)
        self.value_head = ValueHead(hidden_dim=core_hidden)
        self.aux_heads = AuxiliaryHeads(hidden_dim=core_hidden)
        self.dynamics = DynamicsModel(hidden_dim=core_hidden)
        self.opponent_head = OpponentModelHead(hidden_dim=core_hidden)

        # Also keep a flat policy head for compatibility with existing MCTS
        self.flat_policy_head = nn.Sequential(
            nn.Linear(core_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, ACTION_SPACE_SIZE),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        entity_features: torch.Tensor | None = None,
        entity_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        belief_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode observations into latent state.

        Returns
        -------
        core_out : (B, hidden_dim) — latent representation
        new_hidden : LSTM hidden state
        """
        bsz = spatial.shape[0]
        device = spatial.device

        # Spatial encoding
        _, spatial_global = self.spatial_encoder(spatial)

        # Entity encoding
        if entity_features is not None and entity_mask is not None:
            _, entity_summary = self.entity_encoder(entity_features, entity_mask)
        else:
            entity_summary = torch.zeros(
                bsz, self.entity_encoder.embed_dim, device=device,
            )

        # Scalar encoding
        scalar_emb = self.scalar_encoder(scalar)

        # Belief state encoding
        if belief_features is not None:
            belief_emb = self.belief_encoder(belief_features)
        else:
            belief_emb = torch.zeros(bsz, 64, device=device)

        # Fuse all modalities
        fused = torch.cat(
            [entity_summary, spatial_global, scalar_emb, belief_emb], dim=-1,
        )
        fused = self.fusion(fused)

        # LSTM core
        core_out, new_hidden = self.core(fused, hidden)

        return core_out, new_hidden

    def forward(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        entity_features: torch.Tensor | None = None,
        entity_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        belief_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Full forward pass.

        Returns
        -------
        policy_logits : (B, ACTION_SPACE_SIZE) — flat policy for MCTS compat
        value : (B,)
        extras : dict with auxiliary outputs
        """
        core_out, new_hidden = self.encode(
            spatial, scalar, entity_features, entity_mask, hidden,
            belief_features=belief_features,
        )

        # Flat policy (for MCTS compatibility)
        policy_logits = self.flat_policy_head(core_out)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)

        # Value
        value = self.value_head(core_out)

        # Auxiliary targets
        crown_logits, tower_hp, game_length = self.aux_heads(core_out)

        # Autoregressive policy (for training)
        card_logits, x_logits, y_logits = self.policy_head(core_out)

        # Opponent modeling predictions
        opp_deck, opp_next_card, opp_elixir = self.opponent_head(core_out)

        extras = {
            "hidden": new_hidden,
            "core_out": core_out,
            "card_logits": card_logits,
            "x_logits": x_logits,
            "y_logits": y_logits,
            "crown_logits": crown_logits,
            "tower_hp_pred": tower_hp,
            "game_length_pred": game_length,
            "opp_deck_pred": opp_deck,
            "opp_next_card_logits": opp_next_card,
            "opp_elixir_pred": opp_elixir,
        }

        return policy_logits, value, extras

    def predict(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        entity_features: torch.Tensor | None = None,
        entity_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper compatible with existing MCTSPlayer.

        Now passes entity features and hidden state through properly.
        """
        with torch.no_grad():
            logits, value, _ = self.forward(
                spatial, scalar, action_mask,
                entity_features=entity_features,
                entity_mask=entity_mask,
                hidden=hidden,
            )
            policy = f_nn.softmax(logits, dim=-1)
        return policy, value

    def predict_with_state(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        entity_features: torch.Tensor | None = None,
        entity_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Full inference returning policy, value, and all extras including hidden state.

        Use this when you need to maintain LSTM state across timesteps.
        """
        with torch.no_grad():
            logits, value, extras = self.forward(
                spatial, scalar, action_mask,
                entity_features=entity_features,
                entity_mask=entity_mask,
                hidden=hidden,
            )
            policy = f_nn.softmax(logits, dim=-1)
        return policy, value, extras

    def predict_autoregressive(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        entity_features: torch.Tensor | None = None,
        entity_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        temperature: float = 1.0,
        belief_features: torch.Tensor | None = None,
    ) -> tuple[int, int, int, torch.Tensor, dict]:
        """Autoregressive inference: sample card→x→y sequentially.

        Returns (card_idx, x_pos, y_pos, log_prob, extras).
        This is the proper way to generate actions during play.
        """
        with torch.no_grad():
            core_out, new_hidden = self.encode(
                spatial, scalar, entity_features, entity_mask, hidden,
                belief_features=belief_features,
            )
            card_idx, x_pos, y_pos, log_prob = self.policy_head.sample(
                core_out, temperature=temperature,
            )
            value = self.value_head(core_out)
            extras = {
                "hidden": new_hidden,
                "core_out": core_out,
                "value": value,
            }
        return (
            int(card_idx.item()),
            int(x_pos.item()),
            int(y_pos.item()),
            log_prob,
            extras,
        )
