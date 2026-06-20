"""Stat-derived card embeddings (Phase-1 AlphaStar-style scaffolding).

A pure id-lookup table (``nn.Embedding(NUM_CARD_TYPES, d)``) cannot embed a card
it has never seen during training: an unseen id has a randomly-initialized,
meaningless row. This module makes the embedding *stat-derived* so a brand-new
card can be placed in roughly the right region of embedding space from its stats
alone (near-zero-shot), while still allowing the network to learn id-specific
corrections for the cards it does see.

Core formula (``+`` = element-wise sum of two same-dim ``embed_dim`` vectors)::

    card_emb(c) = stat_mlp(stat_vector(c)) + id_table[c]

- ``stat_mlp(stat_vector(c))`` generalizes across cards: similar stats map to
  similar embeddings, so an unseen card embeds sensibly from its stats.
- ``id_table[c]`` is a per-id learned correction (zero-shot path simply omits it).

Integration sketch (a follow-up PR, NOT done here to keep this self-contained)::

    # In an encoder that currently does `self.type_embed = nn.Embedding(...)`:
    self.card_emb = StatBasedCardEmbedding(embed_dim=32)
    # ... given a (B,) LongTensor of card type ids:
    type_vecs = self.card_emb(card_ids)            # (B, 32)
    # For a hypothetical / unseen card known only by its (normalized) stats:
    type_vecs = self.card_emb.forward_from_stats(stat_row)  # id component = 0

The full table is exposed via :meth:`StatBasedCardEmbedding.all_embeddings` for
callers that want to precompute every card's embedding once per forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from crsim.cards import CARD_DEFS, CardDef, CardType, EntityKind
from crsim.constants import NUM_CARD_TYPES

# Fixed, reproducible normalization divisors. Where a stat also appears in
# ``model/features.py`` we mirror its scale exactly (hp/5000, dps/500, speed/4,
# attack_range/10, cost/10, building timer/70) so embeddings live on the same
# scale as the rest of the encoder.
_HP_SCALE = 5000.0
_DPS_SCALE = 500.0
_DAMAGE_SCALE = 1000.0
_HIT_SPEED_SCALE = 5.0
_LOAD_TIME_SCALE = 5.0
_SPEED_SCALE = 4.0
_RANGE_SCALE = 10.0
_COST_SCALE = 10.0
_SPLASH_RADIUS_SCALE = 5.0
_SPAWN_COUNT_SCALE = 15.0
_DEPLOY_TIME_SCALE = 5.0
_COLLISION_RADIUS_SCALE = 2.0
_MASS_SCALE = 20.0
_PROJECTILE_SPEED_SCALE = 20.0
_BUILDING_LIFETIME_SCALE = 70.0
_SHIELD_HP_SCALE = 2000.0


def stat_vector(card_def: CardDef) -> np.ndarray:
    """Return the fixed, normalized stat vector for a single card.

    This is the single source of truth for the stat layout: both
    :func:`build_stat_matrix` (the seen-card path) and the unseen-card path
    (:meth:`StatBasedCardEmbedding.forward_from_stats`) build their inputs the
    same way. ``CardDef`` has no boolean ``is_building`` field, so it is derived
    from ``kind == EntityKind.BUILDING``.

    Returns
    -------
    np.ndarray, shape ``(STAT_DIM,)`` float32
    """
    is_building = card_def.kind == EntityKind.BUILDING
    return np.array(
        [
            # Continuous stats (normalized to roughly [0, 1]).
            card_def.cost / _COST_SCALE,
            card_def.hp / _HP_SCALE,
            card_def.dps / _DPS_SCALE,
            card_def.damage_per_hit / _DAMAGE_SCALE,
            card_def.hit_speed / _HIT_SPEED_SCALE,
            card_def.load_time / _LOAD_TIME_SCALE,
            card_def.speed / _SPEED_SCALE,
            card_def.attack_range / _RANGE_SCALE,
            card_def.sight_range / _RANGE_SCALE,
            card_def.splash_radius / _SPLASH_RADIUS_SCALE,
            card_def.spawn_count / _SPAWN_COUNT_SCALE,
            card_def.deploy_time / _DEPLOY_TIME_SCALE,
            card_def.collision_radius / _COLLISION_RADIUS_SCALE,
            card_def.mass / _MASS_SCALE,
            card_def.projectile_speed / _PROJECTILE_SPEED_SCALE,
            card_def.building_lifetime / _BUILDING_LIFETIME_SCALE,
            card_def.shield_hp / _SHIELD_HP_SCALE,
            # Boolean flags (0/1).
            float(card_def.is_flying),
            float(card_def.is_splash),
            float(is_building),
            float(card_def.has_evolution),
            float(card_def.is_champion),
            float(card_def.has_hero),
            float(card_def.has_shield),
            float(card_def.has_charge),
            float(card_def.stuns),
        ],
        dtype=np.float32,
    )


# Stat-vector width, derived from the layout above so the two never drift.
STAT_DIM: int = stat_vector(CARD_DEFS[CardType.KNIGHT]).shape[0]


def build_stat_matrix() -> np.ndarray:
    """Build the ``(NUM_CARD_TYPES, STAT_DIM)`` matrix of normalized stat rows.

    Row ``i`` is ``stat_vector(CARD_DEFS[CardType(i)])``. The matrix is fixed
    (a function of the card table only) and reproducible across runs.
    """
    matrix = np.zeros((NUM_CARD_TYPES, STAT_DIM), dtype=np.float32)
    for i in range(NUM_CARD_TYPES):
        matrix[i] = stat_vector(CARD_DEFS[CardType(i)])
    return matrix


class StatBasedCardEmbedding(nn.Module):
    """Learned, stat-derived card embedding with an id-specific correction.

    ``card_emb(c) = stat_mlp(stat_vector(c)) + id_table[c]``

    The precomputed stat matrix is held as a (non-trainable) buffer so that
    ``module.to(device)`` moves it alongside the parameters.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        stat_dim: int | None = None,
    ) -> None:
        super().__init__()
        stat_matrix = build_stat_matrix()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Defaults to the built-in STAT_DIM; the precomputed buffer (and hence
        # the seen-card `forward` path) requires stat_dim == STAT_DIM.
        self.stat_dim = stat_dim if stat_dim is not None else stat_matrix.shape[1]

        self.register_buffer(
            "stat_matrix",
            torch.from_numpy(stat_matrix).to(torch.float32),
        )

        self.id_table = nn.Embedding(NUM_CARD_TYPES, embed_dim)
        self.stat_mlp = nn.Sequential(
            nn.Linear(self.stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        """Embed a batch of (seen) card ids.

        Parameters
        ----------
        card_ids : LongTensor, any shape ``(...)`` of valid ``CardType`` ids.

        Returns
        -------
        torch.Tensor, shape ``(..., embed_dim)``.
        """
        stat_rows = self.stat_matrix[card_ids]
        return self.stat_mlp(stat_rows) + self.id_table(card_ids)

    def forward_from_stats(
        self,
        stat_vec: torch.Tensor,
        id_component: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed an UNSEEN card from a raw (normalized) stat vector.

        This is the key generalization path: a card with no entry in the id
        table — e.g. a brand-new release, or a hypothetical card during search —
        is embedded through the stat MLP alone. Build ``stat_vec`` with the same
        normalization as :func:`stat_vector` (use that helper, then convert to a
        tensor) so it lands on the trained scale.

        Parameters
        ----------
        stat_vec : Tensor, shape ``(..., stat_dim)`` — normalized stat vector(s).
        id_component : optional Tensor broadcastable to ``(..., embed_dim)``.
            The id-table contribution. Defaults to zero (pure stat-derived
            embedding) since an unseen card has no id row.

        Returns
        -------
        torch.Tensor, shape ``(..., embed_dim)``.
        """
        stat_emb = self.stat_mlp(stat_vec)
        if id_component is None:
            return stat_emb
        return stat_emb + id_component

    def all_embeddings(self) -> torch.Tensor:
        """Return embeddings for the full card table, shape ``(NUM_CARD_TYPES, embed_dim)``."""
        ids = torch.arange(NUM_CARD_TYPES, device=self.stat_matrix.device)
        return self.forward(ids)
