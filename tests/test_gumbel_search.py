"""Tests for the Gumbel MuZero search."""

from __future__ import annotations

import numpy as np
import pytest

from crsim.cards import CardType
from crsim.constants import ACTION_SPACE_SIZE
from crsim.game import CRGame
from mcts.gumbel_search import (
    GumbelConfig,
    GumbelMuZeroSearch,
    GumbelNode,
    _improved_policy,
    _sequential_halving_schedule,
)
from model.network import CRZeroNet


@pytest.fixture
def model():
    return CRZeroNet(n_res_blocks=2, n_filters=32, use_se=False)


@pytest.fixture
def game():
    deck = list(CardType)[:8]
    return CRGame(deck_p0=deck, deck_p1=deck, seed=42)


class TestGumbelNode:
    def test_init(self):
        node = GumbelNode(action=5, prior=0.3)
        assert node.action == 5
        assert node.prior == 0.3
        assert node.visit_count == 0
        assert node.q_value == 0.0

    def test_q_value(self):
        node = GumbelNode()
        node.visit_count = 10
        node.value_sum = 5.0
        assert abs(node.q_value - 0.5) < 1e-6

    def test_expand(self):
        node = GumbelNode()
        priors = np.array([0.3, 0.5, 0.2])
        mask = np.array([True, True, False])
        node.expand(priors, mask)
        assert node.is_expanded
        assert len(node.children) == 2
        assert node.children[0].prior == 0.3
        assert node.children[1].prior == 0.5


class TestSequentialHalving:
    def test_basic_schedule(self):
        schedule = _sequential_halving_schedule(16, 4)
        assert len(schedule) > 0
        assert all(s > 0 for s in schedule)

    def test_single_action(self):
        schedule = _sequential_halving_schedule(16, 1)
        assert schedule == [16]

    def test_many_actions(self):
        schedule = _sequential_halving_schedule(32, 16)
        assert len(schedule) > 0


class TestImprovedPolicy:
    def test_basic(self):
        root = GumbelNode()
        root.is_expanded = True

        for i in range(3):
            child = GumbelNode(action=i, prior=1.0 / 3)
            child.visit_count = 5
            child.value_sum = [2.5, 1.0, 4.0][i]
            root.children.append(child)

        policy = _improved_policy(root, c_visit=50.0, c_scale=1.0, n_actions=5)
        assert policy.shape == (5,)
        assert abs(policy.sum() - 1.0) < 1e-5
        # Action 2 has highest Q-value, should have highest probability
        assert policy[2] > policy[0] > policy[1]


class TestGumbelMuZeroSearch:
    def test_search_returns_valid_probs(self, model, game):
        searcher = GumbelMuZeroSearch(
            model=model,
            config=GumbelConfig(
                n_simulations=4,
                max_considered_actions=4,
                playout_cap_randomization=False,
            ),
        )

        probs = searcher.search(game, player=0)
        assert probs.shape == (ACTION_SPACE_SIZE,)
        assert abs(probs.sum() - 1.0) < 0.01
        assert np.all(probs >= 0)

    def test_select_action(self, model, game):
        searcher = GumbelMuZeroSearch(
            model=model,
            config=GumbelConfig(
                n_simulations=4,
                max_considered_actions=4,
                playout_cap_randomization=False,
            ),
        )

        action_id, probs = searcher.select_action(game, player=0)
        assert 0 <= action_id < ACTION_SPACE_SIZE
        assert probs.shape == (ACTION_SPACE_SIZE,)

    def test_deterministic_select(self, model, game):
        searcher = GumbelMuZeroSearch(
            model=model,
            config=GumbelConfig(
                n_simulations=4,
                max_considered_actions=4,
                playout_cap_randomization=False,
            ),
        )

        action_id, probs = searcher.select_action(
            game, player=0, deterministic=True,
        )
        assert action_id == int(np.argmax(probs))

    def test_playout_cap_randomization(self, model, game):
        searcher = GumbelMuZeroSearch(
            model=model,
            config=GumbelConfig(
                n_simulations=16,
                playout_cap_randomization=True,
                playout_cap_min=2,
                playout_cap_max=8,
            ),
        )

        # Should work without error
        probs = searcher.search(game, player=0)
        assert probs.shape == (ACTION_SPACE_SIZE,)

    def test_both_players(self, model, game):
        """Search should work for both player 0 and player 1."""
        searcher = GumbelMuZeroSearch(
            model=model,
            config=GumbelConfig(
                n_simulations=4, playout_cap_randomization=False,
            ),
        )

        probs_0 = searcher.search(game, player=0)
        probs_1 = searcher.search(game, player=1)
        assert probs_0.shape == probs_1.shape == (ACTION_SPACE_SIZE,)
