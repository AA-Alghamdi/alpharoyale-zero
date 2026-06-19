"""Tests for the MCTS implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from crsim.constants import ACTION_SPACE_SIZE
from crsim.game import CRGame
from mcts.search import MCTSConfig, MCTSNode, MCTSPlayer, _action_id_to_action
from model.network import CRZeroNet


def test_mcts_node_creation():
    """MCTS nodes initialize correctly."""
    node = MCTSNode()
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert not node.is_expanded
    assert len(node.children) == 0


def test_mcts_node_expand():
    """Node expansion creates children for valid actions."""
    node = MCTSNode()
    priors = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

    # Only 3 valid actions
    mask[0] = True
    mask[100] = True
    mask[ACTION_SPACE_SIZE - 1] = True

    priors[0] = 0.5
    priors[100] = 0.3
    priors[ACTION_SPACE_SIZE - 1] = 0.2

    node.expand(priors, mask)
    assert node.is_expanded
    assert len(node.children) == 3


def test_mcts_ucb():
    """UCB score favors less-visited nodes."""
    parent = MCTSNode()
    parent.visit_count = 100

    child_a = MCTSNode(parent=parent, prior=0.5)
    child_a.visit_count = 50
    child_a.value_sum = 25.0  # Q = 0.5

    child_b = MCTSNode(parent=parent, prior=0.5)
    child_b.visit_count = 1
    child_b.value_sum = 0.5  # Q = 0.5

    # child_b should have higher UCB (less visited, higher exploration bonus)
    assert child_b.ucb_score(100, 2.5) > child_a.ucb_score(100, 2.5)


def test_mcts_search_returns_valid():
    """MCTS search returns a valid probability distribution."""
    model = CRZeroNet(n_res_blocks=2, n_filters=32)
    model.eval()

    config = MCTSConfig(n_simulations=10)  # very few for speed
    player = MCTSPlayer(model=model, config=config, device=torch.device("cpu"))

    game = CRGame(seed=42)
    action_probs = player.search(game, player=0)

    assert action_probs.shape == (ACTION_SPACE_SIZE,)
    assert abs(action_probs.sum() - 1.0) < 1e-4
    assert (action_probs >= 0).all()


def test_mcts_select_action():
    """select_action returns a valid action id."""
    model = CRZeroNet(n_res_blocks=2, n_filters=32)
    model.eval()

    config = MCTSConfig(n_simulations=10)
    player = MCTSPlayer(model=model, config=config, device=torch.device("cpu"))

    game = CRGame(seed=42)
    action_id, probs = player.select_action(game, player=0)

    assert 0 <= action_id < ACTION_SPACE_SIZE
    # Action should be valid
    mask = game.get_valid_actions_mask(0)
    assert mask[action_id], f"Selected invalid action {action_id}"


def test_action_id_conversion():
    """Action ID ↔ Action conversion is consistent."""
    from crsim.constants import ARENA_H, ARENA_W, WAIT_ACTION

    # WAIT
    a = _action_id_to_action(WAIT_ACTION, 0)
    assert a.is_wait
    assert a.player == 0

    # Card placement
    slot, x, y = 2, 9, 5
    action_id = slot * ARENA_W * ARENA_H + x * ARENA_H + y
    a = _action_id_to_action(action_id, 1)
    assert a.player == 1
    assert a.hand_slot == slot
    assert a.x == float(x)
    assert a.y == float(y)


if __name__ == "__main__":
    test_mcts_node_creation()
    test_mcts_node_expand()
    test_mcts_ucb()
    test_mcts_search_returns_valid()
    test_mcts_select_action()
    test_action_id_conversion()
    print("All MCTS tests passed!")
