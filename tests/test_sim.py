"""Tests for the game simulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from crsim.cards import CARD_DEFS
from crsim.constants import (
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    KING_TOWER_HP,
    MAX_ELIXIR,
    PRINCESS_TOWER_HP,
    STARTING_ELIXIR,
    WAIT_ACTION,
)
from crsim.game import Action, CRGame, GamePhase, GameResult


def test_game_creation():
    """Game initializes with correct state."""
    game = CRGame(seed=42)
    assert game.phase == GamePhase.REGULAR
    assert game.result == GameResult.IN_PROGRESS
    assert game.tick_count == 0
    assert len(game.players) == 2
    for ps in game.players:
        assert abs(ps.elixir - STARTING_ELIXIR) < 0.01
        assert len(ps.hand) == 4
        assert len(ps.deck) == 8

    # 6 towers should exist
    towers = [e for e in game.entities if e.is_tower]
    assert len(towers) == 6
    kings = [e for e in towers if e.is_king_tower]
    assert len(kings) == 2


def test_tower_hp():
    """Towers have correct HP."""
    game = CRGame(seed=0)
    for e in game.entities:
        if not e.is_tower:
            continue
        if e.is_king_tower:
            assert e.hp == KING_TOWER_HP
        else:
            assert e.hp == PRINCESS_TOWER_HP


def test_wait_action():
    """Game can advance with both players waiting."""
    game = CRGame(seed=1)
    initial_elixir = [ps.elixir for ps in game.players]
    game.step([
        Action(player=0, hand_slot=-1),
        Action(player=1, hand_slot=-1),
    ])
    assert game.tick_count == 1
    # Elixir should have increased
    for i, ps in enumerate(game.players):
        assert ps.elixir > initial_elixir[i]


def test_elixir_cap():
    """Elixir doesn't exceed MAX_ELIXIR."""
    game = CRGame(seed=2)
    game.players[0].elixir = 9.9
    game.players[1].elixir = 9.9
    for _ in range(100):
        game.step([
            Action(player=0, hand_slot=-1),
            Action(player=1, hand_slot=-1),
        ])
    for ps in game.players:
        assert ps.elixir <= MAX_ELIXIR


def test_valid_actions_mask():
    """Valid actions mask has correct shape and includes WAIT."""
    game = CRGame(seed=3)
    mask = game.get_valid_actions_mask(0)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask[WAIT_ACTION]  # wait is always valid
    # With starting elixir (5), at least some card placements should be valid
    assert mask.sum() > 1


def test_card_placement():
    """Placing a card deducts elixir and spawns an entity."""
    game = CRGame(seed=4)
    initial_entity_count = len([e for e in game.entities if not e.is_tower])
    assert initial_entity_count == 0

    # Find a card in hand that costs <= starting elixir
    ps = game.players[0]
    slot = 0
    card_type = ps.deck[ps.hand[slot]]
    CARD_DEFS[card_type]  # verify card is valid

    game.apply_action(Action(player=0, hand_slot=slot, x=9.0, y=5.0))

    assert ps.elixir < STARTING_ELIXIR
    non_tower = [e for e in game.entities if not e.is_tower]
    assert len(non_tower) > 0


def test_game_runs_to_completion():
    """A game with random actions always terminates within the time limit."""
    from crsim.constants import TOTAL_MAX_TICKS

    game = CRGame(seed=5)
    rng = np.random.default_rng(5)
    # Step a few ticks beyond the hard time limit; the game must have ended.
    max_ticks = TOTAL_MAX_TICKS + 10

    for _ in range(max_ticks):
        if game.done:
            break

        actions = []
        for player in (0, 1):
            mask = game.get_valid_actions_mask(player)
            valid_ids = np.where(mask)[0]
            action_id = int(rng.choice(valid_ids))

            if action_id == WAIT_ACTION:
                actions.append(Action(player=player, hand_slot=-1))
            else:
                slot = action_id // (ARENA_W * ARENA_H)
                rem = action_id % (ARENA_W * ARENA_H)
                x = rem // ARENA_H
                y = rem % ARENA_H
                actions.append(Action(player=player, hand_slot=slot, x=float(x), y=float(y)))

        game.step(actions)

    # By the hard time limit the game must have ended with a definite result.
    assert game.done
    assert game.tick_count <= TOTAL_MAX_TICKS
    assert game.result != GameResult.IN_PROGRESS


def test_clone():
    """Cloned game is independent of original."""
    game = CRGame(seed=6)
    game.step([Action(player=0, hand_slot=-1), Action(player=1, hand_slot=-1)])

    clone = game.clone()
    assert clone.tick_count == game.tick_count

    # Advance clone
    clone.step([Action(player=0, hand_slot=-1), Action(player=1, hand_slot=-1)])
    assert clone.tick_count == game.tick_count + 1
    assert game.tick_count == 1  # original unchanged


if __name__ == "__main__":
    test_game_creation()
    test_tower_hp()
    test_wait_action()
    test_elixir_cap()
    test_valid_actions_mask()
    test_card_placement()
    test_game_runs_to_completion()
    test_clone()
    print("All simulator tests passed!")
