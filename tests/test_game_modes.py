"""Golden tests for authentic match-mode rules: elixir multipliers, overtime."""

from __future__ import annotations

from crsim.constants import (
    DOUBLE_ELIXIR_START_TICKS,
    ELIXIR_REGEN_DOUBLE,
    ELIXIR_REGEN_NORMAL,
    ELIXIR_REGEN_OVERTIME,
    REGULAR_TIME_TICKS,
)
from crsim.game import CRGame, GamePhase, GameResult


def test_single_elixir_in_first_two_minutes():
    game = CRGame()
    game.tick_count = 0
    assert game._elixir_rate() == ELIXIR_REGEN_NORMAL
    game.tick_count = DOUBLE_ELIXIR_START_TICKS - 1
    assert game._elixir_rate() == ELIXIR_REGEN_NORMAL


def test_double_elixir_in_final_minute_of_regulation():
    game = CRGame()
    game.tick_count = DOUBLE_ELIXIR_START_TICKS
    assert game._elixir_rate() == ELIXIR_REGEN_DOUBLE
    game.tick_count = REGULAR_TIME_TICKS - 1
    assert game._elixir_rate() == ELIXIR_REGEN_DOUBLE


def test_triple_elixir_in_overtime():
    game = CRGame()
    game.phase = GamePhase.OVERTIME
    assert game._elixir_rate() == ELIXIR_REGEN_OVERTIME
    assert ELIXIR_REGEN_OVERTIME == ELIXIR_REGEN_NORMAL * 3.0


def test_elixir_multipliers_ramp_1x_2x_3x():
    assert ELIXIR_REGEN_DOUBLE == ELIXIR_REGEN_NORMAL * 2.0
    assert ELIXIR_REGEN_OVERTIME == ELIXIR_REGEN_NORMAL * 3.0


def test_overtime_first_tower_destroyed_wins_instantly():
    game = CRGame()
    game.phase = GamePhase.OVERTIME
    game.tick_count = REGULAR_TIME_TICKS + 100
    # Destroy one of P1's princess towers -> P0 has a crown, should win now.
    game.princess_towers[1][0].hp = 0
    game._check_win()
    assert game.phase == GamePhase.ENDED
    assert game.result == GameResult.P0_WIN


def test_regulation_does_not_end_early_on_single_tower():
    game = CRGame()
    assert game.phase == GamePhase.REGULAR
    game.tick_count = 1000
    game.princess_towers[1][0].hp = 0
    game._check_win()
    # Still regulation: a tower lead does not end the game before time.
    assert game.phase == GamePhase.REGULAR
    assert game.result == GameResult.IN_PROGRESS
