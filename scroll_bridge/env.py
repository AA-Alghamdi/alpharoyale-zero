"""Gymnasium-compatible environment wrapping the Scroll headless battle server.

This provides the same interface as our Python simulator (crsim.game.CRGame)
but backed by the real CR engine via Scroll, so the RL agent trains on
pixel-perfect game physics.

Usage::

    env = ScrollBattleEnv(host="localhost", port=9340)
    env.connect()

    obs = env.reset()
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
    env.close()
"""

from __future__ import annotations

import numpy as np

from crsim.constants import (
    ARENA_H,
    ARENA_W,
    KING_TOWER_HP,
    MAX_ELIXIR,
    NUM_CARD_TYPES,
    PRINCESS_TOWER_HP,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
    TOTAL_MAX_TICKS,
)
from scroll_bridge.client import CardPlay, GameState, ScrollClient

# Default deck: the 8-card starter deck from Scroll's save.rs
DEFAULT_DECK = [
    26000000,  # Knight
    26000001,  # Archers
    26000013,  # Bomber
    28000001,  # Fireball
    28000000,  # Arrows
    26000003,  # Giant
    26000004,  # Prince
    26000005,  # Baby Dragon
]


class ScrollBattleEnv:
    """RL environment backed by the real CR engine via Scroll.

    Implements a similar interface to crsim.game.CRGame so the MCTS
    and training pipeline can swap between simulators transparently.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9340,
        deck_p0: list[int] | None = None,
        deck_p1: list[int] | None = None,
        ticks_per_step: int = 1,
    ) -> None:
        self.client = ScrollClient(host, port)
        self.deck_p0 = deck_p0 or DEFAULT_DECK
        self.deck_p1 = deck_p1 or DEFAULT_DECK
        self.ticks_per_step = ticks_per_step

        self.battle_id: int | None = None
        self.state: GameState | None = None
        self._connected = False

    def connect(self) -> None:
        """Connect to the Scroll server."""
        self.client.connect()
        self._connected = True

    def close(self) -> None:
        """Clean up battle and disconnect."""
        if self.battle_id is not None:
            try:
                self.client.destroy(self.battle_id)
            except Exception:
                pass
            self.battle_id = None
        if self._connected:
            self.client.close()
            self._connected = False

    def reset(self) -> GameState:
        """Start a new battle. Returns initial game state."""
        if self.battle_id is not None:
            try:
                self.client.destroy(self.battle_id)
            except Exception:
                pass

        self.battle_id = self.client.new_battle(self.deck_p0, self.deck_p1)
        self.state = self.client.get_state(self.battle_id)
        return self.state

    def step(
        self,
        commands_p0: list[CardPlay] | None = None,
        commands_p1: list[CardPlay] | None = None,
    ) -> tuple[GameState, float, bool, dict]:
        """Advance the game by ticks_per_step.

        Returns
        -------
        state : GameState
        reward : float — +1 win, -1 loss, 0 ongoing/draw (from player 0 perspective)
        done : bool
        info : dict
        """
        assert self.battle_id is not None, "Call reset() first"

        self.state = self.client.step(
            self.battle_id,
            commands_p0=commands_p0,
            commands_p1=commands_p1,
            ticks=self.ticks_per_step,
        )

        done = self.state.game_over
        reward = 0.0
        if done:
            if self.state.winner == 0:
                reward = 1.0
            elif self.state.winner == 1:
                reward = -1.0

        info = {
            "tick": self.state.tick,
            "checksum": self.state.checksum,
        }

        return self.state, reward, done, info

    def encode_state(self, player: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Encode current game state into neural network input tensors.

        Returns the same format as model.features.encode_state():
        - spatial: (SPATIAL_CHANNELS, ARENA_H, ARENA_W)
        - scalar: (SCALAR_FEATURES,)

        This allows the Scroll env to plug directly into our existing
        MCTS and training pipeline.
        """
        assert self.state is not None, "No state available"

        spatial = np.zeros((SPATIAL_CHANNELS, ARENA_H, ARENA_W), dtype=np.float32)
        scalar = np.zeros(SCALAR_FEATURES, dtype=np.float32)

        # Encode entities into spatial channels
        for ent in self.state.entities:
            owner = ent["owner"]
            etype = ent["entity_type"]
            x = ent["x"]
            y = ent["y"]
            hp = ent["hp"]
            max_hp = ent["max_hp"]

            if 0 <= x < ARENA_W and 0 <= y < ARENA_H:
                # Channel assignment: 0-19 friendly, 20-39 enemy
                is_friendly = (owner == player)
                channel_offset = 0 if is_friendly else 20
                channel = channel_offset + (etype % 20)

                if channel < SPATIAL_CHANNELS - 4:
                    hp_frac = hp / max(max_hp, 1)
                    spatial[channel, y, x] = max(spatial[channel, y, x], hp_frac)

        # Tower HP channels (40-41)
        ps = self.state.player_0 if player == 0 else self.state.player_1
        ops = self.state.player_1 if player == 0 else self.state.player_0

        # Scalar features
        idx = 0
        scalar[idx] = ps.elixir / MAX_ELIXIR
        idx += 1
        scalar[idx] = self.state.tick / max(self.state.max_tick, 1)
        idx += 1

        # Hand encoding: 4 one-hot vectors of length NUM_CARD_TYPES
        for card_id in ps.hand[:4]:
            card_idx = card_id % NUM_CARD_TYPES
            scalar[idx + card_idx] = 1.0
            idx += NUM_CARD_TYPES
        if len(ps.hand) < 4:
            idx += NUM_CARD_TYPES * (4 - len(ps.hand))

        # Next card
        if ps.next_card > 0:
            scalar[idx + (ps.next_card % NUM_CARD_TYPES)] = 1.0
        idx += NUM_CARD_TYPES

        # Time remaining
        scalar[idx] = self.state.tick / TOTAL_MAX_TICKS
        idx += 1

        # Tower alive status (6 values)
        scalar[idx] = 1.0 if ps.king_hp > 0 else 0.0
        idx += 1
        scalar[idx] = 1.0 if ps.princess_left_hp > 0 else 0.0
        idx += 1
        scalar[idx] = 1.0 if ps.princess_right_hp > 0 else 0.0
        idx += 1
        scalar[idx] = 1.0 if ops.king_hp > 0 else 0.0
        idx += 1
        scalar[idx] = 1.0 if ops.princess_left_hp > 0 else 0.0
        idx += 1
        scalar[idx] = 1.0 if ops.princess_right_hp > 0 else 0.0
        idx += 1

        # Tower HP normalized (6 values)
        scalar[idx] = ps.king_hp / KING_TOWER_HP if ps.king_hp > 0 else 0.0
        idx += 1
        scalar[idx] = ps.princess_left_hp / PRINCESS_TOWER_HP if ps.princess_left_hp > 0 else 0.0
        idx += 1
        scalar[idx] = ps.princess_right_hp / PRINCESS_TOWER_HP if ps.princess_right_hp > 0 else 0.0
        idx += 1
        scalar[idx] = ops.king_hp / KING_TOWER_HP if ops.king_hp > 0 else 0.0
        idx += 1
        opl_hp = ops.princess_left_hp
        scalar[idx] = opl_hp / PRINCESS_TOWER_HP if opl_hp > 0 else 0.0
        idx += 1
        opr_hp = ops.princess_right_hp
        scalar[idx] = opr_hp / PRINCESS_TOWER_HP if opr_hp > 0 else 0.0
        idx += 1

        # Crown difference
        crown_diff = ps.crowns - ops.crowns
        if idx < SCALAR_FEATURES:
            scalar[idx] = crown_diff / 3.0

        return spatial, scalar

    @property
    def done(self) -> bool:
        """Whether the current battle is over."""
        if self.state is None:
            return True
        return self.state.game_over
