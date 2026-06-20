"""TCP client for the Scroll headless battle server.

Sends JSON requests over a persistent TCP connection and receives JSON
responses. Thread-safe for use from multiple self-play workers.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass


@dataclass
class GameState:
    """Parsed game state from the Scroll server."""

    tick: int
    max_tick: int
    game_over: bool
    winner: int  # 0, 1, or -1 (draw/ongoing)
    checksum: int

    # Per-player state
    p0_elixir: float
    p0_king_hp: int
    p0_princess_left_hp: int
    p0_princess_right_hp: int
    p0_crowns: int
    p0_hand: list[int]
    p0_next_card: int

    p1_elixir: float
    p1_king_hp: int
    p1_princess_left_hp: int
    p1_princess_right_hp: int
    p1_crowns: int
    p1_hand: list[int]
    p1_next_card: int

    # Entity list
    entities: list[dict]


@dataclass
class CardPlay:
    """A card play command."""

    spell_id: int
    x: int
    y: int

    def to_dict(self) -> dict:
        return {"spell_id": self.spell_id, "x": self.x, "y": self.y}


class ScrollClient:
    """Client for communicating with the Scroll headless battle server.

    Usage::

        client = ScrollClient("localhost", 9340)
        client.connect()

        battle_id = client.new_battle(deck_p0=[...], deck_p1=[...])
        state = client.step(battle_id, commands_p0=[...], commands_p1=[...])
        ...
        client.destroy(battle_id)
        client.close()
    """

    def __init__(self, host: str = "localhost", port: int = 9340) -> None:
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._file = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Establish TCP connection to the Scroll server."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        self._file = self._sock.makefile("r")

    def close(self) -> None:
        """Close the TCP connection."""
        if self._file:
            self._file.close()
        if self._sock:
            self._sock.close()

    def _send_request(self, request: dict) -> dict:
        """Send a JSON request and read the JSON response."""
        assert self._sock is not None, "Not connected"
        with self._lock:
            data = json.dumps(request) + "\n"
            self._sock.sendall(data.encode("utf-8"))
            line = self._file.readline()
            if not line:
                raise ConnectionError("Server closed connection")
            return json.loads(line)

    def ping(self) -> bool:
        """Health check."""
        resp = self._send_request({"type": "ping"})
        return resp.get("type") == "pong"

    def list_spells(self) -> list[dict]:
        """Get available spells from the server."""
        resp = self._send_request({"type": "list_spells"})
        if resp["type"] == "error":
            raise RuntimeError(resp["message"])
        return resp["spells"]

    def new_battle(
        self,
        deck_p0: list[int],
        deck_p1: list[int],
        arena_id: int = 54000001,
    ) -> int:
        """Create a new headless battle. Returns battle_id."""
        resp = self._send_request({
            "type": "new_battle",
            "deck_p0": deck_p0,
            "deck_p1": deck_p1,
            "arena_id": arena_id,
        })
        if resp["type"] == "error":
            raise RuntimeError(resp["message"])
        return resp["battle_id"]

    def step(
        self,
        battle_id: int,
        commands_p0: list[CardPlay] | None = None,
        commands_p1: list[CardPlay] | None = None,
        ticks: int = 1,
    ) -> GameState:
        """Advance battle by `ticks` steps with optional card plays."""
        resp = self._send_request({
            "type": "step",
            "battle_id": battle_id,
            "ticks": ticks,
            "commands_p0": [c.to_dict() for c in (commands_p0 or [])],
            "commands_p1": [c.to_dict() for c in (commands_p1 or [])],
        })
        if resp["type"] == "error":
            raise RuntimeError(resp["message"])
        return _parse_state(resp["state"])

    def get_state(self, battle_id: int) -> GameState:
        """Read current game state without advancing."""
        resp = self._send_request({
            "type": "get_state",
            "battle_id": battle_id,
        })
        if resp["type"] == "error":
            raise RuntimeError(resp["message"])
        return _parse_state(resp["state"])

    def destroy(self, battle_id: int) -> None:
        """Destroy a battle and free server-side resources."""
        resp = self._send_request({
            "type": "destroy",
            "battle_id": battle_id,
        })
        if resp["type"] == "error":
            raise RuntimeError(resp["message"])


def _parse_state(data: dict) -> GameState:
    """Parse a JSON state dict into a GameState dataclass."""
    p0 = data["player_0"]
    p1 = data["player_1"]
    return GameState(
        tick=data["tick"],
        max_tick=data["max_tick"],
        game_over=data["game_over"],
        winner=data["winner"],
        checksum=data["checksum"],
        p0_elixir=p0["elixir"],
        p0_king_hp=p0["king_tower_hp"],
        p0_princess_left_hp=p0["princess_tower_left_hp"],
        p0_princess_right_hp=p0["princess_tower_right_hp"],
        p0_crowns=p0["crowns"],
        p0_hand=p0["hand"],
        p0_next_card=p0["next_card"],
        p1_elixir=p1["elixir"],
        p1_king_hp=p1["king_tower_hp"],
        p1_princess_left_hp=p1["princess_tower_left_hp"],
        p1_princess_right_hp=p1["princess_tower_right_hp"],
        p1_crowns=p1["crowns"],
        p1_hand=p1["hand"],
        p1_next_card=p1["next_card"],
        entities=data.get("entities", []),
    )
