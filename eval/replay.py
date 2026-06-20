"""Game replay recording and analysis.

Records full game states for:
  - Post-game analysis
  - Training data generation
  - Debugging agent behavior
  - Visualization
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReplayFrame:
    """Single tick snapshot of game state."""
    tick: int
    elixir_p0: float
    elixir_p1: float
    crowns_p0: int
    crowns_p1: int
    entity_count: int
    action_p0: int = -1  # action_id, -1 = no action
    action_p1: int = -1
    value_p0: float = 0.0  # model's value estimate
    value_p1: float = 0.0
    policy_entropy_p0: float = 0.0
    policy_entropy_p1: float = 0.0
    entity_positions: list[tuple[float, float, int, int]] = field(
        default_factory=list,
    )  # (x, y, owner, card_type)


@dataclass
class GameReplay:
    """Full game replay data."""
    deck_p0: list[int] = field(default_factory=list)
    deck_p1: list[int] = field(default_factory=list)
    result: str = ""  # "p0_win", "p1_win", "draw"
    total_ticks: int = 0
    frames: list[ReplayFrame] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ReplayRecorder:
    """Record game replays during self-play or evaluation."""

    def __init__(self, record_interval: int = 1) -> None:
        self.record_interval = record_interval
        self._current_replay: GameReplay | None = None

    def start_game(
        self,
        deck_p0: list[int],
        deck_p1: list[int],
        metadata: dict | None = None,
    ) -> None:
        self._current_replay = GameReplay(
            deck_p0=list(deck_p0),
            deck_p1=list(deck_p1),
            metadata=metadata or {},
        )

    def record_tick(
        self,
        game,  # CRGame
        tick: int,
        action_p0: int = -1,
        action_p1: int = -1,
        value_p0: float = 0.0,
        value_p1: float = 0.0,
        policy_entropy_p0: float = 0.0,
        policy_entropy_p1: float = 0.0,
    ) -> None:
        if self._current_replay is None:
            return
        if tick % self.record_interval != 0:
            return

        entities = []
        for e in game.entities:
            if e.alive:
                entities.append((e.x, e.y, e.owner, int(e.card_type)))

        frame = ReplayFrame(
            tick=tick,
            elixir_p0=game.players[0].elixir,
            elixir_p1=game.players[1].elixir,
            crowns_p0=game._count_crowns(0),
            crowns_p1=game._count_crowns(1),
            entity_count=len(entities),
            action_p0=action_p0,
            action_p1=action_p1,
            value_p0=value_p0,
            value_p1=value_p1,
            policy_entropy_p0=policy_entropy_p0,
            policy_entropy_p1=policy_entropy_p1,
            entity_positions=entities,
        )
        self._current_replay.frames.append(frame)

    def end_game(self, result: str, total_ticks: int) -> GameReplay | None:
        if self._current_replay is None:
            return None
        self._current_replay.result = result
        self._current_replay.total_ticks = total_ticks
        replay = self._current_replay
        self._current_replay = None
        return replay


def save_replay(replay: GameReplay, path: str) -> None:
    """Save replay to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "deck_p0": replay.deck_p0,
        "deck_p1": replay.deck_p1,
        "result": replay.result,
        "total_ticks": replay.total_ticks,
        "metadata": replay.metadata,
        "frames": [
            {
                "tick": f.tick,
                "elixir_p0": round(f.elixir_p0, 2),
                "elixir_p1": round(f.elixir_p1, 2),
                "crowns_p0": f.crowns_p0,
                "crowns_p1": f.crowns_p1,
                "entity_count": f.entity_count,
                "action_p0": f.action_p0,
                "action_p1": f.action_p1,
                "value_p0": round(f.value_p0, 4),
                "value_p1": round(f.value_p1, 4),
                "policy_entropy_p0": round(f.policy_entropy_p0, 4),
                "policy_entropy_p1": round(f.policy_entropy_p1, 4),
                "entities": [
                    {"x": round(x, 1), "y": round(y, 1), "owner": o, "type": t}
                    for x, y, o, t in f.entity_positions
                ],
            }
            for f in replay.frames
        ],
    }

    p.write_text(json.dumps(data))
    logger.info("Saved replay (%d frames) to %s", len(replay.frames), path)


def load_replay(path: str) -> GameReplay:
    """Load replay from JSON file."""
    data = json.loads(Path(path).read_text())

    frames = []
    for fd in data["frames"]:
        frames.append(ReplayFrame(
            tick=fd["tick"],
            elixir_p0=fd["elixir_p0"],
            elixir_p1=fd["elixir_p1"],
            crowns_p0=fd["crowns_p0"],
            crowns_p1=fd["crowns_p1"],
            entity_count=fd["entity_count"],
            action_p0=fd.get("action_p0", -1),
            action_p1=fd.get("action_p1", -1),
            value_p0=fd.get("value_p0", 0.0),
            value_p1=fd.get("value_p1", 0.0),
            policy_entropy_p0=fd.get("policy_entropy_p0", 0.0),
            policy_entropy_p1=fd.get("policy_entropy_p1", 0.0),
            entity_positions=[
                (e["x"], e["y"], e["owner"], e["type"])
                for e in fd.get("entities", [])
            ],
        ))

    return GameReplay(
        deck_p0=data["deck_p0"],
        deck_p1=data["deck_p1"],
        result=data["result"],
        total_ticks=data["total_ticks"],
        frames=frames,
        metadata=data.get("metadata", {}),
    )


class MatchupAnalyzer:
    """Analyze matchup statistics from tournament results and replays."""

    def __init__(self) -> None:
        self.matchup_stats: dict[tuple[str, str], dict] = {}

    def add_result(
        self,
        deck_a: str,
        deck_b: str,
        winner: str,
        crowns_a: int = 0,
        crowns_b: int = 0,
    ) -> None:
        key = (min(deck_a, deck_b), max(deck_a, deck_b))
        if key not in self.matchup_stats:
            self.matchup_stats[key] = {
                "games": 0, "a_wins": 0, "b_wins": 0, "draws": 0,
                "total_crowns_a": 0, "total_crowns_b": 0,
            }

        stats = self.matchup_stats[key]
        stats["games"] += 1
        if winner == "a":
            stats["a_wins"] += 1
        elif winner == "b":
            stats["b_wins"] += 1
        else:
            stats["draws"] += 1
        stats["total_crowns_a"] += crowns_a
        stats["total_crowns_b"] += crowns_b

    def get_summary(self) -> dict[tuple[str, str], dict]:
        """Get matchup summary with win rates."""
        summary = {}
        for key, stats in self.matchup_stats.items():
            n = max(stats["games"], 1)
            summary[key] = {
                "games": stats["games"],
                "a_winrate": stats["a_wins"] / n,
                "b_winrate": stats["b_wins"] / n,
                "draw_rate": stats["draws"] / n,
                "avg_crowns_a": stats["total_crowns_a"] / n,
                "avg_crowns_b": stats["total_crowns_b"] / n,
            }
        return summary

    def print_matchup_table(self) -> None:
        """Print a formatted matchup analysis table."""
        summary = self.get_summary()
        logger.info("=== Matchup Analysis ===")
        for (da, db), stats in sorted(summary.items()):
            logger.info(
                "%-25s vs %-25s: %d games, A %.0f%% / B %.0f%% / D %.0f%%",
                da, db,
                stats["games"],
                stats["a_winrate"] * 100,
                stats["b_winrate"] * 100,
                stats["draw_rate"] * 100,
            )
