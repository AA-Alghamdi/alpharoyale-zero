"""Scenario-level behavioural cross-engine comparison.

Stat conformance (see :mod:`verification.conformance`) checks the *inputs* agree.
This checks the *dynamics* agree: it spawns the same 1-vs-1 duel in crsim and in
the oracle (samdickson22/clash-simulator) and compares the outcome — who wins and
how long the kill takes.

Two independently-built engines will never be tick-identical (33 ms vs 50 ms
ticks, different target-acquisition order), so the comparison is intentionally
coarse: the winner must match and time-to-kill must agree within
:data:`TTK_TOL`.

Requires a local oracle clone (``--oracle`` / ``$ORACLE_REPO`` / ``../clash-simulator``)
and the oracle's deps (``numba``, ``msgspec``); it is therefore a dev tool, not a
CI gate. Run it directly:

    python -m verification.behavioral --oracle /path/to/clash-simulator
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from crsim.cards import CARD_DEFS, CardType
from crsim.entities import entity_from_card
from crsim.game import Action, CRGame

from .name_map import build_mapping

# Duel geometry: midfield (on the river, x=9) so towers are out of range, two
# enemy units a fraction of a tile apart so they engage immediately.
DUEL_X = 9.0
DUEL_Y_A = 15.6
DUEL_Y_B = 16.2
TIMEOUT_S = 30.0
TTK_TOL = 0.30  # fractional tolerance on time-to-kill

# Canonical melee/ranged duels with a known winner (None = mirror, expect a tie).
DUELS: list[tuple[str, str]] = [
    ("KNIGHT", "KNIGHT"),
    ("MINI_PEKKA", "KNIGHT"),
    ("MUSKETEER", "KNIGHT"),
    ("VALKYRIE", "MUSKETEER"),
    ("PEKKA", "MINI_PEKKA"),
    ("KNIGHT", "ARCHERS"),
]


@dataclass
class DuelResult:
    a: str
    b: str
    crsim_winner: str | None
    crsim_ttk: float | None
    oracle_winner: str | None
    oracle_ttk: float | None

    @property
    def winners_agree(self) -> bool:
        return self.crsim_winner == self.oracle_winner

    @property
    def ttk_agree(self) -> bool:
        if self.crsim_ttk is None or self.oracle_ttk is None:
            return self.crsim_ttk is None and self.oracle_ttk is None
        denom = max(self.crsim_ttk, self.oracle_ttk, 1e-9)
        return abs(self.crsim_ttk - self.oracle_ttk) / denom <= TTK_TOL


# --------------------------------------------------------------------------- #
# crsim side
# --------------------------------------------------------------------------- #
def _crsim_duel(card_a: str, card_b: str) -> tuple[str | None, float | None]:
    game = CRGame(seed=0)
    for owner, name, y in ((0, card_a, DUEL_Y_A), (1, card_b, DUEL_Y_B)):
        cd = CARD_DEFS[CardType[name]]
        e = entity_from_card(game._alloc_eid(), owner, cd, DUEL_X, y)
        e.deploy_timer = 0.0
        e.is_deployed = True
        game.entities.append(e)

    wait = [Action(0, -1), Action(1, -1)]
    n = int(TIMEOUT_S / 0.05)
    for tick in range(n):
        game.step(wait)
        units = [e for e in game.entities if e.alive and not e.is_tower]
        if len(units) < 2:
            ttk = (tick + 1) * 0.05
            winner = {0: card_a, 1: card_b}.get(units[0].owner) if units else None
            return winner, ttk
    return None, None  # both survived to timeout


# --------------------------------------------------------------------------- #
# oracle side
# --------------------------------------------------------------------------- #
def _resolve_oracle(arg: str | None) -> Path:
    for cand in (arg, os.environ.get("ORACLE_REPO"), str(Path(__file__).resolve().parents[2] / "clash-simulator")):
        if cand and (Path(cand) / "gamedata.json").exists():
            return Path(cand).resolve()
    raise FileNotFoundError("oracle clone not found (pass --oracle /path/to/clash-simulator)")


class OracleEngine:
    """Thin wrapper that loads the oracle once and runs direct-spawn duels."""

    def __init__(self, oracle_path: Path):
        self.path = oracle_path
        self._cwd = os.getcwd()
        sys.path.insert(0, str(oracle_path))
        os.chdir(oracle_path)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                from src.clasher.engine import BattleEngine  # type: ignore
                from src.clasher.entities import Position  # type: ignore

                self._BattleEngine = BattleEngine
                self._Position = Position
                self._engine = BattleEngine("gamedata.json")
        finally:
            os.chdir(self._cwd)

    def card_names(self) -> list[str]:
        with contextlib.redirect_stdout(io.StringIO()):
            os.chdir(self.path)
            battle = self._engine.create_battle()
            names = list(battle.card_loader.load_card_definitions().keys())
            os.chdir(self._cwd)
        return names

    def duel(self, oracle_a: str, oracle_b: str) -> tuple[str | None, float | None]:
        os.chdir(self.path)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                battle = self._engine.create_battle()
                loader = battle.card_loader
                cs_a, cs_b = loader.get_card(oracle_a), loader.get_card(oracle_b)
                battle._spawn_troop(self._Position(DUEL_X, DUEL_Y_A), 0, cs_a)
                battle._spawn_troop(self._Position(DUEL_X, DUEL_Y_B), 1, cs_b)
                n = int(TIMEOUT_S / battle.dt)
                for _ in range(n):
                    battle.step()
                    units = [
                        e
                        for e in battle.entities.values()
                        if type(e).__name__ == "Troop" and e.is_alive
                    ]
                    if len(units) < 2:
                        ttk = battle.time
                        winner = (
                            {0: oracle_a, 1: oracle_b}.get(units[0].player_id) if units else None
                        )
                        return winner, ttk
            return None, None
        finally:
            os.chdir(self._cwd)


def run_duels(oracle_path: Path, duels: list[tuple[str, str]] = DUELS) -> list[DuelResult]:
    oracle = OracleEngine(oracle_path)
    mapping = build_mapping([ct.name for ct in CARD_DEFS], oracle.card_names())
    results: list[DuelResult] = []
    for a, b in duels:
        if a not in mapping or b not in mapping:
            continue
        c_win, c_ttk = _crsim_duel(a, b)
        o_win_oracle, o_ttk = oracle.duel(mapping[a], mapping[b])
        # translate oracle winner (oracle name) back to crsim name
        inv = {v: k for k, v in mapping.items()}
        o_win = inv.get(o_win_oracle) if o_win_oracle else None
        results.append(DuelResult(a, b, c_win, c_ttk, o_win, o_ttk))
    return results


def render(results: list[DuelResult]) -> str:
    lines = ["# Cross-engine behavioural conformance (1v1 duels)", ""]
    lines.append("| A | B | crsim winner | crsim TTK | oracle winner | oracle TTK | winner✓ | ttk✓ |")
    lines.append("|---|---|--------------|----------:|---------------|-----------:|:------:|:----:|")
    for r in results:
        ct = f"{r.crsim_ttk:.2f}s" if r.crsim_ttk is not None else "—"
        ot = f"{r.oracle_ttk:.2f}s" if r.oracle_ttk is not None else "—"
        lines.append(
            f"| {r.a} | {r.b} | {r.crsim_winner or 'tie'} | {ct} | "
            f"{r.oracle_winner or 'tie'} | {ot} | "
            f"{'yes' if r.winners_agree else 'NO'} | {'yes' if r.ttk_agree else 'NO'} |"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Cross-engine behavioural duels")
    ap.add_argument("--oracle", help="Path to a clone of samdickson22/clash-simulator")
    args = ap.parse_args()
    results = run_duels(_resolve_oracle(args.oracle))
    print(render(results))


if __name__ == "__main__":
    main()
