"""Regenerate the vendored oracle card-stat fixture from a local clone of the
second-engine oracle (samdickson22/clash-simulator).

The fixture (``verification/data/oracle_card_stats.json``) is committed so the
cross-engine conformance harness and its CI test run without the external repo
present. Re-run this only when bumping the oracle snapshot:

    python scripts/gen_oracle_fixture.py --oracle /path/to/clash-simulator

If ``--oracle`` is omitted, ``$ORACLE_REPO`` then ``../clash-simulator`` are
tried. The oracle needs its own deps (``pip install numba msgspec``).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "verification" / "data" / "oracle_card_stats.json"

# Stats pulled from the oracle's interpreted card definitions. Units are the
# oracle's native units; the harness converts where they differ from crsim.
STAT_KEYS = (
    "scaled_hitpoints",
    "hitpoints",
    "scaled_damage",
    "damage",
    "hit_speed",  # ms
    "range",  # tiles (spell radius for spells)
    "sight_range",  # tiles
    "speed",  # tiles/s * 30
    "mana_cost",
    "collision_radius",
    "deploy_time",  # ms
    "load_time",  # ms
)


def resolve_oracle(arg: str | None) -> Path:
    for cand in (arg, os.environ.get("ORACLE_REPO"), str(REPO_ROOT.parent / "clash-simulator")):
        if cand and (Path(cand) / "gamedata.json").exists():
            return Path(cand).resolve()
    raise SystemExit(
        "Could not find the oracle clone. Pass --oracle /path/to/clash-simulator "
        "(must contain gamedata.json)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", help="Path to a clone of samdickson22/clash-simulator")
    args = ap.parse_args()

    oracle = resolve_oracle(args.oracle)
    cwd = os.getcwd()
    sys.path.insert(0, str(oracle))
    os.chdir(oracle)  # the oracle loads gamedata.json by relative path
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from src.clasher.engine import BattleEngine  # type: ignore

            engine = BattleEngine("gamedata.json")
            battle = engine.create_battle()
            loader = battle.card_loader
            defs = loader.load_card_definitions()
            stats: dict[str, dict] = {}
            for name in defs:
                try:
                    cs = loader.get_card(name)
                except Exception:
                    cs = None
                if cs is None:
                    continue
                stats[name] = {k: getattr(cs, k, None) for k in STAT_KEYS}
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=oracle, capture_output=True, text=True
        ).stdout.strip()
        gamedata = json.load(open(oracle / "gamedata.json"))
    finally:
        os.chdir(cwd)
        sys.path.remove(str(oracle))

    payload = {
        "_provenance": {
            "source_repo": "https://github.com/samdickson22/clash-simulator",
            "commit": commit,
            "gamedata_fingerprint": gamedata.get("meta", {}).get("fingerprint"),
            "stat_keys": list(STAT_KEYS),
            "note": "Oracle's interpreted L11 card stats. Regenerate with scripts/gen_oracle_fixture.py.",
        },
        "cards": stats,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"Wrote {len(stats)} oracle cards -> {FIXTURE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
