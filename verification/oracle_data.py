"""Load the vendored oracle card-stat fixture.

The fixture is a snapshot of the second engine's (samdickson22/clash-simulator)
interpreted card stats; see ``scripts/gen_oracle_fixture.py`` to regenerate it.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "oracle_card_stats.json"

# Oracle ``speed`` is expressed as tiles/second * 30 (its 30 FPS tick rate);
# crsim stores tiles/second. Divide oracle speed by this to compare.
ORACLE_SPEED_PER_TILE_S = 30.0


def load_oracle() -> dict[str, dict]:
    """Return ``{oracle_card_name: stat_dict}`` from the vendored fixture."""
    with open(FIXTURE_PATH) as f:
        payload = json.load(f)
    return payload["cards"]


def load_provenance() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)["_provenance"]
