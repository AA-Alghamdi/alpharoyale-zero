"""Cross-engine card-stat conformance: compare crsim ``CARD_DEFS`` against the
vendored oracle stats, kind-aware (troops, buildings and spells expose stats
differently).

The comparison is deliberately conservative: a stat is only judged when both
engines express it in a directly comparable way. Spell area damage and troop
projectile damage use different conventions across the two engines and are
reported as ``na`` (informational) rather than scored, so the conformance gate
stays meaningful instead of flagging schema differences as bugs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from crsim.cards import CARD_DEFS, EntityKind

from .name_map import build_mapping
from .oracle_data import ORACLE_SPEED_PER_TILE_S, load_oracle

# percentage tolerances (|a-b| / max(|a|,|b|))
TOL_PCT = {"hp": 8.0, "damage": 8.0, "hit_speed": 12.0, "sight_range": 12.0, "speed": 3.0}
# absolute tile tolerances
TOL_ABS = {"range": 0.35, "radius": 0.35}

AGREE, DIVERGE, NA = "agree", "diverge", "na"


@dataclass
class StatResult:
    stat: str
    ours: float | None
    oracle: float | None
    pct: float | None
    status: str


@dataclass
class CardComparison:
    crsim_name: str
    oracle_name: str
    kind: str
    stats: dict[str, StatResult] = field(default_factory=dict)

    @property
    def divergences(self) -> list[str]:
        return [s for s, r in self.stats.items() if r.status == DIVERGE]


def _pct(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom * 100.0


def _judge_pct(stat: str, ours, oracle) -> StatResult:
    if ours is None or oracle is None:
        return StatResult(stat, ours, oracle, None, NA)
    p = _pct(ours, oracle)
    status = AGREE if p <= TOL_PCT[stat] else DIVERGE
    return StatResult(stat, ours, oracle, p, status)


def _judge_abs(stat: str, ours, oracle) -> StatResult:
    if ours is None or oracle is None:
        return StatResult(stat, ours, oracle, None, NA)
    p = _pct(ours, oracle) if max(ours, oracle) else 0.0
    status = AGREE if abs(ours - oracle) <= TOL_ABS[stat] else DIVERGE
    return StatResult(stat, ours, oracle, p, status)


def _judge_exact(stat: str, ours, oracle) -> StatResult:
    if ours is None or oracle is None:
        return StatResult(stat, ours, oracle, None, NA)
    status = AGREE if ours == oracle else DIVERGE
    return StatResult(stat, ours, oracle, _pct(ours, oracle), status)


def compare_card(crsim_name: str, oracle_name: str, cd, od: dict) -> CardComparison:
    kind = EntityKind(cd.kind)
    cmp = CardComparison(crsim_name, oracle_name, kind.name)
    s = cmp.stats

    s["mana"] = _judge_exact("mana", cd.cost, od.get("mana_cost"))

    if kind is EntityKind.SPELL:
        # Oracle stores spell radius in ``range``; crsim uses ``splash_radius``.
        s["radius"] = _judge_abs("radius", cd.splash_radius or None, od.get("range") or None)
        return cmp

    # Troops and buildings.
    s["hp"] = _judge_pct("hp", cd.hp or None, od.get("scaled_hitpoints"))
    # A few cards (Firecracker, Princess) carry all damage on a special
    # projectile and store 0 here; skip those rather than mis-flag.
    s["damage"] = _judge_pct("damage", cd.damage_per_hit or None, od.get("scaled_damage"))
    s["hit_speed"] = _judge_pct(
        "hit_speed",
        cd.hit_speed or None,
        (od["hit_speed"] / 1000.0) if od.get("hit_speed") else None,
    )
    s["range"] = _judge_abs("range", cd.attack_range or None, od.get("range") or None)
    s["sight_range"] = _judge_pct(
        "sight_range", cd.sight_range or None, od.get("sight_range") or None
    )
    s["speed"] = _judge_pct(
        "speed",
        cd.speed or None,
        (od["speed"] / ORACLE_SPEED_PER_TILE_S) if od.get("speed") else None,
    )
    return cmp


def run_conformance() -> tuple[list[CardComparison], dict]:
    """Compare every mappable crsim card against the oracle.

    Returns ``(comparisons, meta)`` where ``meta`` reports coverage.
    """
    oracle = load_oracle()
    crsim_names = [ct.name for ct in CARD_DEFS]
    mapping = build_mapping(crsim_names, list(oracle.keys()))

    by_name = {ct.name: CARD_DEFS[ct] for ct in CARD_DEFS}
    comparisons = [
        compare_card(cn, on, by_name[cn], oracle[on]) for cn, on in sorted(mapping.items())
    ]
    meta = {
        "crsim_cards": len(crsim_names),
        "mapped": len(mapping),
        "oracle_cards": len(oracle),
    }
    return comparisons, meta


def divergence_set(comparisons: list[CardComparison]) -> set[tuple[str, str]]:
    """Return the set of ``(crsim_name, stat)`` pairs that diverge."""
    return {
        (c.crsim_name, stat)
        for c in comparisons
        for stat in c.divergences
    }
