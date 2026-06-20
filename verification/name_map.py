"""Map crsim card names to the oracle's card names.

Most cards match by normalised name (case/underscore/space-insensitive, with a
trailing-``s`` plural fallback). The rest are renamed in one engine or the other
and need explicit overrides, verified by stat agreement (mana + hp).

Cards with no confident oracle counterpart in the current oracle snapshot are
listed in :data:`NO_ORACLE_COUNTERPART` so coverage is auditable rather than
silently dropped.
"""
from __future__ import annotations

# crsim CardType.name -> oracle card name. Each verified by agreement on mana
# plus at least one of {hp, damage}.
OVERRIDES: dict[str, str] = {
    "ELITE_BARBARIANS": "AngryBarbarians",
    "EXECUTIONER": "AxeMan",
    "FURNACE": "FirespiritHut",
    "MOTHER_WITCH": "WitchMother",
    "CANNON_CART": "MovingCannon",
    "RUNE_GIANT": "GiantBuffer",
}

# crsim cards with no confident match in the oracle's gamedata snapshot:
# champion summons, recent/variant cards, or cards the oracle simply omits.
NO_ORACLE_COUNTERPART: frozenset[str] = frozenset(
    {
        "BUSH_GOBLINS",
        "CURSED_HOG",
        "FLYING_MACHINE",
        "GOBLIN_BRAWLER",
        "GUARDIENNE",
        "HEAL_SPIRIT",
        "SPARKY",
        "SPIRIT_EMPRESS",
        "VINES",
        "VOID",
        "ZAPPIES",
    }
)


def normalise(name: str) -> str:
    return name.lower().replace("_", "").replace(" ", "").replace("-", "").replace(".", "")


def build_mapping(crsim_names: list[str], oracle_names: list[str]) -> dict[str, str]:
    """Return ``{crsim_name: oracle_name}`` for every crsim card that maps."""
    by_norm: dict[str, str] = {}
    for o in oracle_names:
        by_norm.setdefault(normalise(o), o)

    mapping: dict[str, str] = {}
    for name in crsim_names:
        if name in NO_ORACLE_COUNTERPART:
            continue
        if name in OVERRIDES:
            mapping[name] = OVERRIDES[name]
            continue
        n = normalise(name)
        for key in (n, n + "s", n.rstrip("s")):
            if key in by_norm:
                mapping[name] = by_norm[key]
                break
    return mapping
