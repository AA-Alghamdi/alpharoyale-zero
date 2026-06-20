"""Ground-truth card stats at Tournament Standard (Level 11).

These values are transcribed from the Clash Royale Wiki (Fandom) / Liquipedia
"Level Changes" tables, current as of the 2025 balance data. They are the
*reality* anchor: the simulator is correct only insofar as it reproduces these.

Only cards whose Level-11 values have been individually verified against a
primary source are listed here. This set is intentionally small and
high-confidence — it exists to catch gross data errors and balance-patch
staleness, not to be an exhaustive table (that is what ``crsim/cards.py`` and
the authentic game-data overlay are for).

Each entry maps a ``CardType`` name to verified Level-11 ``hp`` / ``damage``
(omit a field when it is not meaningful, e.g. spell hp). ``source`` documents
where the number came from so it can be re-verified.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anchor:
    hp: float | None = None
    damage: float | None = None
    source: str = ""


# Verified Level-11 (tournament standard) values.
WIKI_ANCHORS: dict[str, Anchor] = {
    # Core troops — long-stable values, verified current 2025.
    "KNIGHT": Anchor(hp=1766, damage=202, source="fandom Knight L11"),
    "MUSKETEER": Anchor(hp=720, damage=218, source="fandom/stats-sc Musketeer L11"),
    "PEKKA": Anchor(hp=3760, damage=816, source="fandom P.E.K.K.A L11"),
    "MEGA_KNIGHT": Anchor(hp=3993, source="fandom Mega Knight L11"),
    "VALKYRIE": Anchor(hp=1908, source="fandom Valkyrie L11"),
    "PRINCE": Anchor(hp=1920, source="fandom Prince L11"),
    # Spells — damage only.
    "FIREBALL": Anchor(damage=689, source="fandom Fireball L11"),
    "ROCKET": Anchor(damage=1485, source="fandom Rocket L11"),
    "ZAP": Anchor(damage=192, source="fandom Zap L11"),
    "ARROWS": Anchor(damage=322, source="fandom Arrows L11"),
    # Cards with a KNOWN recent buff — used to detect game-data staleness.
    # Wizard HP was raised +4.8% (Aug 2024): 721 -> 755.
    "WIZARD": Anchor(hp=755, source="fandom Wizard L11 (post Aug-2024 +4.8% HP)"),
    # Mini P.E.K.K.A current L11 per fandom level table.
    "MINI_PEKKA": Anchor(hp=1433, damage=755, source="fandom Mini P.E.K.K.A L11"),
    # Cards whose HP was buffed after the bundled export was cut. Each HP value
    # is corroborated by 3 independent sources; damage omitted where the source
    # figure is a representation difference (multi-pellet / source spread).
    "GIANT_SKELETON": Anchor(hp=3617, source="fandom+Liquipedia+deckmelon Giant Skeleton L11"),
    "HUNTER": Anchor(hp=885, source="fandom+Liquipedia+deckmelon Hunter L11"),
    "MONK": Anchor(hp=2150, source="fandom+deckmelon Monk L11 (champion base)"),
}
