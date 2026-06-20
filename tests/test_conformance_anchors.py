"""Ground-truth conformance: card stats vs verified wiki Level-11 anchors.

This is the "vs reality" accuracy gate (no external engine required). Each
anchor in ``tools/conformance/wiki_anchors.py`` is a hand-verified Level-11
value from the Clash Royale Wiki / Liquipedia. The simulator must reproduce it
within rounding tolerance.

Cards in ``KNOWN_STALE`` are documented data-staleness gaps: the bundled
authentic game-data dump predates a 2024-25 balance change, so our value is the
older one. They are marked ``xfail(strict=True)`` — if a future data refresh
fixes them, the test will XPASS and fail the suite, prompting removal from the
allowlist. This keeps the known-gap list honest and self-cleaning.

The list is currently empty: the two previously-stale cards (Wizard, Mini
P.E.K.K.A) are corrected by ``crsim.gamedata.POST_SNAPSHOT_L11_PATCHES`` (the
bundled cr-api-data export is itself stale on them, so re-pulling it does not
help — see CONFORMANCE.md). The mechanism is retained for the next time the
export lags a balance change.
"""

from __future__ import annotations

import pytest

from crsim.cards import CARD_DEFS, CardType
from tools.conformance.wiki_anchors import WIKI_ANCHORS

# Cards whose bundled data predates a verified recent balance change and whose
# correct value is NOT yet available from the data export. Wizard and Mini
# P.E.K.K.A used to live here; they are now corrected via
# crsim.gamedata.POST_SNAPSHOT_L11_PATCHES, so the set is empty. Keep the
# mechanism for the next export-lag.
# See CONFORMANCE.md for the full prioritised staleness list.
KNOWN_STALE: set[str] = set()

TOLERANCE = 0.01  # ±1% (level-scaling rounding)


def _anchor_cases():
    cases = []
    for name, anc in sorted(WIKI_ANCHORS.items()):
        for stat, attr, val in (
            ("hp", "hp", anc.hp),
            ("damage", "damage_per_hit", anc.damage),
        ):
            if val is None:
                continue
            marks = (
                [pytest.mark.xfail(reason="data dump predates 2024-25 buff", strict=True)]
                if name in KNOWN_STALE
                else []
            )
            cases.append(pytest.param(name, attr, float(val), marks=marks, id=f"{name}-{stat}"))
    return cases


@pytest.mark.parametrize("name,attr,wiki_val", _anchor_cases())
def test_card_matches_wiki_anchor(name: str, attr: str, wiki_val: float) -> None:
    cd = CARD_DEFS[CardType[name]]
    ours = float(getattr(cd, attr, 0.0) or 0.0)
    assert wiki_val > 0
    delta = abs(ours - wiki_val) / wiki_val
    assert delta <= TOLERANCE, f"{name}.{attr}: ours={ours} wiki={wiki_val} ({delta*100:.1f}% off)"
