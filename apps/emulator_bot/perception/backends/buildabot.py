"""BuildABot perception backend.

Wraps ClashRoyaleBuildABot's detector stack (card-hash + YOLO units + tower-HP
numbers + screen) and converts its output to the unified PerceptionResult,
normalizing all labels to the canonical 125-card vocabulary and scaling every
box from the 368x652 detection space to the source frame's real pixels.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

_BA = os.path.expanduser("~/clash-royale-bot/ClashRoyaleBuildABot")
if _BA not in sys.path:
    sys.path.insert(0, _BA)

from clashroyalebuildabot.constants import (  # noqa: E402
    CARD_CONFIG,
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.detector import Detector  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402

from perception.base import register  # noqa: E402
from perception.schema import HandCard, PerceptionResult, Tower, Unit  # noqa: E402
from perception.vocab import card_id, normalize  # noqa: E402

# The account's current 8-card deck (the hand detector needs the deck to ID the
# 4 in-hand cards). Override via BuildABotPerceptor(deck=[...]) for other decks.
DEFAULT_DECK = [
    Cards.KNIGHT, Cards.ARCHERS, Cards.MINIONS, Cards.ARROWS,
    Cards.FIREBALL, Cards.GIANT, Cards.MINIPEKKA, Cards.MUSKETEER,
]


@register("buildabot")
class BuildABotPerceptor:
    name = "buildabot"

    def __init__(self, deck=None):
        self.deck = list(deck) if deck else list(DEFAULT_DECK)
        self.detector = Detector(cards=self.deck)

    def detect(self, image) -> PerceptionResult:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        width, height = image.size

        det_img = image.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT))
        st = self.detector.run(det_img)
        sx, sy = width / SCREENSHOT_WIDTH, height / SCREENSHOT_HEIGHT

        def scale(box):  # box = (left, top, right, bottom) in 368x652
            left, top, right, bottom = box
            return [
                int(left * sx),
                int(top * sy),
                int((right - left) * sx),
                int((bottom - top) * sy),
            ]

        def to_units(dets, owner):
            out = []
            for d in dets:
                canon = normalize(d.unit.name)
                bb = scale(d.position.bbox)
                out.append(Unit(
                    name=canon or d.unit.name,
                    card_id=card_id(canon),
                    owner=owner,
                    confidence=round(float(d.position.conf), 3),
                    bbox_px=bb,
                    center_px=[bb[0] + bb[2] // 2, bb[1] + bb[3] // 2],
                    tile=[int(d.position.tile_x), int(d.position.tile_y)],
                ))
            return out

        allies = to_units(st.allies, "ally")
        enemies = to_units(st.enemies, "enemy")

        hand = []
        for slot in range(4):
            c = st.cards[slot + 1]               # cards[0] is the "next" card
            canon = normalize(c.name)
            box = scale(CARD_CONFIG[slot + 1]) if len(CARD_CONFIG) > slot + 1 else [0, 0, 0, 0]
            hand.append(HandCard(
                slot=slot, name=canon or c.name, card_id=card_id(canon),
                ready=(slot in st.ready), confidence=1.0, bbox_px=box,
            ))

        n = st.numbers
        towers = [
            Tower("enemy", "princess_left", int(n.left_enemy_princess_hp.number)),
            Tower("enemy", "princess_right", int(n.right_enemy_princess_hp.number)),
            Tower("ally", "princess_left", int(n.left_ally_princess_hp.number)),
            Tower("ally", "princess_right", int(n.right_ally_princess_hp.number)),
        ]
        opp_seen = sorted({u.name for u in enemies if u.card_id >= 0})

        return PerceptionResult(
            backend=self.name,
            image_size=[width, height],
            screen=getattr(st.screen, "name", str(st.screen)),
            elixir=float(n.elixir.number),
            hand=hand, allies=allies, enemies=enemies, towers=towers,
            opponent_cards_seen=opp_seen,
        )
