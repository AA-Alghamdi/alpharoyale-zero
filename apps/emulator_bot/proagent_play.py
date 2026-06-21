"""Run the real ProAgent (clash-royale-zero's strong scripted agent) on the LIVE
BlueStacks game.

Bridge: live frame -> BuildABot detector (State) -> a faithful shim exposing the
fields ProAgent reads from a CRGame -> ProAgent.select_action -> decode to
(slot, x, y) -> strategy_bot's tap infra (tile->pixel x1.5 press).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Use the monorepo root for simulator packages and baseline policies.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load ProAgent directly from the file: importing the `eval` package triggers
# eval/__init__ -> evaluator -> torch (absent in BuildABot's venv). baseline_agents
# itself only needs crsim (torch-free).
import importlib.util as _ilu  # noqa: E402

from crsim.actions import action_id_to_action  # noqa: E402
from crsim.cards import CARD_DEFS, CardType, EntityKind, TargetMode  # noqa: E402
from crsim.constants import ARENA_H, ARENA_W, NUM_HAND_SLOTS, WAIT_ACTION  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "cr_baseline", os.path.join(_ROOT, "eval", "baseline_agents.py"))
_bmod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bmod)
ProAgent = _bmod.ProAgent

# reuse the working live-game infra (capture/detect/tap/nav)
import strategy_bot as sb  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from perception.vocab import normalize  # noqa: E402

DECK = [CardType.KNIGHT, CardType.ARCHERS, CardType.MINIONS, CardType.ARROWS,
        CardType.FIREBALL, CardType.GIANT, CardType.MINI_PEKKA, CardType.MUSKETEER]
_DECK_INDEX = {c: i for i, c in enumerate(DECK)}
N_ACTIONS = NUM_HAND_SLOTS * ARENA_W * ARENA_H + 8
ENEMY_CONF_MIN = 0.30   # drop very-low-confidence YOLO phantoms (keep real units)
SPELL_BLAST_TILES = 3.5  # require an enemy within this of a spell target


def _ct(name):
    canon = normalize(name)
    if canon is None:
        return None
    try:
        return CardType[canon]
    except KeyError:
        return None


class _PS:
    def __init__(self, elixir, deck, hand):
        self.elixir = elixir
        self.deck = deck
        self.hand = hand


class _Ent:
    def __init__(self, owner, x, y, is_flying, target_mode, card_type, cd, conf=1.0):
        self.alive = True
        self.is_tower = False
        self.is_building = False
        self.is_king_tower = False
        self.owner = owner
        self.conf = conf
        self.x = float(x)
        self.y = float(y)
        self.is_flying = is_flying
        self.target_mode = target_mode
        self.card_type = card_type
        # carry common stat fields some agent versions read off entities
        self.hp = cd.hp if cd else 100.0
        self.max_hp = cd.hp if cd else 100.0
        self.dps = cd.dps if cd else 0.0
        self.cost = cd.cost if cd else 4


class _Tower:
    def __init__(self, x, hp):
        self.x = float(x)
        self.hp = float(hp)
        self.alive = hp > 0


class _GameShim:
    """Minimal CRGame surface that ProAgent reads."""

    def __init__(self, state):
        n = state.numbers
        # hand -> deck indices (cards[1:5]; cards[0] is the "next" card)
        hand = []
        for slot in range(4):
            ct = _ct(state.cards[slot + 1].name)
            hand.append(_DECK_INDEX.get(ct, 0))
        self.players = [_PS(float(n.elixir.number), DECK, hand),
                        _PS(0.0, DECK, [0, 1, 2, 3])]

        ents = []
        for det, owner in [(state.allies, 0), (state.enemies, 1)]:
            for d in det:
                conf = float(getattr(d.position, "conf", 1.0))
                # drop low-confidence enemy phantoms (they caused spell-on-empty)
                if owner == 1 and conf < ENEMY_CONF_MIN:
                    continue
                ct = _ct(d.unit.name)
                cd = CARD_DEFS.get(ct) if ct is not None else None
                ents.append(_Ent(
                    owner, d.position.tile_x, d.position.tile_y,
                    bool(cd.is_flying) if cd else False,
                    cd.target_mode if cd else TargetMode.GROUND,
                    ct if ct is not None else CardType.KNIGHT, cd, conf,
                ))
        self.entities = ents
        self.enemy_xy = [(e.x, e.y) for e in ents if e.owner == 1]
        self.princess_towers = [
            [_Tower(3, n.left_ally_princess_hp.number),
             _Tower(14, n.right_ally_princess_hp.number)],
            [_Tower(3, n.left_enemy_princess_hp.number),
             _Tower(14, n.right_enemy_princess_hp.number)],
        ]
        self.king_towers = [None, None]

    def get_valid_actions_mask(self, player):
        return np.ones(N_ACTIONS, dtype=bool)


def _is_spell(card_type):
    cd = CARD_DEFS.get(card_type)
    return cd is not None and cd.kind == EntityKind.SPELL


def _enemy_near(shim, x, y, radius=SPELL_BLAST_TILES):
    return any((ex - x) ** 2 + (ey - y) ** 2 <= radius * radius
               for ex, ey in shim.enemy_xy)


def main():
    n_matches = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    agent = ProAgent()
    print(f"=== ProAgent on live BlueStacks: {n_matches} match(es) ===", flush=True)
    sb.detector()  # build BuildABot detector once

    rec_path = os.path.expanduser("~/clash-royale-bot/runs/play_log.jsonl")
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)

    def record(payload):
        with open(rec_path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def state_summary(st):
        n = st.numbers
        return {
            "elixir": float(n.elixir.number),
            "hand": [c.name for c in st.cards[1:]],
            "towers": {
                "enemy_l": n.left_enemy_princess_hp.number,
                "enemy_r": n.right_enemy_princess_hp.number,
                "ally_l": n.left_ally_princess_hp.number,
                "ally_r": n.right_ally_princess_hp.number,
            },
            "allies": [
                {"u": a.unit.name, "x": a.position.tile_x, "y": a.position.tile_y}
                for a in st.allies
            ],
            "enemies": [
                {"u": e.unit.name, "x": e.position.tile_x, "y": e.position.tile_y}
                for e in st.enemies
            ],
        }

    print(f"recording every decision -> {rec_path}", flush=True)

    for m in range(1, n_matches + 1):
        print(f"\n--- Match {m}: navigating to battle ---", flush=True)
        if not sb.goto_battle():
            print("  could not reach battle", flush=True)
            continue
        print("  battle started; ProAgent playing", flush=True)
        t0 = time.time()
        last = 0.0
        plays = 0
        stuck_ticks = 0
        while True:
            img = sb.grab()
            if img is None:
                time.sleep(0.5)
                continue
            st = sb.detect(img)
            screen = st.screen.name if st is not None else "unknown"
            if screen != "in_game":
                # End only on a positive exit signal. Mid-battle detector flicker
                # must not abandon the game and leave the bot idle.
                if sb.battle_button_visible(img) or screen in ("end_of_game", "bypass_end_of_game"):
                    break
                stuck_ticks += 1
                if stuck_ticks > 90:  # ~45s of pure unknown means truly stuck.
                    break
                time.sleep(0.5)
                continue
            stuck_ticks = 0
            shim = _GameShim(st)
            try:
                aid = agent.select_action(shim, 0)
            except Exception as e:
                aid = WAIT_ACTION
                print("  (decide error)", e, flush=True)
            now = time.time()
            decided = {"event": "hold"}
            if aid != WAIT_ACTION and now - last > 0.8:
                a = action_id_to_action(aid, 0)
                ct = DECK[shim.players[0].hand[a.hand_slot]]
                card = ct.name
                # GUARD: never spend a spell on empty board — require a real
                # enemy within the blast radius of the target tile.
                if _is_spell(ct) and not _enemy_near(shim, a.x, a.y):
                    decided = {"event": "skip_spell", "card": card, "x": int(a.x), "y": int(a.y)}
                    print(f"  t={now-t0:4.0f}s SKIP {card} @({int(a.x)},{int(a.y)}) — no enemy in blast", flush=True)
                    last = now
                else:
                    sb.play(a.hand_slot, int(a.x), int(a.y))
                    plays += 1
                    last = now
                    decided = {
                        "event": "play",
                        "card": card,
                        "slot": a.hand_slot,
                        "x": int(a.x),
                        "y": int(a.y),
                    }
                    print(f"  t={now-t0:4.0f}s ProAgent: {card:14} @tile({int(a.x):2d},{int(a.y):2d}) "
                          f"elixir~{shim.players[0].elixir:.0f}", flush=True)
            record({"match": m, "t": round(now - t0, 1), "decision": decided, "state": state_summary(st)})
            time.sleep(0.4)
        duration_s = round(time.time() - t0, 1)
        record({"match": m, "event": "match_end", "plays": plays, "duration_s": duration_s})
        print(f"  match {m} ended after {plays} plays, {duration_s:.0f}s", flush=True)
    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
