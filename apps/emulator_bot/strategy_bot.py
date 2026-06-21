#!/usr/bin/env python3
"""
Strategic Clash Royale agent (v2).

Built on ClashRoyaleBuildABot's perception (card ID + YOLO units + tower HP +
screen state) with a priority-tiered decision policy grounded in the curated
strategy database.

Design goals (addressing v1's flaws):
  - Act with PURPOSE, not on a timer. Most ticks => WAIT. No spam.
  - Defend the cheapest hard counter -> positive elixir trades.
  - Counter-push with survivors; commit the win condition only when safe.
  - Spells only for value. Never leak at 10 elixir.
  - Log (state, action, reward) every decision for offline learning.

Capture path uses `adb exec-out screencap` (PNG) to stay clear of PyAV and the
`keyboard` lib (which crashes Pillow JPEG on macOS). Coordinates are computed in
BuildABot's 720x1280 space and scaled to the real device in click().
"""
import io
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from clashroyalebuildabot.constants import (
    DISPLAY_CARD_DELTA_X,
    DISPLAY_CARD_HEIGHT,
    DISPLAY_CARD_INIT_X,
    DISPLAY_CARD_WIDTH,
    DISPLAY_CARD_Y,
    DISPLAY_HEIGHT,
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)
from clashroyalebuildabot.detectors.detector import Detector
from clashroyalebuildabot.namespaces.cards import Cards
from clashroyalebuildabot.namespaces.screens import Screens
from clashroyalebuildabot.namespaces.units import Target, Transport
from PIL import Image

DEVICE = "127.0.0.1:5555"
DEV_W, DEV_H = 1080, 1920
RUNS = Path.home() / "clash-royale-bot" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
DECISION_LOG = RUNS / "decisions.jsonl"
MATCH_LOG = RUNS / "matches_v2.jsonl"

DECK = [
    Cards.KNIGHT, Cards.ARCHERS, Cards.MINIONS, Cards.ARROWS,
    Cards.FIREBALL, Cards.GIANT, Cards.MINIPEKKA, Cards.MUSKETEER,
]

# Cards in our deck that can hit air, by name preference (high dps first)
AIR_COUNTERS = ["musketeer", "archers", "minions"]
# Single-target ground answers for tanks / win conditions
TANK_COUNTERS = ["mini_pekka", "knight", "minions"]
SPELLS = {"arrows": 3, "fireball": 4}
WIN_CONDITION = "giant"
CHEAP_CYCLE = ["minions", "archers", "knight"]  # never a spell as cycle

OUR_SIDE_MAX_Y = 16  # enemy with tile_y <= this is on/over our side
DEPLOY_X = (1, 16)
DEPLOY_Y = (3, 13)


def _jd(o):
    """JSON default: coerce numpy scalars to native Python."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def adb(*args, binary=False):
    out = subprocess.run(["adb", "-s", DEVICE, *args], capture_output=True)
    return out.stdout if binary else out.stdout.decode("utf-8", "ignore")


def grab():
    # ADB screencap occasionally returns a truncated/empty buffer while the
    # emulator is busy. Retry instead of crashing a long unattended run.
    for _ in range(5):
        raw = adb("exec-out", "screencap", "-p", binary=True)
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            time.sleep(0.5)
    return None


def click720(x, y):
    sx = int(round(x * DEV_W / 720))
    sy = int(round(y * DEV_H / 1280))
    adb("shell", "input", "swipe", str(sx), str(sy), str(sx), str(sy), "90")


def tile_centre(tx, ty):
    x = TILE_INIT_X + (tx + 0.5) * TILE_WIDTH
    y = DISPLAY_HEIGHT - TILE_INIT_Y - (ty + 0.5) * TILE_HEIGHT
    return x, y


def card_centre(i):
    x = DISPLAY_CARD_INIT_X + DISPLAY_CARD_WIDTH / 2 + i * DISPLAY_CARD_DELTA_X
    y = DISPLAY_CARD_Y + DISPLAY_CARD_HEIGHT / 2
    return x, y


def play(slot, tx, ty):
    cx, cy = card_centre(slot)
    click720(cx, cy)
    time.sleep(0.15)
    px, py = tile_centre(tx, ty)
    click720(px, py)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ----------------------- strategic policy -----------------------
def _hand(state):
    """Affordable cards: list of (slot 0-3, Card)."""
    return [(s, state.cards[s + 1]) for s in state.ready]


def _find(hand, names):
    for name in names:
        for slot, card in hand:
            if card.name == name:
                return slot, card
    return None


def _enemies_our_side(state):
    return [e for e in state.enemies if e.position.tile_y <= OUR_SIDE_MAX_Y]


def _lane_x(tile_x):
    return "left" if tile_x < 9 else "right"


def decide(state):
    """Return (slot, tile_x, tile_y, reason) or None to hold.

    This is the swappable seam. A learned policy implements the same signature.
    """
    elixir = state.numbers.elixir.number
    hand = _hand(state)
    if not hand:
        return None

    threats = _enemies_our_side(state)

    # ---- TIER 1: DEFEND the most urgent threat with the cheapest hard counter
    if threats:
        threat = min(threats, key=lambda e: e.position.tile_y)  # closest to our king
        tx, ty = threat.position.tile_x, threat.position.tile_y
        is_air = threat.unit.transport == Transport.AIR
        lane = _lane_x(tx)
        same_lane = [e for e in threats if _lane_x(e.position.tile_x) == lane]
        is_swarm = len(same_lane) >= 3
        is_tank = (
            threat.unit.target == Target.BUILDINGS
            or threat.unit.name in {
                "giant", "hog_rider", "royal_giant", "golem", "pekka",
                "mini_pekka", "prince", "knight", "valkyrie", "giant_skeleton",
                "balloon",
            }
        )

        if is_air:
            pick = _find(hand, AIR_COUNTERS)
            if pick:
                slot, card = pick
                if card.name == "minions":  # drop on top
                    return slot, clamp(tx, *DEPLOY_X), clamp(ty, *DEPLOY_Y), f"defend-air:minions-on {threat.unit.name}"
                # ranged: place a few tiles back so it gets free shots
                return slot, clamp(tx, *DEPLOY_X), clamp(ty - 4, *DEPLOY_Y), f"defend-air:{card.name} vs {threat.unit.name}"
            arrows = _find(hand, ["arrows"])
            if arrows and is_swarm:
                slot, _ = arrows
                return slot, clamp(tx, *DEPLOY_X), clamp(ty, 4, 28), "defend-air:arrows-swarm"

        if is_swarm:
            spell = _find(hand, ["arrows", "fireball"])
            if spell:
                slot, _ = spell
                return slot, clamp(tx, *DEPLOY_X), clamp(ty, 4, 28), f"defend-swarm:{state.cards[slot+1].name}"

        if is_tank:
            pick = _find(hand, TANK_COUNTERS)
            if pick:
                slot, card = pick
                if card.name == "minions":
                    return slot, clamp(tx, *DEPLOY_X), clamp(ty, *DEPLOY_Y), f"defend-tank:minions vs {threat.unit.name}"
                # melee: place just in front of the threat's path
                return slot, clamp(tx, *DEPLOY_X), clamp(ty - 1, *DEPLOY_Y), f"defend-tank:{card.name} vs {threat.unit.name}"

        # generic ground threat -> any reasonable body
        pick = _find(hand, ["knight", "mini_pekka", "archers", "musketeer", "minions"])
        if pick:
            slot, card = pick
            return slot, clamp(tx, *DEPLOY_X), clamp(ty - 1, *DEPLOY_Y), f"defend:{card.name} vs {threat.unit.name}"

    # ---- TIER 2: SPELL FOR VALUE on a cluster of enemy support (anywhere)
    spell = _find(hand, ["fireball", "arrows"])
    if spell and len(state.enemies) >= 3:
        # find the densest 3-cluster centroid
        best = _cluster_centroid(state.enemies, radius=3)
        if best is not None:
            cx, cy, n = best
            slot, card = spell
            need = 3 if card.name == "arrows" else 3
            if n >= need:
                return slot, clamp(cx, 1, 16), clamp(cy, 4, 28), f"spell-value:{card.name} on {n} units"

    # ---- TIER 3: ATTACK / BUILD A PUSH. Do NOT sit at high elixir (that
    # wastes it). Be proactive: support an advancing giant, commit the win
    # condition, or build a fresh push behind the weaker tower.
    giant = _find(hand, [WIN_CONDITION])
    ally_giant = any(a.unit.name == "giant" for a in state.allies)
    lhp = state.numbers.left_enemy_princess_hp.number
    rhp = state.numbers.right_enemy_princess_hp.number
    push_lane_x = 3 if lhp <= rhp else 14
    if not threats:
        # 1) feed an advancing giant with support (cheap, high value)
        if ally_giant and elixir >= 5:
            support = _find(hand, ["musketeer", "archers", "minions", "knight"])
            if support:
                g = max((a for a in state.allies if a.unit.name == "giant"),
                        key=lambda a: a.position.tile_y)
                slot, _ = support
                return slot, clamp(g.position.tile_x, *DEPLOY_X), clamp(g.position.tile_y - 3, *DEPLOY_Y), "attack:support-giant"
        # 2) commit the win condition when we can support it
        if giant and elixir >= 8:
            slot, _ = giant
            return slot, push_lane_x, 4, "attack:giant-at-back"
        # 3) PROACTIVE BUILD: at 7+ elixir with nothing to defend, start a push
        #    instead of leaking. Lead with a mini-tank, support with ranged.
        if elixir >= 7:
            pick = _find(hand, ["mini_pekka", "knight", "musketeer", "archers"])
            if pick:
                slot, card = pick
                depth = 5 if card.name in ("mini_pekka", "knight") else 4
                return slot, push_lane_x, depth, f"build:{card.name}"

    # ---- TIER 4: ANTI-LEAK. Never sit near the cap; cycle the cheapest card.
    if elixir >= 8:
        pick = _find(hand, CHEAP_CYCLE) or (hand[0] if hand else None)
        if pick:
            slot, card = pick
            return slot, push_lane_x, 4, f"anti-leak:{card.name}"

    # ---- otherwise HOLD (only when low on elixir / banking for defense)
    return None


def _cluster_centroid(units, radius=3):
    best = None
    for u in units:
        ux, uy = u.position.tile_x, u.position.tile_y
        near = [v for v in units
                if abs(v.position.tile_x - ux) <= radius
                and abs(v.position.tile_y - uy) <= radius]
        if best is None or len(near) > best[2]:
            cx = sum(v.position.tile_x for v in near) / len(near)
            cy = sum(v.position.tile_y for v in near) / len(near)
            best = (int(cx), int(cy), len(near))
    return best


# ----------------------- logging + outcome -----------------------
def state_features(state):
    n = state.numbers
    return {
        "elixir": n.elixir.number,
        "towers": {
            "enemy_l": n.left_enemy_princess_hp.number,
            "enemy_r": n.right_enemy_princess_hp.number,
            "ally_l": n.left_ally_princess_hp.number,
            "ally_r": n.right_ally_princess_hp.number,
        },
        "hand": [c.name for c in state.cards[1:]],
        "ready": list(state.ready),
        "n_allies": len(state.allies),
        "n_enemies": len(state.enemies),
        "enemies_our_side": [
            {"name": e.unit.name, "x": e.position.tile_x, "y": e.position.tile_y,
             "air": e.unit.transport == Transport.AIR}
            for e in _enemies_our_side(state)
        ],
    }


def crowns_from(state):
    n = state.numbers
    enemy_down = sum(1 for v in [n.left_enemy_princess_hp.number, n.right_enemy_princess_hp.number] if v == 0)
    ally_down = sum(1 for v in [n.left_ally_princess_hp.number, n.right_ally_princess_hp.number] if v == 0)
    return enemy_down, ally_down


# ----------------------- detector (built lazily, once) -----------------------
_DET = None


def detector():
    global _DET
    if _DET is None:
        _DET = Detector(cards=list(DECK))
    return _DET


def detect(img):
    det_img = img.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT))
    return detector().run(det_img)


# ----------------------- match flow -----------------------
# device-coord Battle button (1080x1920): centroid measured from a real lobby
BATTLE_BTN_DEV = (559, 1492)


def battle_button_visible(img):
    """Robust lobby check: is the yellow Battle button present? (raw 1080x1920)
    BuildABot's screen_detector mis-classifies some lobby states as 'unknown',
    so we detect the button by colour instead."""
    a = np.asarray(img)
    if a.shape[0] < 1600:
        return False
    crop = a[1400:1580, 400:720].astype(int)
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    # gold/amber button; capture both bright (255,180,30) and dark (178,131,0)
    gold = (r > 150) & (g > 95) & (g < 205) & (b < 75) & (r - g > 25)
    return gold.mean() > 0.15


def tap_battle_raw():
    x, y = BATTLE_BTN_DEV
    adb("shell", "input", "swipe", str(x), str(y), str(x), str(y), "90")


def goto_battle(timeout=150):
    """From wherever we are, reach an in-game battle.

    Lobby is detected by the yellow Battle button (robust), not the flaky
    screen_detector. Unknown screens that persist (offer/season-pass popups)
    are dismissed via BACK + top-right close-X.
    """
    t0 = time.time()
    pressed = 0
    unknown_ticks = 0
    while time.time() - t0 < timeout:
        img = grab()
        if img is None:
            time.sleep(0.5)
            continue
        st = detect(img)
        s = st.screen.name if st is not None else "unknown"
        if s == "in_game":
            return True
        if s == "end_of_game":
            unknown_ticks = 0
            click720(*Screens.END_OF_GAME.click_xy)
            time.sleep(2.0)
        elif s == "bypass_end_of_game":
            unknown_ticks = 0
            click720(*Screens.BYPASS_END_OF_GAME.click_xy)
            time.sleep(2.0)
        elif battle_button_visible(img):  # lobby (robust colour check)
            unknown_ticks = 0
            tap_battle_raw()
            pressed += 1
            print(f"  pressed Battle ({pressed})", flush=True)
            time.sleep(2.5)
        else:  # matchmaking OR a stuck popup
            unknown_ticks += 1
            if unknown_ticks >= 5:  # persisted -> likely a result/offer popup, dismiss it
                print("  unknown persisted, dismissing popup", flush=True)
                # Result screens need the OK button; offers often need a
                # top-right close-X; some screens only respond to BACK.
                for sx, sy in [(470, 1153), (360, 1160), (690, 180), (660, 300)]:
                    click720(sx, sy)
                    time.sleep(0.4)
                adb("shell", "input", "keyevent", "4")  # BACK
                unknown_ticks = 0
            time.sleep(1.2)
        time.sleep(0.6)
    return False


def play_match(idx):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    t0 = time.time()
    acts = defaultdict(int)
    last_crowns = (0, 0)
    holds = 0
    decisions = 0
    last_act = 0.0
    while True:
        img = grab()
        if img is None:
            time.sleep(0.4)
            continue
        st = detect(img)
        if st is None:
            time.sleep(0.4)
            continue
        if st.screen.name != "in_game":
            break
        if time.time() - t0 > 260:
            break
        last_crowns = crowns_from(st)
        now = time.time()
        # decision rate-limit: at most ~1 commit / 0.8s, but WAIT freely
        action = decide(st)
        if action and now - last_act > 0.8:
            slot, tx, ty, reason = action
            play(slot, tx, ty)
            acts[reason.split(":")[0]] += 1
            decisions += 1
            last_act = now
            rec = {"match": idx, "t": round(now - t0, 1), "action": reason,
                   "slot": slot, "tile": [tx, ty], "state": state_features(st)}
            with open(DECISION_LOG, "a") as f:
                f.write(json.dumps(rec, default=_jd) + "\n")
            print(f"  [{idx}] t={now-t0:4.0f}s {reason}", flush=True)
        else:
            holds += 1
        time.sleep(0.45)

    end_img = grab()
    if end_img is not None:
        end_img.save(RUNS / f"v2_m{idx}_{ts}_end.png")
    ec, ac = last_crowns
    result = "win" if ec > ac else ("loss" if ac > ec else "draw")
    rec = {"match": idx, "ts": ts, "result": result, "our_crowns": ec,
           "enemy_crowns": ac, "decisions": decisions, "holds": holds,
           "duration_s": round(time.time() - t0, 1), "action_mix": dict(acts)}
    with open(MATCH_LOG, "a") as f:
        f.write(json.dumps(rec, default=_jd) + "\n")
    print(f"[{idx}] {result.upper()} crowns {ec}-{ac} | {decisions} plays, "
          f"{holds} holds | mix={dict(acts)}", flush=True)
    return rec


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"=== Strategic CR agent v2: {n} matches ===", flush=True)
    print("Building detector (card hashes + onnx)...", flush=True)
    detector()
    results = []
    for i in range(1, n + 1):
        print(f"\n--- Match {i}/{n} ---", flush=True)
        if not goto_battle():
            print(f"  [{i}] could not reach battle", flush=True)
            continue
        results.append(play_match(i))
        # return to lobby: tap through result/chest screens until Battle button
        for _ in range(12):
            img = grab()
            if battle_button_visible(img):
                break
            st = detect(img)
            s = st.screen.name if st is not None else "unknown"
            if s == "end_of_game":
                click720(*Screens.END_OF_GAME.click_xy)
            elif s == "bypass_end_of_game":
                click720(*Screens.BYPASS_END_OF_GAME.click_xy)
            else:
                click720(*Screens.END_OF_GAME.click_xy)
            time.sleep(1.5)
    wins = sum(1 for r in results if r["result"] == "win")
    print(f"\n=== Done. {wins}/{len(results)} wins. Logs: {MATCH_LOG} ===", flush=True)


if __name__ == "__main__":
    main()
