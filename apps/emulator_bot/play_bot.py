#!/usr/bin/env python3
"""
Autonomous Clash Royale bot (v1, heuristic).

Pipeline:  adb screencap -> state read (elixir bar + lane activity) ->
           elixir-aware card deploy -> ADB taps.  Loops over N matches,
           handles the menu/result screens, logs every match to JSONL.

This is the Stage-1 heuristic player. The decision function `choose_action`
is the clean seam where a learned policy will later be dropped in.

Coordinates are calibrated for a 1080x1920 portrait device (BlueStacks /
Galaxy S21 profile).
"""
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

DEVICE = "127.0.0.1:5555"
W, H = 1080, 1920
RUNS = Path.home() / "clash-royale-bot" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
LOG = RUNS / "matches.jsonl"

# --- calibrated UI coordinates (1080x1920) ---
BATTLE_BTN = (540, 1530)          # menu "Battle" button
CARD_SLOTS = [(450, 1770), (640, 1770), (810, 1770), (980, 1770)]
LANES = {"left": 360, "right": 700}
DEPLOY_Y_ATTACK = 1090            # just behind our side of the bridge
DEPLOY_Y_DEFEND = 1250            # closer to our towers
# elixir bar: a horizontal magenta strip near the bottom
# filled = bright pink, empty = dark blue (4,50,118)
ELIXIR_ROW = 1880
ELIXIR_X0, ELIXIR_X1 = 340, 990
# region to sniff the yellow "Battle" button (menu detection)
BATTLE_REGION = (470, 1490, 610, 1570)


def adb(*args, binary=False):
    cmd = ["adb", "-s", DEVICE, *args]
    out = subprocess.run(cmd, capture_output=True, check=False)
    return out.stdout if binary else out.stdout.decode("utf-8", "ignore")


def screencap():
    raw = adb("exec-out", "screencap", "-p", binary=True)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def tap(x, y):
    # BlueStacks ignores instantaneous `input tap`; a short press (swipe in
    # place with a duration) registers reliably.
    x, y = str(int(x)), str(int(y))
    adb("shell", "input", "swipe", x, y, x, y, "90")


def _bar_masks(img):
    arr = np.asarray(img)
    row = arr[ELIXIR_ROW, ELIXIR_X0:ELIXIR_X1, :].astype(int)
    r, g, b = row[:, 0], row[:, 1], row[:, 2]
    filled = (r > 150) & (b > 150) & (g < 220)          # bright pink fill
    empty = (r < 45) & (g > 30) & (g < 100) & (b > 90) & (b < 170)  # dark blue
    return filled, empty


def read_elixir(img):
    """Estimate current elixir (0-10) from the magenta bar fill fraction."""
    filled, _ = _bar_masks(img)
    return round(filled.mean() * 10)


def at_menu(img):
    """True if the yellow Battle button is visible."""
    x0, y0, x1, y1 = BATTLE_REGION
    crop = np.asarray(img)[y0:y1, x0:x1, :].astype(int)
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    yellow = (r > 200) & (g > 140) & (b < 90)
    return yellow.mean() > 0.15


def _elixir_bar_present(img):
    filled, empty = _bar_masks(img)
    return (filled.mean() + empty.mean()) > 0.45


def in_match(img):
    """True if the elixir bar is present and we're not on the menu."""
    return _elixir_bar_present(img) and not at_menu(img)


def choose_lane(img):
    """Pick the lane with more enemy activity (defend it / counter there)."""
    arr = np.asarray(img).astype(int)
    # enemy half, just above the river, split L/R
    band = arr[640:780, :, :]
    left = band[:, : W // 2]
    right = band[:, W // 2 :]
    # 'activity' ~ local color variance vs the green/blue board
    lvar = left.std()
    rvar = right.std()
    return "left" if lvar >= rvar else "right"


def choose_action(img, elixir, slot_cycle):
    """STAGE-1 HEURISTIC POLICY.  <-- learned policy swaps in here.

    Returns (slot_index, (x, y)) or None to hold.
    Rule: only commit when elixir >= 5 (any starter card affordable),
    then deploy into the contested lane just behind our bridge.
    """
    if elixir < 5:
        return None
    slot = slot_cycle[0]
    lane = choose_lane(img)
    x = LANES[lane]
    y = DEPLOY_Y_ATTACK
    return slot, (x, y), lane


def deploy(slot, pos):
    sx, sy = CARD_SLOTS[slot]
    tap(sx, sy)
    time.sleep(0.18)
    tap(*pos)


def start_match():
    """From the menu, press Battle and wait until the battle starts.

    Crucial: while matchmaking (neither menu nor match on screen) we must
    just WAIT. Tapping there cancels the search and bounces us to the menu.
    """
    pressed = False
    for _ in range(60):
        img = screencap()
        if in_match(img):
            return True
        if at_menu(img) and not pressed:
            print("  pressing Battle...", flush=True)
            tap(*BATTLE_BTN)
            pressed = True
            time.sleep(2.5)
        elif at_menu(img) and pressed:
            # bounced back to menu (e.g. opponent declined) -> press again
            tap(*BATTLE_BTN)
            time.sleep(2.5)
        else:
            # matchmaking / loading -> wait, do NOT tap
            time.sleep(1.2)
        time.sleep(1.0)
    return False


def back_to_menu(timeout=70):
    """After a match, tap through result/chest screens until the menu."""
    t0 = time.time()
    step = 0
    while time.time() - t0 < timeout:
        img = screencap()
        if at_menu(img):
            return True
        # cycle through the spots that advance post-game screens
        spots = [(540, 1740), (540, 850), (540, 1530)]
        tap(*spots[step % len(spots)])
        if step % 4 == 3:
            adb("shell", "input", "keyevent", "4")  # BACK
        step += 1
        time.sleep(1.3)
    return False


def play_match(match_idx):
    """Play one battle to completion. Returns a result dict."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    deploys = 0
    slot_cycle = [0, 1, 2, 3]
    last_deploy = 0.0
    t0 = time.time()
    mid_frame_saved = False
    while True:
        img = screencap()
        if not in_match(img):
            # match ended
            break
        if time.time() - t0 > 240:  # safety cap (~match length + OT)
            break
        elixir = read_elixir(img)
        now = time.time()
        # save one mid-match frame for the user to watch
        if not mid_frame_saved and now - t0 > 30:
            img.save(RUNS / f"m{match_idx}_{ts}_mid.png")
            mid_frame_saved = True
        if now - last_deploy > 2.2:
            act = choose_action(img, elixir, slot_cycle)
            if act:
                slot, pos, lane = act
                deploy(slot, pos)
                deploys += 1
                slot_cycle = slot_cycle[1:] + slot_cycle[:1]
                last_deploy = now
                print(f"  [m{match_idx}] deploy #{deploys} slot{slot}->{lane} "
                      f"(elixir~{elixir})")
        time.sleep(0.5)
    end_img = screencap()
    end_path = RUNS / f"m{match_idx}_{ts}_end.png"
    end_img.save(end_path)
    dur = round(time.time() - t0, 1)
    result = {
        "match": match_idx,
        "ts": ts,
        "deploys": deploys,
        "duration_s": dur,
        "end_screenshot": str(end_path),
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[m{match_idx}] done: {deploys} deploys in {dur}s -> {end_path.name}")
    return result


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"=== Autonomous CR bot: playing {n} matches on {DEVICE} ===")
    results = []
    for i in range(1, n + 1):
        print(f"\n--- Match {i}/{n}: navigating to battle ---")
        if not back_to_menu(timeout=40):
            print("  (could not confirm menu; trying to start anyway)")
        if not start_match():
            print(f"  [m{i}] could not start a match; skipping")
            continue
        print(f"  [m{i}] battle started, playing...")
        results.append(play_match(i))
    print(f"\n=== Finished. {len(results)} matches played. Log: {LOG} ===")


if __name__ == "__main__":
    main()
