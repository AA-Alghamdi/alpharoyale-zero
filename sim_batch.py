#!/usr/bin/env python3
"""
Large-scale diverse-simulation harness.

Plays thousands of CR sim games across diverse decks (sampled from the 125
cards), mixed agents, and varied openings (seed + decision interval), in
parallel. Logs every game and aggregates:
  - agent win rates (relative strength)
  - per-card win contribution (which cards correlate with winning)
  - deck-diversity coverage

This is the data-generation + strategy-exploration engine the emulator can't be
(real-time). Output -> runs_sim/.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from crsim.actions import action_id_to_action as _action_from_id
from crsim.cards import CARD_DEFS
from crsim.game import Action, CRGame, GameResult
from eval.baseline_agents import (
    HeuristicAgent,
    MetaAgent,
    RandomAgent,
)

OUT = Path(__file__).parent / "runs_sim"
OUT.mkdir(exist_ok=True)

CARD_KEYS = list(CARD_DEFS.keys())
AGENTS = {"meta": MetaAgent, "heuristic": HeuristicAgent, "random": RandomAgent}


def card_name(k):
    cd = CARD_DEFS[k]
    return getattr(cd, "name", str(k))


def _cost(k):
    cd = CARD_DEFS[k]
    return getattr(cd, "cost", getattr(cd, "elixir", 4))


def sample_deck(rng):
    """Sample an 8-card deck with a sane elixir spread (avg 3.0-4.6)."""
    for _ in range(40):
        deck = rng.sample(CARD_KEYS, 8)
        avg = sum(_cost(k) for k in deck) / 8
        if 3.0 <= avg <= 4.6:
            return deck
    return rng.sample(CARD_KEYS, 8)


def play_one(cfg):
    """Play a single game. cfg = (deck0, deck1, ag0, ag1, seed, interval)."""
    deck0, deck1, ag0, ag1, seed, interval = cfg
    try:
        a0 = AGENTS[ag0]()
        a1 = AGENTS[ag1]()
        game = CRGame(deck_p0=deck0, deck_p1=deck1, seed=seed)
        ticks = 0
        while not game.done:
            if game.tick_count % interval == 0:
                act0 = _action_from_id(a0.select_action(game, 0), 0)
                act1 = _action_from_id(a1.select_action(game, 1), 1)
            else:
                act0 = Action(player=0, hand_slot=-1)
                act1 = Action(player=1, hand_slot=-1)
            game.step([act0, act1])
            ticks += 1
            if ticks > 20000:
                break
        if game.result == GameResult.P0_WIN:
            winner = 0
        elif game.result == GameResult.P1_WIN:
            winner = 1
        else:
            winner = -1
        return {
            "ok": True, "winner": winner, "ag0": ag0, "ag1": ag1,
            "deck0": [card_name(c) for c in deck0], "deck1": [card_name(c) for c in deck1],
            "ticks": ticks, "interval": interval,
        }
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


def build_configs(n, rng):
    # diverse deck pool
    pool = [sample_deck(rng) for _ in range(max(16, n // 20))]
    agent_pairs = [
        ("meta", "random"), ("meta", "heuristic"), ("heuristic", "random"),
        ("meta", "meta"), ("heuristic", "heuristic"), ("random", "random"),
    ]
    intervals = [4, 8, 12]  # varied "tempo"/openings
    cfgs = []
    for i in range(n):
        d0 = rng.choice(pool)
        d1 = rng.choice(pool)
        ag0, ag1 = rng.choice(agent_pairs)
        cfgs.append((d0, d1, ag0, ag1, i, rng.choice(intervals)))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfgs = build_configs(args.games, rng)
    print(f"Running {len(cfgs)} games on {args.procs} procs...", flush=True)

    t0 = time.time()
    results = []
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(play_one, cfgs, chunksize=16)):
            results.append(r)
            if (i + 1) % 250 == 0:
                ok = sum(1 for x in results if x.get("ok"))
                print(f"  {i+1}/{len(cfgs)} done ({ok} ok) {time.time()-t0:.0f}s", flush=True)
    dt = time.time() - t0

    ok = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    ts = f"seed{args.seed}_{len(ok)}games"
    with open(OUT / f"games_{ts}.jsonl", "w") as f:
        for r in ok:
            f.write(json.dumps(r) + "\n")

    # ---- aggregate ----
    agent_games = defaultdict(lambda: [0, 0])  # agent -> [wins, games]
    card_wl = defaultdict(lambda: [0, 0])  # card -> [wins, games]
    for r in ok:
        w = r["winner"]
        for side, ag, deck in [(0, r["ag0"], r["deck0"]), (1, r["ag1"], r["deck1"])]:
            agent_games[ag][1] += 1
            won = (w == side)
            if won:
                agent_games[ag][0] += 1
            for c in deck:
                card_wl[c][1] += 1
                if won:
                    card_wl[c][0] += 1

    print(f"\n=== {len(ok)} games in {dt:.0f}s ({dt/max(len(ok),1)*1000:.0f} ms/game, "
          f"{len(ok)/dt:.0f} games/s); {len(errs)} errors ===")
    if errs:
        print("  sample error:", errs[0]["err"])
    print("\nAGENT WIN RATES:")
    for ag, (w, g) in sorted(agent_games.items(), key=lambda x: -x[1][0]/max(x[1][1],1)):
        print(f"  {ag:10} {w/max(g,1):.1%}  ({w}/{g})")
    # top/bottom cards by win correlation (min games)
    ranked = [(c, w/g, g) for c, (w, g) in card_wl.items() if g >= 30]
    ranked.sort(key=lambda x: -x[1])
    print("\nTOP 12 CARDS by win-rate (>=30 games):")
    for c, wr, g in ranked[:12]:
        print(f"  {c:28} {wr:.1%}  ({g})")
    print("\nBOTTOM 8 CARDS:")
    for c, wr, g in ranked[-8:]:
        print(f"  {c:28} {wr:.1%}  ({g})")

    summary = {
        "games": len(ok), "errors": len(errs), "seconds": dt,
        "agent_winrates": {ag: w/max(g,1) for ag, (w, g) in agent_games.items()},
        "card_winrates": {c: {"wr": w/g, "games": g} for c, (w, g) in card_wl.items() if g >= 30},
    }
    json.dump(summary, open(OUT / f"summary_{ts}.json", "w"), indent=1)
    print(f"\nSaved -> runs_sim/games_{ts}.jsonl + summary_{ts}.json")


if __name__ == "__main__":
    main()
