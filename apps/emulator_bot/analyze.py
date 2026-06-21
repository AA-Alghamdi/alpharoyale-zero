#!/usr/bin/env python3
"""Analyze v2 match + decision logs: win rate, action mix, crown timing,
and simple weakness signals to guide the next policy iteration."""
import json
from collections import Counter
from pathlib import Path

RUNS = Path.home() / "clash-royale-bot" / "runs"


def load(p):
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main():
    matches = load(RUNS / "matches_v2.jsonl")
    decisions = load(RUNS / "decisions.jsonl")

    print(f"=== {len(matches)} matches, {len(decisions)} decisions ===\n")

    if matches:
        wins = sum(1 for m in matches if m["result"] == "win")
        losses = sum(1 for m in matches if m["result"] == "loss")
        draws = sum(1 for m in matches if m["result"] == "draw")
        print(f"RECORD: {wins}W-{losses}L-{draws}D  (win rate {wins/len(matches):.0%})")
        cr_for = sum(m["our_crowns"] for m in matches)
        cr_against = sum(m["enemy_crowns"] for m in matches)
        print(f"CROWNS: {cr_for} for / {cr_against} against "
              f"({cr_for/len(matches):.1f} vs {cr_against/len(matches):.1f} per match)")
        avg_plays = sum(m["decisions"] for m in matches) / len(matches)
        avg_holds = sum(m["holds"] for m in matches) / len(matches)
        print(f"TEMPO: {avg_plays:.0f} plays / {avg_holds:.0f} holds per match "
              f"(commit rate {avg_plays/(avg_plays+avg_holds):.0%})\n")
        print("PER MATCH:")
        for m in matches:
            print(f"  m{m['match']}: {m['result']:4} {m['our_crowns']}-{m['enemy_crowns']} "
                  f"| {m['decisions']} plays | mix={m.get('action_mix', {})}")
        print()

    if decisions:
        tiers = Counter(d["action"].split(":")[0] for d in decisions)
        print("ACTION MIX (all decisions):")
        for tier, n in tiers.most_common():
            print(f"  {tier:14} {n:4}  ({n/len(decisions):.0%})")
        # what threats are we facing / countering
        cards_played = Counter(
            d["state"]["hand"][d["slot"]] if d["slot"] < len(d["state"]["hand"]) else "?"
            for d in decisions
        )
        print("\nCARDS DEPLOYED:")
        for c, n in cards_played.most_common():
            print(f"  {c:12} {n}")
        # elixir at decision time (are we acting too rich/too poor?)
        elx = [d["state"]["elixir"] for d in decisions]
        print(f"\nELIXIR @ play: avg {sum(elx)/len(elx):.1f}, "
              f"min {min(elx)}, max {max(elx)}")
        # threats seen
        threat_cards = Counter(
            e["name"] for d in decisions for e in d["state"].get("enemies_our_side", [])
        )
        print("\nENEMY THREATS FACED (on our side, top 10):")
        for c, n in threat_cards.most_common(10):
            print(f"  {c:14} {n}")


if __name__ == "__main__":
    main()
