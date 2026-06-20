# Conformance harness

GPU-free tooling to measure how accurate the simulator's card stats are, by
comparing them against two independent oracles:

1. **Reality** — hand-verified Level-11 wiki anchors (`wiki_anchors.py`).
   Authoritative but narrow. Enforced as a regression test in
   `tests/test_conformance_anchors.py`.
2. **An independent engine** — [samdickson22/clash-simulator][ref], a separately
   authored Python CR engine with its own `gamedata.json`. Agreement is strong
   evidence of correctness; disagreement localises a discrepancy to a card/stat.

## Run

```bash
# Human-readable report (anchors always; cross-engine if the reference is present)
python -m tools.conformance.compare_stats

# Point at a clone of the reference engine
CLASHER_DIR=/path/to/clash-simulator python -m tools.conformance.compare_stats

# Write the markdown report to a file
python -m tools.conformance.compare_stats --out conformance_report.md
```

The reference engine is optional. To enable the cross-engine section:

```bash
git clone https://github.com/samdickson22/clash-simulator ~/refs/clash-simulator
```

## Interpreting the output

- **Anchor mismatches** mean our value disagrees with verified reality — either a
  bug or balance-patch staleness (see `KNOWN_STALE` in the test).
- **Cross-engine signed HP median** captures the *systematic* offset between the
  two engines' level-scaling conventions (currently ≈ −1.3%); it is not a
  per-card bug.
- **Outliers** (|Δ| > 5%) are the per-card items worth investigating. Many are
  representation differences (e.g. Arrows total-vs-per-wave damage, multi-unit
  cards like Rascals) rather than errors — see `CONFORMANCE.md` for the
  annotated list.

[ref]: https://github.com/samdickson22/clash-simulator
