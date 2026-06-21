# Cross-engine verification (R1)

Golden interaction tests (`tests/test_interactions.py`) check crsim against
*hand-written* expected outcomes. This package adds a second, independent check:
agreement with a **separately-built Clash Royale engine** — the "oracle",
[samdickson22/clash-simulator](https://github.com/samdickson22/clash-simulator).

Two independent decodings of authentic Supercell data agreeing on a card's stats,
and two independent simulators agreeing on who wins a duel, is much stronger
evidence of fidelity than either engine alone.

## What's here

| Module | Purpose | Runs in CI? |
|--------|---------|:-----------:|
| `conformance.py` | Kind-aware card-stat comparison (crsim `CARD_DEFS` vs oracle). | yes |
| `report.py` | Renders `reports/card_stat_conformance.md`. | — |
| `behavioral.py` | 1v1 duel comparison (winner + time-to-kill). | no (needs oracle clone) |
| `data/oracle_card_stats.json` | Vendored snapshot of the oracle's interpreted stats. | yes (read by CI) |
| `reports/card_stat_conformance.md` | Generated conformance report (committed for visibility). | — |

The CI gates live in `tests/test_cross_engine_stats.py` (stat conformance, runs
off the vendored fixture — no oracle needed) and `tests/test_cross_engine_behavioral.py`
(duels, auto-skipped unless an oracle clone is present).

## Stat conformance

`conformance.py` compares stats *kind-aware* (troops/buildings expose hp, damage,
hit-speed, range, sight, speed; spells expose a radius). It only scores a stat
when both engines express it comparably, converting units where they differ
(oracle speed is tiles/s × 30; oracle spell `range` is crsim's `splash_radius`).
Schema-incompatible stats (spell area damage, projectile-only troop damage) are
reported as `na`, not scored.

The test is a **regression gate**: the set of `(card, stat)` pairs that diverge
beyond tolerance must equal `KNOWN_DIVERGENCES` in
`tests/test_cross_engine_stats.py`. A crsim change that moves a stat away from the
oracle fails the test; so does resolving a known divergence (forcing the table to
stay honest). Each known divergence is annotated with which engine matches
canonical (tournament-standard, level-11) Clash Royale.

Notably the harness already surfaced **oracle data errors** — e.g. Goblin Hut is
5 elixir and Lumberjack is 4 elixir / melee / very-fast in crsim (correct), while
the oracle has them wrong.

Regenerate the report:

```bash
python -m verification.report
```

## Behavioural conformance

`behavioral.py` spawns the same 1v1 duel in both engines (direct-spawned at
midfield so towers don't interfere) and compares the winner and time-to-kill.
Because the engines tick at different rates (33 ms vs 50 ms) and acquire targets
in different orders, the comparison is coarse: winners must match and TTK must
agree within `TTK_TOL` (30%). Multi-unit swarms (e.g. Archers) differ in spawn
geometry across engines, so only their winner is checked.

```bash
python -m verification.behavioral --oracle /path/to/clash-simulator
```

## Refreshing the oracle snapshot

The fixture is committed so CI needs no external repo. To bump it to a newer
oracle revision, clone the oracle, install its deps, and regenerate:

```bash
git clone https://github.com/samdickson22/clash-simulator
pip install numba msgspec
python scripts/gen_oracle_fixture.py --oracle ./clash-simulator
python -m verification.report
```

Then re-run `pytest tests/test_cross_engine_stats.py`; if divergences changed,
update `KNOWN_DIVERGENCES` with annotations.
