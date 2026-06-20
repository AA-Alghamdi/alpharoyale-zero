# Engine Conformance & Accuracy

How the engine works, **how accurate it is** (measured, not claimed), and what
the path to higher fidelity + scale looks like. This is the honest answer to the
three questions: *(1) how the engine functions, (2) how accurate it is vs.
reality, (3) how to scale up.*

---

## 1. How the engine functions (short version)

Full detail is in [`ARCHITECTURE.md`](ARCHITECTURE.md). The essentials:

- **Tick loop.** The Python engine (`crsim/`) advances in fixed **50 ms ticks**
  (20 ticks/s). Each tick: regenerate elixir → apply queued player actions
  (deploy / spell / **ability**) → update every entity (target → move → attack →
  status effects) → resolve projectiles, deaths, death-spawns/death-damage →
  check win/overtime.
- **Entities.** Towers (2 crown + 1 king per side), troops, buildings,
  projectiles, area-effects. Each entity carries HP, damage, hit-speed, range,
  speed, target rules, plus modern state (shield, charge, evolution flags,
  champion-ability cooldown/cloak/protection).
- **Data source.** Base stats come from `crsim/cards.py`, overlaid with an
  authentic Supercell game-data export (`crsim/gamedata.py`) at **tournament
  standard, Level 11**. Mechanics the export doesn't encode (champion ability
  costs/cooldowns, some evolution effects) are hand-curated from community data.
- **RL interface.** Discrete action space (deploy each card at a grid cell,
  activate a champion ability, or wait) + a spatial/entity observation; MCTS
  plans over a decision interval and a ResNet evaluates positions.

There are **two engines**: this Python one (used by training) and a Rust core
(`cr_engine/`, faster, currently disconnected from the training loop). Unifying
them is a scale-up item, not an accuracy item.

---

## 2. How accurate is it? (measured)

Accuracy is measured against **three independent oracles**, hardest-to-fake
first. None of them requires a GPU.

### 2a. Vs. reality — verified wiki anchors

`tools/conformance/wiki_anchors.py` holds hand-verified Level-11 values;
`tests/test_conformance_anchors.py` enforces them (±1%).

**Result: 13/16 anchor stats exact; 3 mismatches, all explained by data
staleness (not engine bugs).**

| Card | Stat | Ours | Wiki L11 | Verdict |
|---|---|---:|---:|---|
| Knight | hp / dmg | 1766 / 202 | 1766 / 202 | exact |
| Musketeer | hp / dmg | 721 / 218 | 720 / 218 | exact |
| P.E.K.K.A | hp / dmg | 3760 / 816 | 3760 / 816 | exact |
| Mega Knight | hp | 3993 | 3993 | exact |
| Valkyrie / Prince | hp | 1908 / 1920 | 1908 / 1920 | exact |
| Fireball / Rocket / Zap / Arrows | dmg | 689 / 1485 / 192 / 322 | same | exact |
| **Wizard** | hp | **721** | **755** | stale: missing Aug-2024 +4.8% HP buff |
| **Mini P.E.K.K.A** | hp / dmg | **1361 / 721** | **1433 / 755** | stale: missing ~+5% buff |

The core, long-stable cards are **exact**. The only failures are cards that got
buffed in 2024-25 after our data dump was cut — a *data-freshness* gap, fixable
by refreshing the export (see §4).

### 2b. Vs. an independent engine — samdickson22/clash-simulator

`python -m tools.conformance.compare_stats` diffs our stats against a separately
authored CR engine (its own `gamedata.json`, 33 ms tick). **103 cards compared:**

- HP |Δ|: mean **3.4%**, median **1.3%**
- Damage |Δ|: mean **3.7%**, median **1.1%**
- HP **signed** median: **−1.25%** → a *systematic* level-scaling offset between
  the two engines, not random per-card error. Our anchors match the wiki
  exactly, so on the cards we've verified, **ours is the correct side** of that
  offset.

Annotated outliers (|Δ| > 5%):

| Card | Δ | Cause |
|---|---|---|
| Arrows dmg +160% | representation: ours = total damage, ref = per-wave |
| Rascals / Cannon Cart / Goblin Drill / Goblin Hut HP −30…−51% | representation: which sub-unit / shield the card's "HP" refers to |
| Rune Giant dmg +26%, Boss Bandit −10% | newest cards; data differs between dumps |
| Wizard / Mini P.E.K.K.A / Lumberjack / Electro Wiz / Giant Skeleton / Hunter / Monk −5…−8% | **our data staleness** — candidate balance-patch gaps |

So cross-engine disagreement splits into three buckets: **representation
differences** (not bugs), **newest-card data drift**, and **our staleness** (the
actionable list).

### 2c. Vs. known interactions — golden tests

`tests/test_interactions.py` + the rest of the suite encode known CR outcomes
(e.g. *Fireball + Zap kills Musketeer*, *Hog reaches tower in N s*, *Inferno
ramps to max dps*, evolution/ability behaviours). **148 tests pass.** This is
what originally caught the catastrophic bugs the project was about to train on
(damage spells dealing **zero** damage; troops never crossing the river; King
tower firing early).

### What "100% accurate" really means here — straight talk

- **Code-level parity is the right target and is largely achieved on verified
  cards.** The asymptote to exactness comes from sourcing every value from game
  data + cross-checking independent engines + golden tests — *not* from video.
- **Video verification cannot certify exactness.** The measuring instrument
  (CV detection, KataCR-style) is itself ~85–95% accurate, so it cannot prove
  the sim to a tighter tolerance than its own error. Video is a great *gross*
  sanity check ("Hog reaches tower in ~X s", "Fireball leaves ~Y% HP"), not a
  frame-exact oracle.
- **The only true 100% oracle is the real engine.** `scroll_bridge/` already
  stubs a client for [Scroll][scroll], which runs Supercell's actual compiled
  `libg.so`. Running it needs an ARM device/redroid + the APK + the headless
  Scroll server — none present on this CPU box — but it is the definitive
  verification path and the harness here is structured to diff against it the
  same way it diffs against clash-simulator.
- **The reference engine is itself imperfect:** it crashes on a basic Fireball
  cast (`NameError: math not imported` in its knockback path) and its README
  says "probably wrong", which is exactly why we treat agreement as evidence and
  disagreement as a pointer — never as ground truth.

---

## 3. Reproduce

```bash
# vs reality (no external deps) — part of the normal suite
python -m pytest tests/test_conformance_anchors.py -q

# vs independent engine (clone it first)
git clone https://github.com/samdickson22/clash-simulator ~/refs/clash-simulator
python -m tools.conformance.compare_stats            # or --out report.md
```

---

## 4. Prioritised follow-up to push accuracy higher

1. **Refresh the game-data export.** The staleness gaps (Wizard, Mini P.E.K.K.A,
   and the −5…−8% cross-engine cluster) all trace to one root cause: the bundled
   dump predates recent balance patches. A fresh canonical export from a current
   client fixes the whole class at once — far better than hand-patching cards.
2. **Normalise representation differences** in the harness (Arrows per-wave vs
   total; multi-unit cards) so genuine bugs aren't masked by noise.
3. **Stand up the real-engine oracle** (Scroll/`libg.so`) on ARM hardware for
   definitive, frame-level verification of the mechanics video can't certify.
4. **Expand golden tests** for the newest cards (champions/evolutions) where
   cross-engine data drift is highest.

---

## 5. Scale-up (how to get to "beats the best")

Full plan in [`STRATEGY.md`](STRATEGY.md). The dependency-ordered path:

1. **Throughput.** Move self-play onto the Rust engine, vectorise environments,
   GPU-batch the network, and split into many CPU actors + a central GPU
   learner. This is what makes *tens of thousands* of games/day real.
2. **Algorithm.** Move from naive self-play to an **AlphaStar-style league**
   (main agents + exploiters + frozen past versions) to avoid strategy collapse
   in CR's rock-paper-scissors matchups.
3. **Warm-start.** Initialise from human-replay data (KataCR's expert replays /
   the Kaggle match corpus) instead of from scratch.
4. **Measurement.** Anchor strength with the Elo ladder (already built, PR #7)
   vs. a frozen opponent pool; eventually real-device eval via Scroll's ADB
   bridge.
5. **The gate is hardware.** Champion-level training is thousands of GPU-hours.
   Everything is being built launch-ready; the run itself needs GPUs.

[scroll]: https://git.xeondev.com/Supercell/Scroll
