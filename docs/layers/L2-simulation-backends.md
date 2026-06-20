# L2 — Simulation Backends

Three backends implement the same environment contract (§3 of `architecture.md`). You pick one
per use case; the RL stack above never knows which is running.

```
            ┌─────────────────────────── same contract ───────────────────────────┐
            │                                                                       │
   crsim.game.CRGame            cr_engine (rust_adapter.py)         ScrollBattleEnv
   pure Python                  Rust + PyO3                         Python → Scroll → libg.so
   ~80% fidelity                ~80% fidelity                       100% fidelity
   0.5–2K ticks/s/core          native, high throughput             5–20K ticks/s/core (ARM/x86)
   zero setup                   maturin build                       redroid + APK + RE
```

## A. `crsim/` — Python reference simulator

The workhorse for development and MCTS rollouts. Modules:
`constants.py` (geometry/timing/action space), `cards.py` (roster), `entities.py` (runtime
entity), `pathfinding.py` (flow fields), `gamedata.py`/`gamedata_loader.py` (authentic stats),
`game.py` (the tick loop), plus `heroes.py`/`evolutions.py`/`hidden_stats.py` for modern cards.

Per-tick loop (`game.py`):
1. regen elixir (phase-dependent, cap 10)
2. process queued card placements from both players (simultaneous-move)
3. spawner buildings produce units
4. each entity: acquire target → move (or attack if in range, respecting min range)
5. remove dead, check win (crowns / tower destroyed / time)

**Pathfinding** is a precomputed **flow field per (side, target-mode)**, recomputed only when
buildings appear/disappear — avoids per-entity A*. Three modes: ground-buildings-only
(Giant/Hog → nearest enemy building via a bridge), ground-any-target, air (straight line).

**Vectorization** (`VectorizedCRSim`): run N games in batched NumPy arrays
(`[n_envs, max_entities, …]`) for training throughput.

Limitations: hand-written physics ≈ 80% fidelity; less complete than the best external Python
sim (`samdickson22/clash-simulator`, 732K ticks/s, ~full mechanics) — which is a candidate
backend swap if crsim fidelity becomes the bottleneck.

## B. `cr_engine/` — Rust reimplementation (PyO3)

Same rules as crsim, rewritten in Rust for speed and exposed to Python via `src/python.rs` and
`crsim/rust_adapter.py`. Modules: `data.rs` (CSV parse + scaling — the canonical data path),
`arena.rs`, `entity.rs`, `movement.rs`, `combat.rs`, `spells.rs`, `engine.rs`. Built with
maturin; `bench.rs` benchmarks tick throughput.

Use it as the scaled self-play rollout engine once correctness matches crsim. Keep `data.rs`
as the single source of truth for stats and treat `crsim/gamedata.py` as a verified port of it.

## C. `scroll_bridge/` — the real engine

100%-faithful: it drives Supercell's actual compiled engine. Detailed in
`docs/layers/L3-scroll-bridge-and-re.md`. This is ground truth for validation and final eval;
not yet usable for training because of the P0 battle-bootstrap blocker.

## The hybrid strategy (recommended operating mode)

Don't treat the backends as competitors — compose them:

1. **Train** on crsim/cr_engine (fast, no setup) for the bulk of self-play.
2. **Calibrate** crsim/cr_engine against Scroll: replay identical action sequences, diff tower
   HP / outcome / checksum, and tune the Python/Rust physics toward the real engine. This
   *shrinks the sim-to-real gap* — the validation harness is a permanent component.
3. **Final-evaluate** trained agents on Scroll (and optionally the realtime path on a live
   client) so reported strength is against the real game, not our approximation.

This way the project gets crsim's iteration speed and Scroll's fidelity without blocking
training on the RE work.

## Backend-divergence risk
If crsim and Scroll disagree materially, a crsim-trained policy can exploit sim artifacts and
degrade on the real engine. Two mitigations: (1) the calibration harness above; (2) keep
`features.py`/`action_space.py` encodings identical across backends so at least the *interface*
can't drift, only the physics — which the harness then measures.
