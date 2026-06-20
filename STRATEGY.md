# Clash Royale Zero — Optimized Strategy

*Grounded in a code-level audit of the repo (June 2026), not the prior session's claims.*

---

## 1. Brutal current-state assessment (what the code actually shows)

| Area | Claimed | Reality (verified) |
|---|---|---|
| Game simulator | "Exact replica of live CR" | A hand-built **approximation**. No comparison to real games exists. |
| Sim correctness tests | "verified" | `tests/test_sim.py` tests only structure (creation, tower HP, elixir cap, mask, placement, clone). **Zero** card-interaction correctness tests. |
| Card data | "125 cards, authentic stats" | `crsim/cards.py` is **hand-coded** (2,380 lines). `crsim/gamedata_loader.py` (the "authentic data" loader) is **imported nowhere** and there is **no `gamedata.json`**. |
| Two engines | "Rust engine 4,776× faster, integrated" | Two **separate** combat engines with **separate data**: Python `crsim` (hand-coded `cards.py`) and Rust `cr_engine` (CSVs). The training pipeline (`self_play_v2`) uses **Python `CRGame`**; the Rust adapter is referenced only by `parallel_sim.py`, which **no training script uses**. |
| Evolutions / champions / hidden stats | "implemented" | `crsim/evolutions.py`, `heroes.py`, `hidden_stats.py` exist but are **not imported** by `game.py`, `mcts`, or `training`. Disconnected. |
| MCTS search | "Gumbel MuZero" | Each simulation steps the real sim **exactly 1 tick (50 ms)** then bootstraps the NN. Effectively ~1-ply planning — almost no lookahead. |
| State encoding | `(44,32,18)` per the docstring | Actually **254 channels** (`2*125+4`) — one density plane per card type. Built with **pure-Python triple-nested loops** (~1 ms each). |
| Imitation warm-start | "ready" | `KaggleDeckDataset` learns only `V(decks)→win%` (no play). `ExpertReplayDataset` needs a `replay_dir` of state→action pairs that **does not exist** and has no extractor. |
| Training run | "end-to-end complete" | Never actually ran before this week (search-every-tick + 292 GB buffer made it impossible). **Now fixed** — it runs, loss decreases, checkpoints save (PR #4). |

**One-sentence diagnosis:** the project is *wide* (dozens of modules) but has **no verified depth** — the foundation, a single accurate fast simulator with a single source of truth, does not exist, and nearly everything downstream is built on the unvalidated, slower of the two engines.

---

## 2. The core strategic problem

> **You cannot train a world-class agent on a simulator you have not verified, and you cannot scale on a simulator you maintain twice.**

Two root causes, in priority order:

1. **No ground truth.** Nothing measures how wrong the sim is, so any training optimizes the agent to exploit *our bugs*, not to play Clash Royale. (Self-play is especially ruthless at finding sim exploits.)
2. **Two divergent engines + dead data loader.** Mechanics and stats are implemented and stored twice and drift apart. This makes both verification and scaling intractable.

Everything else (encoding bloat, shallow search, missing imitation data, compute) is real but **secondary** — fixing them on top of an unverified, duplicated sim is wasted effort.

---

## 3. Reframe the goal so progress is measurable

"Beat the world #1" is the north star, but it is **not an operational target** — there's no way to measure it here, and it requires data + compute we don't have. Replace it with a **strength ladder**, where each rung has an objective pass/fail:

| Rung | Target | How it's measured |
|---|---|---|
| R0 | Loop runs, loss ↓ | ✅ done (PR #4) |
| R1 | Sim matches reality on canonical interactions | Golden-test suite passes (Section 4) |
| R2 | Beats a scripted heuristic bot | Win rate > 80% vs a fixed rule-based opponent |
| R3 | Monotonic self-play Elo growth | Frozen-pool Elo rises over training, no collapse |
| R4 | Beats strong reference agents | Elo vs a league of past checkpoints + heuristics |
| R5 | Plays the real app competently | Real-device eval (perception + control) on mid-ladder |
| R6 (north star) | Champion-level | Only meaningful with R1–R5 + serious compute |

R1–R4 are achievable on this hardware *if* we fix the foundation. R5–R6 need GPUs and real-device infrastructure you would have to provide.

---

## 4. The optimized plan (dependency-ordered, each phase has a hard gate)

### Phase A — One engine, one source of truth  *(unblocks everything)*
**Decisions:**
- **Make the Rust engine authoritative** for mechanics. It's the only one that can hit the throughput a champion run needs, and maintaining two combat loops is the classic trap.
- **Single data source:** the `cr_engine/gamedata_v2` CSVs (derived from the APK). Generate the Python-side card metadata *from the same CSVs* (the encoder/action-space need card→elixir/type lookups, not a second combat model).
- **Retire / freeze** `crsim/game.py` combat as production. Keep it only as a slow reference oracle *if and only if* it passes the cross-engine conformance test below; otherwise delete it to stop the drift.

**Gate A:** `self_play_v2` runs on the Rust engine via `rust_adapter`, a smoke training run completes on it, and a **cross-engine conformance test** (same seed + script → same key events) passes for the shared card set.

### Phase B — Verification harness + golden tests  *(the missing gate)*
- Encode **canonical interactions** as executable tests from authoritative sources (wiki / Stats Royale / deckshop), e.g.:
  - "Fireball + Zap kills a Musketeer"; "Fireball alone does **not**"
  - "Knight one-shots a Skeleton"; "Mini Pekka kills Musketeer in 2 hits"
  - "Hog Rider reaches the tower in N seconds from bridge"
  - "Golem death spawns 2 Golemites"; "Valkyrie hits 360°"
  - tower HP/damage, elixir regen timing (1 per 2.8 s; 2×/3× phases), deploy time, projectile travel.
- Add 3–5 **real match videos** decoded to rough event timelines for end-to-end sanity (not frame-perfect — just "does the macro flow match").
- Fix card stats/mechanics until the suite is green. **Wire `evolutions.py`/`heroes.py` in or explicitly scope them out** — don't leave them dead.

**Gate B:** ≥ 30 canonical interaction tests pass; documented list of *known* inaccuracies with severity. (We will never be frame-identical to live CR — the goal is "no interaction-changing errors.")

### Phase C — Make self-play fast & sample-efficient  *(leverage)*
- **Encoding redesign:** replace 254 per-card planes with ~20–30 **semantic planes** (friendly/enemy × {ground HP, air HP, DPS, building, range, speed} + towers + placement legality + bridges). Compute it **in Rust** (or vectorized numpy). Effect: ~10× buffer memory, faster encode, far better sample efficiency, smaller net.
- **Search horizon:** roll each simulation forward **one decision interval (N ticks), not 1 tick**, so the search actually sees the consequence of a placement before bootstrapping. Cheap relative to NN cost; large quality gain. (Longer term: train the latent dynamics model for true MuZero rollouts.)
- **GPU batching:** wire the batched-inference path (`parallel_sim`/GPUBatcher) so many self-play games share batched NN evals — the NN forward dominates cost (measured: ~2 ms tiny / 10–50 ms real per eval; a search is ~30+ evals).

**Gate C:** ≥ 10× self-play decisions/sec vs today on the same hardware; buffer of 500k positions fits in RAM.

### Phase D — Warm-start + opponents + eval
- **Value warm-start** from Kaggle/API deck-outcome data (this is all that data can give — no play).
- **Policy** comes from self-play, not imitation (we lack frame action traces). Only invest in frame imitation if a KataCR-style dataset is actually obtained.
- **League / PFSP** against frozen checkpoints + the heuristic bot; track Elo.

**Gate D:** R2 (>80% vs heuristic) and R3 (rising frozen-pool Elo) achieved.

### Phase E — Scale (needs your hardware)
- Multi-GPU self-play + training, millions of games, the full league.
- Optional real-device pipeline (YOLOv8 perception + ADB) for R5.

**Gate E:** sustained Elo growth at scale; real-device competence.

---

## 5. Key architectural decisions (the "optimize" part)

1. **Authoritative engine = Rust; data = one CSV set.** Biggest single decision. Kills the dual-maintenance trap and is the only path to champion-scale throughput.
2. **Verification is a *gate*, not a phase you can skip.** No training claims without a green golden-test suite. This is exactly what the prior sessions skipped.
3. **Semantic encoding over per-card planes.** Highest-leverage efficiency change; also generalizes better to unseen decks.
4. **Search must look ahead ≥ 1 decision interval.** 1-tick MCTS is barely planning.
5. **Be honest about imitation.** Deck-outcome data ≠ play data. Don't burn weeks trying to imitate play from data that has no actions.
6. **Measure with a ladder + Elo.** "World #1" is unmeasurable here; monotonic Elo against a frozen pool is.

---

## 6. Compute reality

This VM has **no GPU**. AlphaZero/MuZero-class results are **thousands of GPU-hours** (self-play dominated). Concretely:
- R1–R3 (sim verification, encoding, beat-heuristic, early Elo) are feasible **on CPU here**, slowly.
- R4+ needs GPUs. You must provide cloud GPU budget or hardware. Without it, the realistic ceiling is "strong in-sim vs heuristics," not "champion."

State this plainly so expectations match reality.

---

## 7. Risk register / kill criteria

- **Sim can't be made accurate enough** (live CR has undocumented mechanics): cap the goal at "internally consistent competitive bot," not "real-CR champion." Decide after Phase B.
- **Rust engine missing too many cards/mechanics:** measured in Phase A; may require porting logic from the Python sim into Rust.
- **No frame-level human data ever obtained:** accept self-play-only policy; fine for R1–R4.
- **No GPU budget:** stop at R3; ship a strong in-sim agent + the verified engine as the deliverable.

---

## 8. Concrete next two weeks (if you say go)

1. **Phase A**: wire `self_play_v2` onto `rust_adapter`, add the cross-engine conformance test, pick `gamedata_v2` as the single data source, auto-generate Python card metadata from it. *(Gate A)*
2. **Phase B start**: stand up `tests/test_interactions.py` with the first ~15 canonical interactions; fix the sim until green. *(toward Gate B)*

These two are the foundation the entire goal rests on, and they are exactly what kept getting skipped.
