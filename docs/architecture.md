# ClashRoyale-Zero — System Architecture

> Synthesis + forward design doc. Grounded in the code on the feature branches
> (`feature-v2`, `devin/*`) and the Scroll RE handoff. `main` is currently a stub;
> the implementation referenced here lives on those branches.
>
> RVAs/offsets quoted from the RE work are the **working hypothesis**, build-specific to
> `libg.so` from Clash Royale **v1.3.2**, not verified ground truth.

---

## 1. What this project is

An **AlphaZero/MuZero-style self-play agent for Clash Royale** — learn strong play purely from
self-play (no human data, no hand-crafted policy), then optionally deploy it onto the real game
through a perception layer.

The defining design choice is that the project keeps **three interchangeable simulation
backends behind one environment contract**, trading fidelity against speed/setup:

| Backend | Lang | Fidelity | Throughput | Setup | Role |
|---------|------|----------|------------|-------|------|
| `crsim/` | Python | ~80% (authentic CSV stats) | 0.5–2K ticks/s/core | none | dev, MCTS rollouts, fast iteration |
| `cr_engine/` | Rust + PyO3 | ~80% (same data, faster) | high (native) | `maturin` build | scaled self-play rollouts |
| `scroll_bridge/` | Python → Scroll (Rust) → `libg.so` | **100% (it *is* the game)** | 5–20K ticks/s/core (ARM-on-x86) | redroid + APK + RE | ground truth, validation, final eval |

The whole point of the Scroll backend is *zero sim-to-real gap*. The cost is that you must
drive an undocumented ARM binary and construct the battle objects its own client normally
builds — which is the current **P0 blocker** (see §6 and `docs/p0-engine-fix-spec.md`).

---

## 2. Layered architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ L5  SEARCH + LEARNING            (Python · PyTorch)                          │
│     MCTS family: search / gumbel_search / is_mcts / muzero_search           │
│     Nets: network.py (ResNet-20 SE, ~30M) · transformer_net · opponent_model│
│     Training: self_play(_v2) · trainer(_v2) · replay_buffer(500K) ·         │
│               curriculum · imitation · distributed (Ray/DDP)                │
│     Eval: evaluator (Elo) · tournament · replay                             │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │  obs (44×18×32 + 116) · action (2305) · reward · done
┌───────────────┴──────────────────────────────────────────────────────────────┐
│ L4  ENVIRONMENT CONTRACT  (Gym-like; backend-agnostic)                       │
│     crsim.game.CRGame  ≡  ScrollBattleEnv  ≡  cr_engine (rust_adapter)       │
│     features.py: state→tensor · action_space.py: (card,x,y)+WAIT + masking   │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │  reset / step(commands) / observe / (snapshot,restore)
┌───────────────┼──────────────────────────────────────────────────────────────┐
│ L3  SIM BACKENDS (pick one; same contract)                                   │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────────────────────────┐ │
│  │ crsim/     │  │ cr_engine/   │  │ scroll_bridge/  ──JSON/TCP:9340──►      │ │
│  │ (Python)   │  │ (Rust/PyO3)  │  │ Scroll headless server (Rust)           │ │
│  │ game.py    │  │ engine.rs    │  │   └─ libg.so (real CR) in redroid       │ │
│  └────────────┘  └──────────────┘  │   ◆◆ P0 blocker: battle bootstrap ◆◆    │ │
│                                     └───────────────────────────────────────┘ │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │  reads stats by column name; ignores unknown columns
┌───────────────┴──────────────────────────────────────────────────────────────┐
│ L1/L2  GAME DATA + RULES                                                      │
│   gamedata_v2/*.csv (extracted APK) · gamedata_full/*.json (modern cards,    │
│   evos, champions) · gamedata.py (Python port of data.rs) · level scaling     │
└────────────────────────────────────────────────────────────────────────────┘

         ── optional deployment path ──
   L6 REALTIME: perception.py (CV: detect units/towers/elixir) → controller.py →
                pipeline.py → tap the real client (emulator). Lets a trained
                policy play the actual game.
```

See the per-layer deep dives:
- `docs/layers/L1-game-data.md` — authentic data, level scaling, the card converter, evos/champions
- `docs/layers/L2-simulation-backends.md` — the three backends + the hybrid strategy
- `docs/layers/L3-scroll-bridge-and-re.md` — Scroll protocol, RE findings, offset table
- `docs/layers/L4-environment-state-action.md` — obs/action/reward/masking/perspective
- `docs/layers/L5-search-and-learning.md` — MCTS variants, nets, training, eval

---

## 3. The one contract that holds it together

`ScrollBattleEnv` is written to mirror `crsim.game.CRGame` so "the MCTS and training pipeline
can swap between simulators transparently" (its own docstring). **This is the most important
architectural property of the system** — keep it sacred:

```
reset()                  -> obs
step(action | commands)  -> (obs, reward, done, info)   # advances ticks_per_step
observe()                -> obs
# recommended additions:
snapshot()/restore()     -> StateHandle                  # for MCTS rollouts & crash repro
legal_action_mask()      -> bool[2305]
```

- The learner (L5) must never branch on which backend is underneath. Today crsim is the
  workhorse and Scroll is being brought online; when the P0 fix lands, swapping to Scroll is a
  config change, not a code change.
- `features.py` (state→tensor) and `action_space.py` ((card,x,y)+WAIT, masking) sit at this
  seam and must produce identical encodings regardless of backend, or a policy trained on
  crsim won't transfer to Scroll.

---

## 4. Core dimensions (shared across backends)

| Thing | Value |
|-------|-------|
| Arena grid | 18 × 32 tiles; river rows 15–16; bridges at cols 3–4 / 13–14 |
| Tick rate | 2 Hz (0.5 s/tick); 360+360+360 ticks (regular/OT/sudden death) |
| Elixir | ~0.179/tick normal, ×2 OT, ×3 sudden death, cap 10 |
| Towers | per side: 1 King + 2 Princess |
| Observation | 44 spatial channels (18×32) + 116 scalar features, current-player perspective (board flipped for P1) |
| Action space | 2305 = 4 hand slots × 18 × 32 + 1 WAIT; illegal actions masked to −1e9 |
| Decision cadence | agent acts every `decision_interval_ticks`, engine steps every tick |

---

## 5. End-to-end data flow (self-play step)

```
NN(obs) ─► policy prior + value
        │
        ▼
MCTS (N sims, snapshot/restore rollouts) ─► visit-count policy π
        │
        ▼
sample action a ~ π  ──► env.step(a)  ──► (obs', r, done)
        │                                   │
        └──────────── store (obs, π, z) ───►┘   (z = final game result)
                              │
                              ▼
                    ReplayBuffer (500K ring)
                              │
                              ▼
                Trainer (AdamW, cosine, FP16, DDP/Ray)
                  loss = CE(policy, π) + MSE(value, z) + L2
                              │
                              ▼
                  new checkpoint ─► opponent pool (league) ─► back to self-play
```

---

## 6. Critical path & the P0 blocker (short version)

**P0 — make Scroll run a real battle.** Scroll's hand-written `start_mission` boots a battle
but never constructs the **arena object (`[battle+0xb0]`)**; the per-side towers are wired only
in `begin_battle`'s late block, which is hard-gated on that arena being non-null. Arena null ⇒
towers null ⇒ next tick's `isAlive` dereferences a null tower (`ldr [tower+0x8]`) ⇒ SIGSEGV.
The arena + towers are built only by `load_battle_state` (`0x11a428`), which `start_mission`
skips — **even though the home analog `load_home_state` (`0x11A274`) is already wired.**

Ranked fix (full spec in `docs/p0-engine-fix-spec.md`):
1. **Restore the missing `load_battle_state` call** (diff `start_mission` vs the
   already-working home boot path). Highest leverage, most correct.
2. Drive `load_battle_state` directly with a constructed battle-state stream.
3. Manual build (allocate arena, factory-build 3 towers/side) — last resort, leaves the
   `world`/level object (`[battle+0]`, null vtable) unbuilt and invites further null chasing.

**Win condition (no success claims without this log line):** the benchmark steps **200,000
ticks, no crash, towers present.** Keep that benchmark as a permanent canary.

This is *not* on the critical path for the rest of the system: crsim already satisfies the
contract, so L4/L5 can keep training while P0 is cracked. P0 unlocks 100%-fidelity training
and trustworthy final eval, not the ability to train at all.

---

## 7. Recommended sequencing

```
P0  Scroll runs a clean 200k-tick battle (arena+towers)        ← unblocks 100% fidelity
P1  Wire ScrollBattleEnv fully behind the contract; validate
      crsim vs Scroll on identical action sequences (tune crsim)
P2  Scale self-play on cr_engine (Rust) for throughput; crsim for dev
P3  League self-play + Elo; curriculum (starter→random decks); reward annealing
P4  Search efficiency: Gumbel MuZero (16 sims ≈ AlphaZero 800) + KataGo tricks
P5  Modern-card converter for deck variety (downstream); IS-MCTS for hidden info
P6  (optional) Realtime deployment via perception/controller on emulator
```

---

## 8. Top risks (see layer docs for detail)

1. **Backend divergence** — if crsim and Scroll disagree on physics, a crsim-trained policy
   degrades on Scroll. Mitigation: the L3 validation harness (replay identical actions, diff
   tower HP/outcome/checksum) is a first-class component, not an afterthought.
2. **RE fragility** — every offset is build-pinned to v1.3.2 `libg.so`; a re-extract
   invalidates the table. Pin the binary; keep a guest↔host PC map for fault attribution.
3. **Search vs real-time** — vanilla MCTS at 800 sims is heavy for a 2 Hz game; Gumbel
   MuZero / playout-cap randomization are the realistic path (already scaffolded in `mcts/`).
4. **Imperfect information** — opponent hand/elixir hidden; pure MCTS is unsound. Use IS-MCTS
   + the opponent model, or lean on model-free PPO-style learning for the hidden-info parts.
5. **CPU-only CI** — `train_v2.py` targets CUDA but CI runs CPU; keep a tiny-config smoke
   path green in CI and gate heavy training behind GPU availability.
