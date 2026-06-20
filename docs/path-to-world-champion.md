# Path to a World-Champion-Level Agent — A Staged Program

Goal: build the strongest possible Clash Royale agent and scale it toward beating the world
champion. This doc is the **program**: method commitments, a phased plan with explicit go/no-go
gates, how we *measure* progress without access to the champion, compute staging, and the
honest-stop criteria that tell us early whether "world champion" is reachable or whether the
realistic target is "beats strong humans."

It is built directly on the four binding constraints established in `data-embeddings-rl.md`
and the feasibility analysis: **(1) fidelity, (2) mixed-strategy convergence, (3) the
deck/format meta, (4) compute** — in roughly that order of importance. Read those first.

---

## 0. What "best player" actually requires (the constraints, restated as requirements)

| Constraint | Why it gates "world champion" | Requirement it imposes |
|---|---|---|
| **Fidelity** | Pro skill lives in the *micro* (tile-fraction placement, frame-exact timing, aggro pulls). An ~80% sim deletes exactly that layer. | Final training + eval on the **real engine (Scroll)**. Gates everything. |
| **Mixed-strategy / simultaneous moves** | Optimal CR play is a *mixed* (randomized) strategy; naive self-play **cycles** and converges to nothing. | **League / fictitious-play / exploiter** training, not vanilla self-play. Target low *exploitability*, not just high internal Elo. |
| **Deck / format meta** | Competitive CR is multi-deck with draft/ban; a single-deck agent doesn't engage it. | An **outer policy** over deck construction + bans, with the in-game agent as evaluator. |
| **Compute** | The precedents in this game class needed scale far beyond a single-box budget. | **Staged** compute with go/no-go gates; never commit the full spend upfront. |

The honest framing: **more RL training saturates well before these do.** Past a point you are
equilibrium-limited, fidelity-limited, and format-limited. This program attacks those limits
directly instead of just scaling self-play.

---

## 1. The measurement problem (solve this first or you're flying blind)

You cannot grind thousands of games against the world champion. So "are we approaching
champion level?" must be answered by **proxies**, and the program is only as honest as its
measurement. Four instruments, in increasing trustworthiness:

1. **Internal Elo vs. the population.** Cheap, but measures *self-consistent* improvement, not
   absolute human level. Necessary, never sufficient — it can rise while the agent is just
   beating its own blind spots.
2. **Anchored Elo.** Calibrate the internal ladder against fixed external reference points:
   the built-in AI, scripted baselines, and **pro-replay agreement** (does the agent's policy
   match / outperform the moves top humans made in recorded games?). Gives an absolute-ish
   scale.
3. **Exploitability (the key metric).** Freeze the agent; train a fresh **best-response
   exploiter** against it with full RL. How badly can a dedicated adversary beat it? **Low
   exploitability is the real proxy for "robust against a creative human."** This is what
   directly addresses the mixed-strategy concern — internal Elo can't see a hole that no
   population member happens to probe, but a trained exploiter will find it.
4. **Human ladder.** A graded chain we *can* actually run: built-in bot → strong ladder humans
   → top ladder / semi-pro → pros (if recruitable) → champion. Each rung is a gate.

> Build instruments 1–3 in Phase 0/1. If exploitability stays high no matter how much we
> train, that is the early signal that the equilibrium-convergence problem is binding and
> "world champion" needs re-scoping — *before* spending the big compute.

---

## 2. Method commitments (decisive, grounded)

- **Algorithm backbone: large-scale model-free RL** (IMPALA/PPO-family, AlphaStar-like actor-
  learner) — the right tool for real-time, simultaneous-move, imperfect-info play. **Not**
  800-sim AlphaZero MCTS at inference (wrong template; see the feasibility analysis).
- **Search: optional, training-time only.** Gumbel-MuZero-style few-sim search can sharpen the
  policy target during training; inference must be search-light to meet the 2 Hz budget.
- **Convergence: PFSP-style league + explicit exploiter agents** (main agents, main-exploiters,
  league-exploiters, frozen past selves). This is what drives toward an *approximate Nash*
  instead of cycling — it's a requirement, not a tuning option.
- **Representation: spatial + entity-transformer fusion with stat-based card embeddings**
  (`data-embeddings-rl.md`). The stat-based embeddings are what make deck *generalization* and
  new-card play possible.
- **Warm-start: imitation / offline RL from replays**, then online self-play to exceed the data
  ceiling (the imitation→RL transition). Bootstraps embeddings + a sane policy cheaply.
- **Hidden info: exploit deducibility.** Elixir (deterministic regen − observed spends) and
  deck cycle (fixed order) are *computable* — feed exact trackers as features so the residual
  uncertainty is opponent *intent*, handled by the opponent model. Reserve IS-MCTS for the
  genuinely ambiguous cases.

---

## 3. The phased program (each phase: objective · method · **gate** · honest-stop)

### Phase 0 — Fidelity foundation & the gap measurement *(do first)*
- **Objective:** real-engine battles that run, and a *quantified* crsim↔Scroll micro-gap.
- **Method:** land the P0 battle-bootstrap (`p0-engine-fix-spec.md`); build the calibration
  harness (replay identical action sequences on crsim vs Scroll, diff tower-HP / outcome /
  checksum); assemble a **micro-scenario battery** (precise placements, aggro pulls, projectile
  timing, kiting) and measure where the sims diverge.
- **Gate:** 200k-tick Scroll benchmark passes **and** the micro-gap is quantified per scenario.
- **Honest-stop:** if the real engine cannot be driven to run battles at all (P0 unsolvable
  with available access), the whole real-fidelity path is blocked — escalate before investing
  in the RL stack for the champion goal.

### Phase 1 — Single-deck mastery (prove the core on the tractable sub-problem)
- **Objective:** superhuman-vs-strong-humans on **one fixed meta deck**, mirror + matched.
- **Method:** full encoder/embedding/action stack; imitation warm-start; **league with
  exploiters**; train on crsim/cr_engine for volume.
- **Gate:** (a) internal Elo plateaus, (b) **exploiter win-rate against the frozen agent is
  low** (the real test), (c) beats strong ladder humans in fixed-deck sets.
- **Honest-stop:** if exploitability stays high after league training matures, the
  mixed-strategy convergence problem is binding — fix methodology (more diverse exploiters,
  regret-based training) before scaling. Do **not** paper over it with more self-play.

### Phase 2 — Fidelity transfer (move the final stretch onto the real engine)
- **Objective:** the Phase-1 agent, hardened on Scroll, retains its edge under exact physics.
- **Method:** fine-tune / late-train on Scroll; re-run the micro-scenario battery and the human
  ladder *on the real engine*.
- **Gate:** Scroll-trained agent beats the crsim-only agent head-to-head on Scroll **and**
  holds up vs humans on the real engine (no micro-skill collapse).
- **Honest-stop:** if the micro-gap (Phase 0) is too large to close and the policy collapses on
  transfer, world-champion is off the table on this sim stack — re-scope to "beats strong
  humans on crsim," and report that honestly.

### Phase 3 — Deck generalization
- **Objective:** strong across the meta deck set, not one deck.
- **Method:** train across a deck distribution using the stat-based embeddings; randomize
  matchups; curriculum from fixed → random decks.
- **Gate:** high win-rate across the top meta decks and unseen reasonable decks; embeddings
  generalize to a held-out card.

### Phase 4 — The meta / format layer (the part people forget)
- **Objective:** competitive in the *actual* match format (multi-deck, draft/ban).
- **Method:** an **outer policy/search over deck construction and bans**, using the Phase-3
  in-game agent as the match evaluator (a hierarchical game: deck-selection meta-game on top of
  the in-battle game).
- **Gate:** beats strong humans under tournament format, not just single fixed-deck games.

### Phase 5 — Champion-level certification & hardening
- **Objective:** evidence at/near champion level, with quantified residual exploitability.
- **Method:** recruit top human players for structured match sets; run a **continuous exploiter
  program** (always training fresh adversaries; fold their discoveries back into the league);
  certify with enough games for statistical significance.
- **Gate:** target human win-rate against top pros with **low, stable exploitability**.
- **Honest-stop:** if human win-rate plateaus below target or exploiters keep finding cracks,
  declare the achieved level honestly (e.g., "top-0.x%, beats most pros, with asterisks") —
  the realistic shape of every result in this game class so far.

---

## 4. Compute staging (honest, gated, no invented numbers)

I won't pretend to a precise FLOP budget — the precedents in this game class used scale far
beyond a single box, and exact figures aren't something I'll fabricate. The disciplined way to
spend is **staged with go/no-go gates**, not a big upfront commitment:

- **Phase 0–1 (proof of core):** the existing ~single-node budget (e.g. 8×A100) is appropriate
  to prove fidelity + single-deck mastery + the exploitability methodology. Cheapest, highest
  information-per-dollar — and it's where the program most likely *fails fast* if it's going to.
- **Phase 2–3 (fidelity + generalization):** scales up — Scroll throughput (many redroid
  workers) + larger league. Mid commitment.
- **Phase 4–5 (meta + certification):** the expensive tail (large league, continuous exploiter
  program, deck meta-search). Only fund this **after** Phases 1–2 gates pass.

Compute is a top risk, so it is *staged behind evidence*: each phase's gate is the
authorization to spend on the next.

---

## 5. Top risks & kill criteria (decide these now, not later)

| Risk | Early signal | Decision |
|---|---|---|
| Real engine can't be driven (P0) | Phase 0 fails | Escalate; champion path blocked until resolved |
| Micro-fidelity gap too large to close | Phase 0 battery + Phase 2 transfer | Re-scope to "beats strong humans"; report honestly |
| Self-play won't converge (cycling) | High exploiter win-rate persists in Phase 1 | Fix methodology (regret/Nash, exploiter diversity) before scaling |
| Brittleness to creative humans | Exploiters keep finding cracks in Phase 5 | Continuous exploiter hardening; accept asterisked result |
| Compute ceiling | Gates pass but next phase unaffordable | Stop at the highest funded gate; report achieved level |
| Deck meta unsolved | Phase 4 underperforms | Constrain to fixed-deck claim (an explicit asterisk) |

The ethos: **explicit gates and an honest stop.** We state up front what level the evidence
supports, and we don't claim "world champion" unless Phase 5's gate is actually met.

---

## 6. The one thing to do first

**Land P0 and measure the fidelity gap (Phase 0).** It is the cheapest phase, it gates
everything above it, and it is the single fact that most determines whether "beat the world
champion" is reachable at all on this stack. Everything else is sequenced behind it.
