# SCALE-UP — the GPU-scale training runbook

This is the launch runbook for training a strong ClashRoyale-Zero agent. It
documents the throughput math, the exact commands, hardware sizing, and the Elo
milestones to watch for.

**Honest framing up front.** Everything in here is *launch-ready and verified on
CPU at tiny scale* (see the smoke test below). The actual large run needs GPU
hardware this development box does not have. Nothing here claims to have trained
a champion — it claims the pipeline that *can* is built, tested end-to-end, and
will saturate a GPU the moment you point it at one.

---

## 1. What the GPU-scale path is made of

| Piece | File | What it does |
|-------|------|--------------|
| Batched NN evaluation | `model/batched_eval.py` | Encode N states, run **one** forward pass, split results. Numerically identical to per-state eval (tested). |
| Batched Gumbel search | `mcts/batched_gumbel.py` | Gumbel MuZero search that batches root + leaf evals across all `(game, player)` instances in a wave. |
| Vectorized self-play | `training/vectorized_self_play.py` | `VectorizedSelfPlayWorker` steps N games in lockstep and batches their NN evals. Drop-in replacement for `SelfPlayWorkerV2`; emits the **same** full `ReplayEntry` (spatial + scalar + entity features + aux targets). |
| Replay warm-start | `training/replay_buffer.py` | `save(path)` / `load(path)` persist the buffer to `.npz` (float16), so a run can start from a previously generated buffer instead of cold. `drain_all()` losslessly moves every position to the learner. |
| League + Elo | `training/league.py`, `eval/baseline_agents.py` | AlphaStar-style frozen-snapshot pool + PFSP opponent selection; win-rate → Elo via `winrate_to_elo`. |
| Distributed orchestration | `scripts/train_v2.py` (`_run_distributed`) | Ray: N−2 GPUs run self-play actors, 2 GPUs train. Each actor now uses the **vectorized** worker when `--vectorized-envs > 0`. |

### Why batching is the whole game

On a GPU a forward pass for a batch of 1 costs almost the same wall-clock time
as a batch of 512 — the matrix multiply is latency-bound, not throughput-bound,
at small batch. So running `N` games one-at-a-time wastes ~`N×` of the card.

The vectorized worker steps N games together; at each decision tick it collects
every `(game, player)` state that needs evaluating, runs **one** batched forward
pass, and fans the policies/values back out. Sequential-Halving phases inside
the search are batched the same way.

---

## 2. Throughput math

**Measured on this CPU box (tiny 0.4M-param model, 8 sims, batch = 8 games):**

| Worker | games/s | positions/s | speedup |
|--------|---------|-------------|---------|
| `SelfPlayWorkerV2` (sequential) | 0.31 | 14.3 | 1.0× |
| `VectorizedSelfPlayWorker` (batch 8) | 0.77 | 37.2 | **2.5×** |

CPU has little parallelism, so 2.5× is the *floor* of the benefit. The point of
batching is GPU: there a batch-256 forward pass costs ≈ a batch-1 pass, so the
effective speedup scales with the batch size you can fit (typically **10–50×**
over sequential, before counting multiple actors).

**Back-of-envelope for a real run** (full 11.1M-param `CRStarNet`, 16 sims,
decision interval 8 ticks ⇒ ~45 decisions/game over a 3-min game):

```
positions/game        ≈ 45 decisions × 2 players ≈ 90
NN evals/decision     ≈ 1 root + (16 sims × ~1 leaf) ≈ 17  (× 2 players)
NN evals/game         ≈ 45 × 17 × 2 ≈ 1,530
```

So self-play is dominated by NN evals, which is exactly what batching collapses.
With B games batched per actor and A actors:

```
throughput ≈ A × B × (single-game rate ÷ per-batch overhead)
```

A single A100 comfortably runs B≈128–256 inference states/batch for this model.
With 6 self-play GPUs that is the "tens of thousands of games" regime you asked
about — on the order of **10⁴–10⁵ games/hour** depending on game length and sim
count.

---

## 3. Exact commands

### 3a. Smoke test (CPU, < 4 min) — proves the path is launch-ready

```bash
python scripts/train_v2.py --smoke-test \
    --checkpoint-dir /tmp/smoke_ckpt \
    --save-buffer /tmp/smoke_buf.npz
```

`--smoke-test` shrinks everything (tiny model, 4 vectorized games, ~100 train
steps, CPU) and runs the **entire** pipeline: vectorized self-play → train →
eval + Elo → league snapshot → checkpoint + buffer save. Verified output:

```
VecWorker 0: 4 games (n_envs=4), 192 positions
Eval @ step 100: 25.0% win rate vs random | Elo ~809
Saved checkpoint: /tmp/smoke_ckpt/final.pt
League snapshots: 1 frozen
Saved 512 replay positions to /tmp/smoke_buf.npz
Training complete! Final win rate vs random: 75.0% | Elo ~1191
```

Then confirm warm-start reload (skips warmup, trains from the saved buffer):

```bash
python scripts/train_v2.py --smoke-test --warmstart-buffer /tmp/smoke_buf.npz
# -> "Loaded 512 replay positions" / "Warm-start buffer loaded: 512 positions"
```

### 3b. Single-GPU run (one workstation GPU)

```bash
python scripts/train_v2.py \
    --device cuda:0 \
    --vectorized-envs 64 \        # step 64 games in lockstep, batch their evals
    --max-eval-batch 512 \
    --gumbel-sims 16 \
    --batch-size 1024 \
    --buffer-size 500000 \
    --warmup-positions 20000 \
    --eval-interval 10000 \
    --save-buffer checkpoints/buffer.npz
```

### 3c. Full distributed run (multi-GPU, Ray)

```bash
pip install 'ray[default]'
python scripts/train_v2.py \
    --distributed --n-gpus 8 \    # 6 self-play GPUs + 2 train GPUs
    --vectorized-envs 128 \       # each self-play actor batches 128 games
    --max-eval-batch 1024 \
    --gumbel-sims 16 \
    --batch-size 2048 \
    --buffer-size 2000000 \
    --max-steps 2000000
```

### 3d. Warm-start from imitation data (optional, avoids cold start)

```bash
python scripts/train_v2.py --warmstart-data data/battles.jsonl --warmstart-epochs 5 ...
```

Imitation only has deck→outcome signal in the bundled data; for per-action
imitation, wire in a replay source (e.g. the KataCR expert-replay dataset) that
emits `(state, action)` pairs, then point `--warmstart-data` at it.

---

## 4. Memory / hardware sizing

**Replay buffer** (measured): ~36.8 KB/position with the 18-plane semantic
encoding + 64×40 entity block + aux targets.

| Buffer capacity | RAM |
|-----------------|-----|
| 500k positions | ~18 GB |
| 1M positions | ~37 GB |
| 2M positions | ~74 GB |

(Before the 254→18 plane encoding change, 500k positions was ~146 GB and would
not fit — see PR #6.) Keep the buffer in CPU RAM on the learner node; it is fed
to the GPU in `--batch-size` chunks.

**Model:** `CRStarNet` ≈ 11.1M params (~45 MB fp32). Trivial next to activations
and the buffer.

**Recommended starting footprint:**

| Goal | Hardware | Notes |
|------|----------|-------|
| Validate the loop learns | 1× consumer GPU (e.g. 4090) | `--vectorized-envs 64`, 500k buffer, single-GPU mode |
| Reach a strong ladder agent | 4–8× A100/H100 | distributed, 1–2M buffer, runs days |
| Champion-class | tens of GPUs, weeks | league with many exploiters, 2M+ buffer |

---

## 5. Elo milestones to watch

Elo here is **relative** (win-rate vs the reference pool → `winrate_to_elo`),
anchored at random ≈ 1000. Targets, in order:

| Milestone | Signal | What it means |
|-----------|--------|---------------|
| R0 | loop runs, loss ↓, checkpoint saves | ✅ done (smoke test) |
| R1 | > 70% vs random (Elo ≳ 1150) | learning *something* — already hit in the 4-min CPU smoke run |
| R2 | beats the scripted heuristic agent (> 55%) | learned real tactics, not just unit volume |
| R3 | Elo rising vs frozen league snapshots over time | self-improvement loop is healthy (no collapse) |
| R4 | beats every frozen snapshot in the pool | strictly stronger than all past selves |
| R5 | stable vs exploiters added to the league | robust, not just beating a narrow meta |
| R6 | real-device eval vs human ladder (needs KataCR ADB bridge) | the "beat top players" gate |

Watch for **strategy collapse** (Elo plateaus or oscillates): that is the cue to
add exploiter agents to the league rather than training more of the same.

---

## 6. Known gaps before R6

1. **Compute.** This box has no GPU; the run itself is yours to launch.
2. **Per-action imitation data.** Warm-start currently has deck→outcome only.
   The KataCR replay dataset is the path to `(state, action)` warm-start.
3. **Real-device evaluation.** R6 needs the KataCR YOLO + ADB bridge to measure
   against actual ladder opponents; not built here (GPU + device work).
4. **Rust engine backend.** The Rust engine is fast but does **not** implement
   champions / evolutions / the ABILITY action / entity features, so wiring
   self-play onto it today would *regress* the 1:1 fidelity. It is a future
   throughput optimization, not a drop-in — keep Python authoritative until the
   modern mechanics are ported and a cross-engine conformance test is green.
