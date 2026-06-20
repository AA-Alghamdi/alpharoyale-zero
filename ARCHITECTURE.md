# ClashRoyale-Zero: Architecture Document

## 1. System Overview

ClashRoyale-Zero is an AlphaZero-style reinforcement learning system for Clash Royale.
It combines a high-fidelity game simulator, a deep residual neural network, and Monte
Carlo Tree Search (MCTS) to learn superhuman play entirely through self-play — no human
data, no hand-crafted heuristics.

```
+-------------------------------------------------------------+
|                    SELF-PLAY LOOP                            |
|                                                              |
|  +------------+     +---------+     +------------------+     |
|  | Simulator  |<--->| Gumbel  |<--->| CRStarNet        |     |
|  | (CRSim)    |     | MuZero  |     | (entity-xformer  |     |
|  |            |     | search  |     |  + spatial ResNet)|    |
|  +------------+     +---------+     +------------------+     |
|        |                                    ^                |
|        v                                    |                |
|  +------------------+    +------------------+                |
|  | Replay Buffer    |--->| Trainer          |                |
|  | (~500K positions,|    | (AdamW, AMP;     |                |
|  |  float16 ring)   |    |  DDP at scale)   |                |
|  +------------------+    +------------------+                |
+-------------------------------------------------------------+
```

> **Doc vs. shipped defaults.** This document describes the GPU-scale
> *production target* (e.g. 800-sim MCTS, a 256-filter / 20-block ResNet,
> 2×A100 DDP). The shipped code defaults are smaller and faster — Gumbel-MuZero
> search (16 sims, sequential halving) over `CRStarNet` (128 filters, 10
> blocks) — and scale up via config when GPU hardware is available. Concrete
> interface numbers below (tick rate, action-space size, channel counts) match
> the code.

### Why AlphaZero for Clash Royale?

| Property           | Go / Chess        | Clash Royale               |
|--------------------|-------------------|----------------------------|
| Turn structure     | Alternating       | Simultaneous / real-time   |
| Action space       | ~361 (Go)         | 2306 (card × position)     |
| State observability| Perfect           | Partial (opponent hand)    |
| Time dimension     | None              | Continuous (elixir, timer) |
| Branching factor   | ~250 (Go)         | up to 2306                 |

To handle real-time + simultaneous play, we **discretize time into 50 ms ticks** (20 Hz) and
treat each tick as a simultaneous-move game. MCTS searches over the joint action
space with **opponent modeling** (the NN predicts both players' policies).

---

## 2. Game Simulator (CRSim)

### 2.1 Arena

- **Grid**: 18 × 32 tiles (width × height)
- **River**: rows 15–16, passable only at two bridge columns (3–4 and 13–14)
- **Towers**: per side — 1 King Tower (center back), 2 Princess Towers (flanks)
- **Placement zones**: your own half (rows 0–14 for player 0, rows 17–31 for player 1),
  excluding tower tiles and river

```
Row 31  ┌──────────────────────────────────┐
        │          ENEMY KING (9,30)       │
Row 29  │   PRINCESS(4,27)  PRINCESS(13,27)│
        │                                  │
        │         ENEMY HALF               │
Row 17  │~~BRIDGE~~  RIVER  ~~BRIDGE~~     │
Row 16  │~~(3-4)~~  (river) ~~(13-14)~~   │
        │         ALLY HALF                │
Row 2   │   PRINCESS(4,4)   PRINCESS(13,4) │
Row 0   │          ALLY KING (9,1)         │
        └──────────────────────────────────┘
```

### 2.2 Time Model

- **Tick rate**: 2 Hz (one tick = 0.5 real seconds)
- **Regular time**: 360 ticks (3 minutes)
- **Overtime**: 360 ticks (3 minutes, 2× elixir regen)
- **Sudden death**: 360 ticks (3 minutes, 3× elixir)
- **Elixir regen**: 1 elixir per 2.8 s → ~0.179 per tick (normal),
  doubled/tripled in overtime/sudden death

### 2.3 Card Roster (20 Core Cards)

| Card            | Type     | Cost | HP   | DPS  | Speed  | Range | Target   |
|-----------------|----------|------|------|------|--------|-------|----------|
| Knight          | Troop    | 3    | 1452 | 167  | Medium | Melee | Ground   |
| Archers         | Troop    | 3    | 304  | 107  | Medium | 5.0   | Air+Gnd  |
| Musketeer       | Troop    | 4    | 598  | 176  | Medium | 6.0   | Air+Gnd  |
| Giant           | Troop    | 5    | 3344 | 120  | Slow   | Melee | Buildings|
| Mini PEKKA      | Troop    | 4    | 1056 | 325  | Fast   | Melee | Ground   |
| Valkyrie        | Troop    | 4    | 1654 | 126  | Medium | Melee | Ground   |
| Wizard          | Troop    | 5    | 598  | 176  | Medium | 5.5   | Air+Gnd  |
| Hog Rider       | Troop    | 4    | 1408 | 176  | V.Fast | Melee | Buildings|
| Minions         | Troop    | 3    | 190  | 84   | Fast   | 2.0   | Air+Gnd  |
| Baby Dragon     | Troop    | 4    | 1064 | 100  | Fast   | 3.5   | Air+Gnd  |
| Skeleton Army   | Troop    | 3    | 67×15| 67×15| Fast   | Melee | Ground   |
| Goblin Barrel   | Spell    | 3    | 167×3| 99×3 | —      | —     | Ground   |
| Fireball        | Spell    | 4    | —    | 572  | —      | 2.5   | Area     |
| Arrows          | Spell    | 3    | —    | 243  | —      | 4.0   | Area     |
| Zap             | Spell    | 2    | —    | 159  | —      | 2.5   | Area     |
| Lightning       | Spell    | 6    | —    | 877  | —      | 3.5   | 3 targets|
| Cannon          | Building | 3    | 742  | 127  | —      | 5.5   | Ground   |
| Inferno Tower   | Building | 5    | 1408 | 40→400| —     | 6.0   | Air+Gnd  |
| Tombstone       | Building | 3    | 422  | —    | —      | —     | Spawner  |
| Elixir Collector| Building | 6    | 888  | —    | —      | —     | Elixir   |

### 2.4 Entity Update Loop (per tick)

```python
def tick(game_state):
    # 1. Regenerate elixir
    for player in [0, 1]:
        game_state.elixir[player] += elixir_rate(game_state.phase)
        game_state.elixir[player] = min(game_state.elixir[player], 10.0)

    # 2. Process queued actions (card placements from both players)
    for action in game_state.pending_actions:
        spawn_entity(game_state, action)

    # 3. Spawner buildings produce units
    process_spawners(game_state)

    # 4. Each entity: acquire target → move → attack
    for entity in game_state.entities:
        entity.target = find_target(entity, game_state)
        if distance(entity, entity.target) > entity.attack_range:
            move_toward(entity, entity.target, game_state)
        else:
            entity.attack_timer -= TICK_DURATION
            if entity.attack_timer <= 0:
                deal_damage(entity, entity.target)
                entity.attack_timer = entity.attack_interval

    # 5. Remove dead entities, check win conditions
    remove_dead(game_state)
    check_win(game_state)
```

### 2.5 Pathfinding

Troops navigate a **flow field** precomputed per target type:
- **Ground buildings-only** (Giant, Hog Rider): shortest path to nearest enemy building
  via a bridge
- **Ground any-target**: path toward nearest enemy, crossing bridges if needed
- **Air**: straight-line (no obstacles)

Flow fields are cached for each (side, target-mode) combination and recomputed only
when buildings are created/destroyed (rare). This avoids per-entity A* overhead.

### 2.6 Vectorized Simulation

For training throughput, we run N games in parallel using NumPy:

```python
class VectorizedCRSim:
    """Runs N independent games in a single batched update."""
    def __init__(self, n_envs: int):
        # Entity storage: [n_envs, max_entities, feature_dim]
        self.positions = np.zeros((n_envs, MAX_ENTITIES, 2), dtype=np.float32)
        self.hp        = np.zeros((n_envs, MAX_ENTITIES), dtype=np.float32)
        self.alive     = np.zeros((n_envs, MAX_ENTITIES), dtype=bool)
        # ...
```

---

## 3. State Representation

### 3.1 Spatial Features (18 × H=32 × W=18)

Rather than one plane per card type (which scales badly with a 125-card pool),
the encoder uses **18 fixed semantic planes**. Card identity is carried by the
entity tokens (§3.3) and the scalar one-hots (§3.2), not by the spatial grid.

| Channel | Constant                 | Description                                   |
|---------|--------------------------|-----------------------------------------------|
| 0 / 1   | `CH_FRIENDLY/ENEMY_HP`   | HP-weighted unit density                      |
| 2 / 3   | `CH_FRIENDLY/ENEMY_DPS`  | DPS-weighted threat density                   |
| 4 / 5   | `CH_FRIENDLY/ENEMY_GROUND` | Ground-unit density                         |
| 6 / 7   | `CH_FRIENDLY/ENEMY_AIR`  | Air-unit density                              |
| 8 / 9   | `CH_FRIENDLY/ENEMY_BUILDING` | Non-tower building HP                     |
| 10 / 11 | `CH_FRIENDLY/ENEMY_TOWER_HP` | Tower HP at the tower cell                |
| 12      | `CH_PLACEMENT_MASK`      | Valid placement cells for the current player  |
| 13      | `CH_STATIC_MAP`          | River = 1.0, bridge = 0.5                      |
| 14 / 15 | `CH_FRIENDLY/ENEMY_WINCON` | Building-targeting (win-condition) density  |
| 16 / 17 | `CH_FRIENDLY/ENEMY_READY` | Attack-ready unit density                    |

**Total: 18 spatial planes** (`SPATIAL_CHANNELS`), each `ARENA_H × ARENA_W` =
32 × 18. Densities are Gaussian-splatted (σ = 0.5 tiles).

### 3.2 Scalar Features (length `SCALAR_FEATURES` = 641)

`SCALAR_FEATURES = 2 + 5 × NUM_CARD_TYPES + 14`, which is **641** for the
125-card pool. The per-card blocks are one-hot/intensity vectors over the full
card set (hand slots, next card, etc.), so this grows with the card pool:

| Feature group                          | Size                 |
|----------------------------------------|----------------------|
| Globals (elixir, phase)                | 2                    |
| Per-card blocks (× 5 over card pool)   | 5 × 125 = 625        |
| Time / tower status / tower HP / score | 14                   |

### 3.3 Entity Tokens (64 × 40)

The primary network also consumes a variable-length **entity list**: up to
`MAX_ENTITY_SLOTS = 64` units, each encoded as a `ENTITY_FEATURE_DIM = 40`-d
token (position, velocity, HP, status timers, champion readiness, evolved
state, tower flag, …) and processed by a self-attention transformer. This is
what lets the network reason about individual units regardless of card pool
size. See `model/features.py::extract_entity_features`.

### 3.4 Encoding Details

- All spatial and scalar values are normalized to approximately [0, 1] or [-1, 1]
- The state is always encoded from the **perspective of the current player** — the
  board is flipped vertically for player 1 so the network always "sees" its own side
  at the bottom. This halves the effective state space.

---

## 4. Action Space

### 4.1 Discrete Formulation

```
Action = (card_index, x, y)  or  ABILITY  or  WAIT

card_index ∈ {0, 1, 2, 3}     (4 hand slots)
x          ∈ {0, ..., 17}     (18 columns)
y          ∈ {0, ..., 31}     (32 rows)

Total actions = 4 × 18 × 32 + ABILITY + WAIT = 2306
```

### 4.2 Action Masking

Invalid actions are masked before softmax:
- Not enough elixir for the card → mask all (card, *, *) actions
- Position outside valid placement zone → mask that (card, x, y)
- Position in river/on tower → mask

The mask is a binary vector of length 2306, applied as:
```python
logits[~mask] = -1e9  # before softmax
```

---

## 5. Neural Network

### 5.1 Architecture Overview

```
Spatial Input (44×18×32)           Scalar Input (116)
       │                                   │
  Conv2d 3×3, 256                     Linear 256
       │                                   │
  20× ResBlock(256)                   ReLU + Linear 256
       │                                   │
       └───────── Concat ─────────────────┘
                    │
            ┌───────┴───────┐
            │               │
       Policy Head     Value Head
            │               │
      Conv 1×1, 2       Conv 1×1, 1
      BN + ReLU         BN + ReLU
      Flatten           Flatten
      Linear→2305       Linear→256
      (+ action mask)   ReLU
                        Linear→1
                        Tanh
            │               │
       π (policy)      v (value)
```

### 5.2 Residual Block

```
x ──→ Conv2d 3×3 → BN → ReLU → Conv2d 3×3 → BN → (+x) → ReLU
       256 filters        256 filters
```

With **Squeeze-and-Excitation** (SE) after the second BN:
```
                   ... → BN → SE(reduction=16) → (+x) → ReLU
```

SE lets the network learn channel-wise attention — e.g., "pay more attention to the
Hog Rider channel when deciding defense."

### 5.3 Hyperparameters

| Param              | Value  |
|--------------------|--------|
| Residual blocks    | 20     |
| Filters per block  | 256    |
| SE reduction ratio | 16     |
| Scalar embed dim   | 256    |
| Policy output dim  | 2305   |
| Optimizer          | AdamW  |
| Learning rate      | 2e-4 → cosine decay to 1e-5 |
| Weight decay       | 1e-4   |
| Batch size         | 2048   |
| Mixed precision    | FP16 (AMP on A100) |

### 5.4 Loss Function

```
L = L_policy + c_v * L_value + c_l2 * L_reg

L_policy = -π_mcts · log(π_nn)          (cross-entropy)
L_value  = (v_nn - z)²                  (MSE, z ∈ {-1, 0, +1})
L_reg    = Σ||θ||²                       (weight decay via AdamW)

c_v  = 1.0
c_l2 = handled by AdamW weight_decay
```

---

## 6. Monte Carlo Tree Search (MCTS)

### 6.1 Algorithm (AlphaZero variant)

For each game state, MCTS runs **N_sim = 800 simulations**.

**SELECT**: From root, traverse the tree by choosing the action that maximizes:
```
UCB(s, a) = Q(s, a) + c_puct · P(s, a) · √(N_parent) / (1 + N(s, a))
```

Where:
- Q(s,a): mean value of subtree
- P(s,a): prior probability from neural network
- N(s,a): visit count
- c_puct = 2.5

**EXPAND + EVALUATE**: At a leaf node, run the neural network to get (π, v).
Create child nodes with P(s,a) = π(a).

**BACKUP**: Propagate v back up the tree, updating Q and N for each ancestor.

### 6.2 Simultaneous Moves

Since both players act simultaneously, we use a **double-oracle** approach:

1. At each MCTS node, the state includes both players' perspectives
2. The "move" at each node is a joint action (a₁, a₂)
3. In practice, we sample the opponent's action from their policy (predicted by the NN)
   and search only over our own actions

This is equivalent to **Smooth UCT** — it converges to a Nash equilibrium in
self-play, which is exactly what we want.

### 6.3 Root Enhancements

- **Dirichlet noise**: At the root node, add exploration noise:
  `P(s,a) = 0.75 · π_nn(a) + 0.25 · Dir(α=0.15)`
- **Temperature**: For the first 20 moves, sample proportional to visit counts:
  `π(a) ∝ N(s,a)^(1/τ)` with τ=1. After move 20, τ→0 (pick the most-visited).

### 6.4 Virtual Loss (for parallel search)

When running MCTS with multiple threads, apply virtual loss = 3 to avoid thread
collision on the same path. This encourages diverse exploration.

---

## 7. Self-Play Pipeline

### 7.1 Architecture

```
                        ┌──────────────────────────┐
                        │   Parameter Server        │
                        │   (latest model weights)  │
                        └───────────┬──────────────┘
                                    │ broadcast
                 ┌──────────────────┼──────────────────┐
                 │                  │                   │
        ┌────────┴──────┐  ┌───────┴───────┐  ┌───────┴───────┐
        │ Self-Play      │  │ Self-Play      │  │ Self-Play      │
        │ Worker (GPU 0) │  │ Worker (GPU 1) │  │ ... (GPU 5)   │
        │ 64 parallel    │  │ 64 parallel    │  │ 64 parallel    │
        │ games          │  │ games          │  │ games          │
        └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
                │                   │                    │
                └───────────────────┼────────────────────┘
                                    │ (s, π, z) tuples
                                    v
                        ┌──────────────────────────┐
                        │   Replay Buffer           │
                        │   (500K positions,        │
                        │    ring buffer)            │
                        └───────────┬──────────────┘
                                    │ sample batches
                                    v
                        ┌──────────────────────────┐
                        │   Trainer (GPU 6–7, DDP)  │
                        │   batch=2048, AdamW       │
                        └──────────────────────────┘
```

### 7.2 GPU Allocation (8× A100 80GB)

| GPUs  | Role           | Details                                           |
|-------|----------------|---------------------------------------------------|
| 0–5   | Self-play      | 64 vectorized envs/GPU × MCTS, ~200 games/min    |
| 6–7   | Training (DDP) | Batch 2048, ~15 updates/sec with AMP              |

**Estimated throughput**: ~1200 games/minute → ~1.7M games in 24 hours.

### 7.3 Game Generation Loop

```python
def self_play_worker(gpu_id, model, replay_buffer):
    envs = VectorizedCRSim(n_envs=64)
    mcts = BatchedMCTS(model, n_simulations=800, c_puct=2.5)

    while True:
        # Get latest model weights (async)
        model.load_state_dict(param_server.get_latest())

        # Play a batch of games
        trajectories = []
        obs = envs.reset()

        while not envs.all_done():
            # MCTS search (batched NN inference)
            policies = mcts.search(obs, envs)  # [64, 2305]
            actions = sample_action(policies, temperature)
            obs, rewards, dones = envs.step(actions)
            trajectories.append((obs, policies, actions))

        # Assign game outcomes and push to replay buffer
        for traj in process_trajectories(trajectories, rewards):
            replay_buffer.push(traj)
```

---

## 8. Training Loop

### 8.1 Training Step

```python
def train_step(model, optimizer, batch):
    states, target_policies, target_values = batch

    # Forward pass (AMP)
    with torch.cuda.amp.autocast():
        pred_policies, pred_values = model(states)
        policy_loss = -(target_policies * pred_policies.log_softmax(-1)).sum(-1).mean()
        value_loss = F.mse_loss(pred_values.squeeze(), target_values)
        loss = policy_loss + value_loss

    # Backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### 8.2 Training Schedule

| Phase   | Hours | Description                                   |
|---------|-------|-----------------------------------------------|
| Warmup  | 0–2   | Random play, fill replay buffer               |
| Phase 1 | 2–8   | Self-play with MCTS (200 sims), τ=1.0        |
| Phase 2 | 8–16  | Self-play with MCTS (400 sims), anneal τ      |
| Phase 3 | 16–22 | Full MCTS (800 sims), lower LR                |
| Phase 4 | 22–24 | Final polish, evaluation, model selection      |

### 8.3 Curriculum

Start simple, add complexity:
1. **Hours 0–4**: Fixed 8-card "starter" deck vs itself
2. **Hours 4–12**: Random deck from the 125-card pool
3. **Hours 12–24**: Deck building included as part of the strategy

---

## 9. Evaluation

### 9.1 Model Checkpointing

Every 30 minutes:
1. Save model checkpoint
2. Play 100 games against the previous best model
3. If win rate > 55%, promote to new best model
4. Track Elo rating over time

### 9.2 Elo System

- Initial Elo: 1000
- K-factor: 32
- Track Elo for every checkpoint
- Plot Elo curve to visualize learning progress

### 9.3 Diagnostic Metrics

- Games per second
- Average game length
- Policy entropy (should decrease over training)
- Value prediction accuracy
- Elixir efficiency (average elixir spent vs damage dealt)
- Win rate by deck composition

---

## 10. Deployment & Inference

For real-time play against humans:
- Single GPU inference with MCTS (200 sims) runs in < 250ms per decision
- Decision made every few ticks (decision interval) = well within real-time
- Can interface with the actual game via screen capture + touch simulation
  (requires separate integration layer)

---

## 11. Key Design Decisions

| Decision                        | Rationale                                            |
|---------------------------------|------------------------------------------------------|
| 50 ms tick rate (20 Hz)         | Balances fidelity vs search depth                    |
| 125-card roster                 | Authentic Supercell card set (+10 champions, 35 evos) |
| 18×32 grid                     | Matches actual arena proportions                      |
| 256 filters, 20 ResBlocks      | Sweet spot for A100 throughput vs capacity            |
| 800 MCTS simulations           | AlphaZero default; 200–400 used in early phases      |
| Opponent action sampling        | Practical approximation for simultaneous moves        |
| Flow-field pathfinding          | O(1) per entity per tick; precomputed                 |
| Ring replay buffer              | Prioritizes recent self-play data                     |

---

## 12. Extensions (Post-24h)

1. **Full card roster** (100+ cards)
2. **Learned deck building** (meta-game optimization)
3. **Opponent modeling** (adapt to specific play styles)
4. **Transfer to real game** (screen capture + touch API integration)
5. **Population-based training** (PBT for hyperparameter tuning)
6. **Larger network** (40 ResBlocks, 384 filters) with more compute
