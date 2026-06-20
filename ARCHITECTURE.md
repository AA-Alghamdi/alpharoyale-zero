# ClashRoyale-Zero: Architecture Document

## 1. System Overview

ClashRoyale-Zero is an AlphaZero-style RL system that learns Clash Royale play
through self-play. It couples a deterministic game simulator (125 cards,
6 champions, 35 evolutions) with two neural architectures, Gumbel-MuZero
search, and an AlphaStar-style training league.

```
                              TRAINING LOOP
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ┌────────────┐     ┌─────────────┐     ┌────────────────────────┐ │
  │  │  CRSim     │◄───►│ Gumbel-     │◄───►│  CRStarNet             │ │
  │  │  (crsim/)  │     │ MuZero      │     │  entity-transformer    │ │
  │  │  50ms tick  │     │ 16 sims,    │     │  + spatial ResNet      │ │
  │  │  18×32 grid │     │ seq-halving │     │  + LSTM core           │ │
  │  └─────┬──────┘     └─────────────┘     │  + autoregressive head │ │
  │        │                                 └──────────┬─────────────┘ │
  │        │ trajectories                               │ gradients     │
  │        ▼                                            │               │
  │  ┌──────────────────┐    ┌─────────────────────────┘               │
  │  │ Replay Buffer    │───►│ TrainerV2                                │
  │  │ 50K ring, f16    │    │ AdamW + AMP + cosine LR                 │
  │  │ PER sampling     │    │ aux heads + dynamics + pruning           │
  │  └──────────────────┘    └─────────────────────────────────────────┘
  │                                                                     │
  │  ┌──────────────────────────────────────────────────────────────┐   │
  │  │  AlphaStar League                                            │   │
  │  │  3 main + 2 league-exploiters + 2 main-exploiters            │   │
  │  │  PFSP opponent selection, exploiter reset at 70% WR          │   │
  │  └──────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────┘

                          REAL-TIME PIPELINE
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Screen ──► YOLOv8 + OCR ──► PerceivedState ──► CRStarNet ──► ADB │
  │  (realtime/perception.py)   (realtime/pipeline.py)  (controller.py)│
  └─────────────────────────────────────────────────────────────────────┘
```

**Shipped defaults vs. production target.**  The code ships with lightweight
defaults (Gumbel 16-sim, CRStarNet 128 filters / 10 blocks) that run on CPU.
Config knobs scale to GPU hardware: 800-sim MCTS, 256 filters / 20 blocks,
8×A100 DDP. All interface dimensions below match the shipped code.

---

## 2. Game Simulator (`crsim/`)

### 2.1 Arena Geometry

- **Grid**: `ARENA_W=18` columns × `ARENA_H=32` rows
- **River**: rows 15-16, passable at two bridges (cols 3-4 and 13-14)
- **Towers**: per side: 1 King Tower (center-back) + 2 Princess Towers (flanks)
- **Placement zones**: your own half, excluding towers and river

```
Row 31  ┌──────────────────────────────────┐
        │          ENEMY KING (9,30)       │
Row 29  │   PRINCESS(4,27)  PRINCESS(13,27)│
        │         ENEMY HALF               │
Row 17  │                                  │
Row 16  │~~BRIDGE~~  RIVER  ~~BRIDGE~~     │
Row 15  │~~(3-4)~~         ~~(13-14)~~     │
        │         ALLY HALF                │
Row 4   │   PRINCESS(4,4)   PRINCESS(13,4) │
Row 1   │          ALLY KING (9,1)         │
Row 0   └──────────────────────────────────┘
```

### 2.2 Time Model

| Phase        | Duration     | Elixir Rate      | Trigger                    |
|-------------|-------------|------------------|----------------------------|
| Regular     | 360 ticks   | 0.1786/tick      | Game start                 |
| Double Elixir| continues  | 2× (0.3571/tick) | 2:00 remaining             |
| Overtime    | 360 ticks   | 2× rate          | Tie at regulation end      |
| Sudden Death| 360 ticks   | 3× rate          | Tie at overtime end        |

- **Tick duration**: 50ms (20 Hz). Set via `TICK_DURATION`.
- **Decision interval**: 10 ticks (0.5s) — agents re-decide every 10 ticks.
- **Max elixir**: 10.0, regen capped at max.

### 2.3 Card Roster

**125 cards** across three entity kinds:

| Kind       | Count | Examples                                         |
|-----------|-------|--------------------------------------------------|
| Troops    | ~90   | Knight, Archers, PEKKA, Golem, Mega Knight, ...  |
| Spells    | ~15   | Fireball, Zap, Lightning, Poison, Freeze, ...    |
| Buildings | ~20   | Cannon, Inferno Tower, Elixir Collector, ...     |

All stats sourced from authentic Supercell Level 11 tournament-standard data.
Cards are enumerated in `CardType` (0=KNIGHT through 124=GUARDIENNE). Each card
has a `CardDef` with ~60 fields covering cost, HP, DPS, damage, hit speed,
range, sight range, speed, targeting, spawning, charge, shield, death spawn,
death damage, splash, crown tower damage %, inferno ramp, and more.

**6 Champions**: Archer Queen, Golden Knight, Skeleton King, Mighty Miner, Monk,
Little Prince. Each has:
- Manually-activated ABILITY action (separate from card placement)
- Ability elixir cost + cooldown timer
- Unique ability effects (cloak+rage, dash+chain, army-of-dead, underground,
  deflect, lance+charge)

**35 Evolutions** (defined in `crsim/evolutions.py`): Barbarians, Knight,
Archers, Valkyrie, Tesla, PEKKA, Mega Knight, etc. Each has:
- Cycle count (1-2 deploys before evolved form available)
- Special ability (damage reduction, spawn-on-attack, knockback, heal, etc.)
- Optional stat boosts (HP%, attack speed%)

### 2.4 Entity System (`crsim/entities.py`)

Each in-game unit is an `Entity` dataclass with 80+ fields:

```
Core:       entity_id, card_type, owner, x, y, hp, max_hp
Combat:     dps, damage_per_hit, hit_speed, attack_range, sight_range
            attack_timer, target_id, target_mode (GROUND/AIR_AND_GROUND/BUILDINGS)
Movement:   speed, is_flying, velocity_x, velocity_y
Mechanics:  is_charging, charge_distance, charge_damage_mult
            has_shield, shield_hp
            death_spawn_card_type, death_spawn_count, death_spawn_hp
            death_damage, death_damage_radius
            is_splash, splash_radius, crown_tower_damage_percent
            minimum_range, hit_count
Status:     freeze_timer, slow_timer, stun_timer, poison_timer, rage_timer
            invisible_timer, damage_reduction_timer
Champion:   is_champion, ability_cost, ability_cooldown, ability_cooldown_timer
            attack_speed_timer
Evolution:  is_evolved, evo_shield, evo_shield_mult
```

### 2.5 Game Loop (`crsim/game.py`)

`CRGame.step(actions: list[Action])` advances one tick:

```
1. Regenerate elixir for both players
2. Process queued actions:
   - Card placements → spawn entities (deduct elixir, cycle hand)
   - Champion ability → dispatch to ability handler
3. Spawner buildings produce units (if interval elapsed)
4. For each entity:
   a. Acquire target (nearest enemy matching target_mode)
   b. Move toward target (flow-field pathfinding)
   c. If in range: attack (apply damage, splash, death effects)
   d. Tick status effects (freeze, slow, poison, rage, stun)
5. Remove dead entities (trigger death spawns, death damage)
6. Check win conditions (tower destruction, crown count, time)
7. Handle game phase transitions (overtime, sudden death)
```

### 2.6 Pathfinding (`crsim/pathfinding.py`)

Pre-computed **flow fields** per (side, target-mode):
- Ground buildings-only: shortest path to nearest enemy building via bridge
- Ground any-target: path toward nearest enemy through bridges
- Air: straight-line (ignore terrain)

Cached and recomputed only when buildings are created/destroyed. O(1) per
entity per tick via flow field lookup.

### 2.7 Vectorized Self-Play (`training/vectorized_selfplay.py`)

Runs `n_envs` (default 32) games in lockstep:
- All active positions evaluated in a **single batched forward pass**
- Both engine backends supported: Python (`CRGame`) or Rust (`CRGameRust`)
- Temperature annealing: full exploration → greedy after N decisions
- Horizontal flip augmentation (50% of games)
- Produces `ReplayEntry` tuples with spatial, scalar, entity, policy, value,
  and auxiliary targets

---

## 3. State Representation (`model/features.py`)

### 3.1 Spatial Features: `(18, ARENA_H=32, ARENA_W=18)`

18 fixed semantic planes (not one-per-card — scales to any card pool):

| Ch  | Name                       | Content                              |
|-----|----------------------------|--------------------------------------|
| 0-1 | `FRIENDLY/ENEMY_HP`        | HP-weighted Gaussian density         |
| 2-3 | `FRIENDLY/ENEMY_DPS`       | DPS-weighted threat density          |
| 4-5 | `FRIENDLY/ENEMY_GROUND`    | Ground-unit density                  |
| 6-7 | `FRIENDLY/ENEMY_AIR`       | Air-unit density                     |
| 8-9 | `FRIENDLY/ENEMY_BUILDING`  | Non-tower building HP                |
|10-11| `FRIENDLY/ENEMY_TOWER_HP`  | Tower HP at tower cell               |
| 12  | `PLACEMENT_MASK`           | Valid placement cells for this player|
| 13  | `STATIC_MAP`               | River=1.0, bridge=0.5                |
|14-15| `FRIENDLY/ENEMY_WINCON`    | Building-targeting unit density      |
|16-17| `FRIENDLY/ENEMY_READY`     | Attack-ready unit density            |

Densities are Gaussian-splatted (σ=0.5 tiles) around entity positions.

### 3.2 Scalar Features: length `SCALAR_FEATURES = 641`

`641 = 2 + 5×125 + 14`:

| Group                        | Size      | Detail                          |
|-----------------------------|-----------|---------------------------------|
| Globals                     | 2         | elixir/10, game phase           |
| Hand cards (4 slots)        | 4×125=500 | One-hot over card pool per slot |
| Next card                   | 125       | One-hot over card pool          |
| Time remaining              | 1         | normalized to [0,1]             |
| Tower alive flags           | 6         | 3 per side (king, L, R)         |
| Tower HP                    | 6         | 3 per side, normalized          |
| Crown score diff            | 1         | (my-opp)/3                      |

### 3.3 Entity Tokens: `(MAX_ENTITY_SLOTS=64, ENTITY_FEATURE_DIM=40)`

Variable-length entity list for the EntityTransformer:

```
Per entity (40-d):
  type_id / NUM_CARD_TYPES          (card identity, normalized)
  owner                             (0=friendly, 1=enemy)
  x / ARENA_W, y / ARENA_H         (position, normalized)
  hp / max_hp                       (health fraction)
  dps, damage_per_hit               (normalized combat stats)
  hit_speed, attack_range           (attack parameters)
  speed, is_flying                  (movement)
  target_mode                       (ground/air/buildings encoding)
  is_splash, splash_radius          (area damage)
  attack_timer / hit_speed          (readiness)
  has_shield, shield_hp             (shield state)
  is_charging                       (charge state)
  freeze/slow/stun/poison_timer     (status effects)
  is_champion, ability_ready        (champion state)
  is_evolved                        (evolution state)
  is_tower                          (tower flag)
```

Mask: `(64,)` bool — True for occupied slots, False for padding.

### 3.4 Perspective Normalization

State is always encoded from the **acting player's perspective**:
- Player 1's board is flipped vertically so the NN always sees its own
  side at the bottom. This halves the effective state space.

### 3.5 Auxiliary Targets (`extract_auxiliary_targets`)

For richer training signal:
- **Crown target**: 7-class (crown diff from -3 to +3)
- **Tower HP target**: `(6,)` — HP fraction of all 6 towers at game end
- **Game length target**: normalized final tick count

---

## 4. Action Space (`crsim/actions.py`, `crsim/constants.py`)

### 4.1 Layout

```
ACTION_SPACE_SIZE = 2306

[0, 2304)       Card placements: slot × ARENA_W × ARENA_H
                 id = slot * 576 + x * 32 + y
                 slot ∈ {0,1,2,3}, x ∈ {0..17}, y ∈ {0..31}

2304             ABILITY_ACTION — activate champion ability
2305             WAIT_ACTION — do nothing (bank elixir)
```

### 4.2 Action Masking

Invalid actions masked to `-1e9` before softmax:
- Not enough elixir for card in that slot
- Position outside valid placement zone (enemy half, river, tower tiles)
- ABILITY masked when no champion deployed or on cooldown
- WAIT is always valid

### 4.3 Autoregressive Decomposition (CRStarNet)

For training efficiency, CRStarNet also supports autoregressive action
factoring: sample **card_slot** (5-way: 4 slots + wait) → **x_pos** (18-way)
→ **y_pos** (32-way) sequentially, reducing the effective branching factor.

---

## 5. Neural Networks

### 5.1 CRZeroNet (`model/network.py`) — Baseline

```
Spatial (18,32,18) ──► Conv(18→256) ──► 20× SE-ResBlock(256) ──► global pool
                                                                      │
Scalar (641,)      ──► Linear(641→256) ──► ReLU ──► Linear(256) ──────┤
                                                                      │ concat
                                                                      ▼
                                                                   merged
                                                    ┌─────────────────┤
                                                    ▼                 ▼
                                              Policy Head       Value Head
                                              Conv→Linear       Conv→Linear
                                              → 2306 logits     → Tanh → (1,)
```

SE-ResBlock: `Conv3×3 → BN → ReLU → Conv3×3 → BN → SE(r=16) → (+skip) → ReLU`

### 5.2 CRStarNet (`model/transformer_net.py`) — Primary

Five parallel encoding towers fused through an LSTM core:

```
Entity tokens (64,40)     Spatial (18,32,18)     Scalar (641,)   Belief (128,)
       │                         │                     │              │
  EntityEncoder             SpatialEncoder        ScalarEncoder  BeliefEncoder
  3-layer self-attn         10× SE-ResBlock       641→256→128    128→64
  CLS aggregation           global avg pool
  → (128,)                  → (128,)              → (128,)       → (64,)
       │                         │                     │              │
       └─────────────┬───────────┘                     │              │
                     │         ┌────────────────────────┘              │
                     └─────────┤                                      │
                               ▼                                      │
                         concat (128+128+128+64 = 448) ◄──────────────┘
                               │
                         fusion Linear(448→512)
                               │
                         LSTMCore (2-layer, hidden=512)
                               │
                     ┌─────────┼─────────┬──────────┐
                     ▼         ▼         ▼          ▼
               FlatPolicy   Value   AuxHeads   OpponentHead
               →2306        →Tanh   crown(7)   deck(125)
                                    tower(6)   next(125)
                                    length(1)  elixir(1)
                     │
              AutoregressivePolicyHead
              card(5) → x(18) → y(32)
```

**EntityEncoder**: type embedding (32-d) + input projection to 128-d, 3
transformer layers with 4 heads each, CLS token aggregation for a fixed-size
(128,) summary.

**DynamicsModel** (EfficientZero-style): `state + action_embed → next_state +
reward`. Used for model-based planning and trained with dynamics loss.

### 5.3 Hyperparameters

| Param                | Default (shipped)     | Production target    |
|---------------------|-----------------------|----------------------|
| Spatial filters      | 128                   | 256                  |
| Spatial ResBlocks    | 10                    | 20                   |
| SE reduction ratio   | 16                    | 16                   |
| Entity embed dim     | 128                   | 128                  |
| Entity transformer layers | 3              | 6                    |
| LSTM hidden          | 512                   | 512                  |
| Optimizer            | AdamW                 | AdamW                |
| Learning rate        | 3e-4 → cosine decay   | 3e-4 → 1e-5         |
| Weight decay         | 1e-4                  | 1e-4                 |
| Batch size           | 2048                  | 2048                 |
| Grad clip            | 1.0                   | 1.0                  |
| Mixed precision      | FP16 AMP              | FP16 AMP             |

### 5.4 Loss Function

```
L = L_policy + L_value + 0.3 × L_aux + 0.5 × L_dynamics

L_policy   = -π_target · log_softmax(logits)        (KL from MCTS targets)
L_value    = MSE(v_pred, z)                          (z ∈ {-1, 0, +1})
L_aux      = CE(crown_logits, crown_target)
           + MSE(tower_hp_pred, tower_hp_target)
           + MSE(game_length_pred, game_length_target)
L_dynamics = MSE(predicted_next_state, detach(actual_next_state))
```

**KataGo policy target pruning**: actions with visit count < 2% of total are
zeroed from the target distribution to prevent learning MCTS noise.

---

## 6. Search (`mcts/`)

### 6.1 Gumbel-MuZero (`mcts/gumbel_search.py`) — Primary

From "Policy improvement by planning with Gumbel" (ICLR 2022):

1. **Gumbel-Top-k**: add Gumbel noise to log-priors, select top-k actions
   (k=16 by default) without replacement
2. **Sequential Halving**: allocate simulations in phases, halving the
   candidate set each phase — guaranteed policy improvement
3. **Improved policy**: `π_improved(a) ∝ π(a) · exp(Q(a) / σ)` where
   `σ = c_scale × (c_visit + max_visits)`

16 simulations instead of 800 → ~50× faster inference.

```
Config defaults:
  n_simulations:       16
  max_considered:      16
  c_visit:             50.0
  c_scale:             1.0
  dirichlet_alpha:     0.15
  dirichlet_frac:      0.25
  playout_cap_rand:    True (KataGo-style, range [4, 32])
  rollout_ticks:       8 (roll forward one decision interval before bootstrap)
```

Each simulation: apply candidate action → sample opponent action from policy →
roll forward `rollout_ticks` with WAITs → bootstrap value from NN. This ensures
the bootstrap reflects the action's consequences.

### 6.2 Vanilla MCTS (`mcts/search.py`)

Standard AlphaZero UCB search: `UCB(s,a) = Q(s,a) + c_puct · P(s,a) · √N_parent / (1+N(s,a))`.
Supports virtual loss for parallel search.

### 6.3 MuZero Search (`mcts/muzero_search.py`)

Extends vanilla MCTS with a learned dynamics model: instead of cloning the
game, the search operates in latent space using `DynamicsModel` predictions.

### 6.4 Simultaneous Moves

Both players act simultaneously. The search samples the opponent's action from
the NN's predicted policy and searches only over our own actions — equivalent
to **Smooth UCT**, converging to Nash equilibrium in self-play.

---

## 7. Training Pipeline

### 7.1 Data Flow

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                     DATA FLOW                                   │
  │                                                                 │
  │  CRGame.step()                                                  │
  │       │                                                         │
  │       ▼                                                         │
  │  encode_state(game, player)                                     │
  │       │                                                         │
  │       ├──► spatial (18, 32, 18)  ──┐                            │
  │       ├──► scalar (641,)         ──┤                            │
  │       └──► entity (64, 40) + mask──┤                            │
  │                                    ▼                            │
  │                              CRStarNet.forward()                │
  │                                    │                            │
  │                         ┌──────────┼──────────┐                 │
  │                         ▼          ▼          ▼                 │
  │                    policy(2306) value(1) aux_targets             │
  │                         │          │                            │
  │                         ▼          ▼                            │
  │                    action sampling + game outcome                │
  │                         │                                       │
  │                         ▼                                       │
  │                    ReplayEntry                                   │
  │                    {spatial, scalar, entity, mask,               │
  │                     policy, value, crown, tower_hp, game_len}   │
  │                         │                                       │
  │                         ▼                                       │
  │                    ReplayBuffer (50K ring, float16)              │
  │                         │                                       │
  │                         ▼                                       │
  │                    TrainerV2.train_step()                        │
  │                    sample_full(2048) → forward → loss → backward│
  └─────────────────────────────────────────────────────────────────┘
```

### 7.2 Replay Buffer (`training/replay_buffer.py`)

- **Ring buffer**: capacity 50K positions (configurable)
- **Storage**: spatial + entity features in **float16** to halve memory;
  scalar/policy in float32
- **Sampling**: uniform `sample()` or `sample_full()` (with entity + aux);
  also `sample_prioritized()` with PER (alpha=0.6, beta=0.4)
- Thread-safe with lock for concurrent self-play workers + trainer

### 7.3 TrainerV2 (`training/trainer_v2.py`)

- **AdamW** with cosine annealing warm restarts (T_0=10K, T_mult=2)
- **FP16 AMP** via GradScaler
- **Gradient clipping**: max norm 1.0
- **KataGo policy pruning**: zero targets below 2% of total visits
- **Auxiliary losses**: crown CE + tower HP MSE + game length MSE (weight 0.3)
- **Dynamics loss**: predict next latent state from current + action (weight 0.5)
- **Checkpointing**: every 5K steps, saves model + optimizer + scheduler +
  scaler state
- **TensorBoard logging**: loss/policy, loss/value, loss/aux, loss/dynamics,
  grad_norm, lr, buffer_size

### 7.4 Imitation Warm-Start (`training/imitation.py`)

Two warm-start methods to bootstrap before self-play:

1. **Behavioral Cloning**: record `(state, action, outcome)` from MetaAgent
   playing on the verified engine → supervise policy (CE) + value (MSE).
   `generate_expert_dataset()` → `.npz` → `train_behavioral_cloning()`.

2. **Value Warm-Start**: regress value head on deck matchup outcomes from
   external data (Kaggle 37.9M-match dump, KataCR replays). Encode each
   matchup's initial state, fit `V ≈ outcome`. Accepts JSONL/CSV formats.

### 7.5 AlphaStar League (`training/league.py`)

Three agent types (from Vinyals et al., Nature 2019):

| Type              | Count | Opponent Selection    | Purpose                   |
|-------------------|-------|-----------------------|---------------------------|
| Main Agent        | 3     | PFSP (80%) + self (20%)| Final deployed agents    |
| League Exploiter  | 2     | Uniform over all      | Find global exploits      |
| Main Exploiter    | 2     | Main agents only      | Patch main agent weaknesses|

**PFSP** (Prioritized Fictitious Self-Play): priority `(1 - win_rate)^p` —
harder opponents sampled more often. Win rates tracked as exponential moving
average.

**Exploiter reset**: when mean win rate > 70%, snapshot weights to the frozen
pool, reset to initial supervised weights, clear history. The exploit becomes a
"hard opponent" for main agents.

### 7.6 Scale-Up Architecture (GPU)

```
  GPU 0-5: Self-play workers (VectorizedSelfPlay)
           64 envs/GPU × batched NN eval
           ~200 games/min per GPU → ~1200 total

  GPU 6-7: Training (DDP)
           batch=2048, ~15 updates/sec with AMP

  CPU:     League coordinator, replay buffer, checkpointing
```

Estimated: ~1.7M games in 24h on 8×A100. Self-play workers pull latest weights
from parameter server asynchronously.

---

## 8. Evaluation (`eval/`)

### 8.1 Reference Agents (`eval/baseline_agents.py`)

| Agent          | Description                                            |
|---------------|--------------------------------------------------------|
| WaitAgent      | Never plays — trivial lower bound                      |
| RandomAgent    | Random legal placement with configurable play_prob     |
| HeuristicAgent | Defend threatened lane, push when elixir is banked     |
| MetaAgent      | Multi-strategy: counter-push, beatdown, cycle, spell   |
| SearchAgent    | Wraps GumbelMuZeroSearch for NN-based play             |

### 8.2 Elo Ladder (`eval/ladder.py`)

Order-independent Bradley-Terry Elo from full round-robin:
- Play every pair `n_games` times (sides alternate)
- Fit Elo via maximum-likelihood MM iteration
- Bayesian prior (2 virtual draws vs phantom) regularizes extreme records
- Mean-anchored to 1000 for comparability across runs

### 8.3 Tournament (`eval/tournament.py`)

Online Elo with K-factor — for tracking a moving training run, not for final
ranking.

---

## 9. Verification (`verification/`)

### 9.1 Cross-Engine Conformance (`verification/conformance.py`)

Compare `crsim` card stats against the vendored oracle (samdickson22/
clash-simulator data):
- Per-card: HP, damage, hit_speed, range, sight_range, speed, mana cost
- Tolerances: 8% for HP/damage, 12% for hit_speed/sight_range, 0.35 tiles
  for range/radius
- Kind-aware: spells only compare radius; troops/buildings compare full stats

### 9.2 Behavioral Tests (`tests/test_interactions.py`)

24+ golden interaction scenarios:
- Fireball+Zap kills Musketeer
- Knight vs Knight symmetric outcome
- Giant ignores troops, walks to building
- Inferno Tower ramps damage
- Shield break mechanics
- Death spawn triggers (Golem, Lava Hound)
- And more

### 9.3 Cross-Engine Behavioral (`tests/test_cross_engine_behavioral.py`)

Run identical scenarios on both Python engine and oracle, compare:
- Duel outcomes (5/6 duels agree as of R1)
- Turn-by-turn damage

---

## 10. Perception Pipeline (`realtime/`)

### 10.1 Perception (`realtime/perception.py`)

Screen capture → game state extraction:

1. **Entity detection**: YOLOv8 object detection (troops, buildings, spells)
   with fallback to color segmentation (blue=friendly, red=enemy)
2. **Elixir reading**: HSV purple mask on elixir bar region → fill ratio × 10
3. **Hand detection**: template matching against card art for each slot
4. **Tower HP**: green/red health bar pixel ratio at tower positions

Coordinate mapping: `screen_to_arena()` / `arena_to_screen()` for 1080×1920
reference resolution. Arena bounds: pixels (60,310)-(1020,1440).

### 10.2 Opponent Modeling (`model/opponent_model.py`)

Track hidden opponent state during real-time play:
- **PlayHistoryTracker**: cards seen, last play tick per card
- **OpponentBeliefState**: estimated elixir (regen model), deck probability
  distribution, card cycle features
- Encodes to 128-d feature vector for CRStarNet's belief encoder

### 10.3 Pipeline (`realtime/pipeline.py`)

End-to-end loop: capture → perceive → decide → act via ADB touch.
Target: <100ms per decision cycle.

```python
while running:
    frame = controller.capture_screen()       # ADB screencap
    state = perception.process_frame(frame)   # YOLOv8 + OCR
    action = decide(state)                    # CRStarNet inference
    if action:
        controller.play_card(slot, x, y)      # ADB swipe gesture
```

### 10.4 Controller (`realtime/controller.py`)

ADB-based touch control:
- Card drag: swipe from hand slot center to arena target position
- Screen capture via `adb exec-out screencap -p`
- Configurable touch delay and drag duration

---

## 11. External Data Sources

| Source                                | Format      | Used For                    |
|---------------------------------------|-------------|------------------------------|
| Supercell game data (APK-extracted)   | JSON/CSV    | Card stats via `gamedata_loader.py` |
| Kaggle 37.9M-match dataset           | CSV/JSONL   | Value warm-start matchups    |
| KataCR replay dataset                | Replays     | Behavioral cloning data      |
| samdickson22/clash-simulator          | JSON oracle | Cross-engine verification    |
| Supercell API `/v1/cards`             | JSON        | Card ID mapping              |
| cr-csv (smlbiobot)                   | CSV         | Decoded game data CSVs       |

---

## 12. Key Design Decisions

| Decision                          | Rationale                                       |
|-----------------------------------|-------------------------------------------------|
| 50ms ticks (20 Hz)                | Balances fidelity vs search depth               |
| 18 semantic spatial planes        | Scales to any card pool (vs 1 plane per card)   |
| Entity transformer + spatial ResNet| Card identity in tokens; spatial for geometry   |
| LSTM core                         | Temporal reasoning across decisions              |
| Autoregressive policy head        | card→x→y reduces effective branching 2306→5+18+32|
| Gumbel-MuZero (16 sims)          | 50× faster than vanilla 800-sim MCTS            |
| Decision interval = 10 ticks      | Human-like ~2 actions/sec                       |
| Float16 replay buffer             | Halves memory for large spatial observations     |
| Perspective-normalized encoding   | Halves effective state space                     |
| PFSP opponent selection           | Focus training on weaknesses                     |
| Exploiter reset at 70% WR        | Continuously discover new exploits               |

---

## 13. Module Map

```
clash-royale-zero/
├── crsim/                    # Game simulator
│   ├── game.py               #   CRGame: tick loop, action handling, win conditions
│   ├── entities.py           #   Entity dataclass (80+ fields)
│   ├── cards.py              #   125 CardType enum + CardDef with authentic stats
│   ├── constants.py          #   Arena geometry, timing, action space dimensions
│   ├── actions.py            #   Canonical action codec (flat id ↔ Action)
│   ├── evolutions.py         #   35 EvolutionDef with ability effects
│   ├── heroes.py             #   Champion ability definitions
│   ├── pathfinding.py        #   Flow-field pathfinding
│   ├── hidden_stats.py       #   Hidden/derived card stats
│   ├── gamedata.py           #   Legacy game data
│   ├── gamedata_loader.py    #   Load stats from JSON/CSV/API
│   └── rust_adapter.py       #   CRGameRust wrapper for native engine
│
├── model/                    # Neural networks
│   ├── network.py            #   CRZeroNet: ResNet + SE baseline
│   ├── transformer_net.py    #   CRStarNet: entity-xformer + LSTM + heads
│   ├── features.py           #   encode_state, entity features, aux targets
│   └── opponent_model.py     #   Opponent belief state tracking
│
├── mcts/                     # Search algorithms
│   ├── gumbel_search.py      #   Gumbel-MuZero (primary, 16 sims)
│   ├── search.py             #   Vanilla AlphaZero MCTS
│   ├── muzero_search.py      #   MuZero (latent-space search)
│   └── is_mcts.py            #   Information-set MCTS variant
│
├── training/                 # Training infrastructure
│   ├── vectorized_selfplay.py#   Batched self-play generation
│   ├── trainer_v2.py         #   TrainerV2: AdamW + AMP + aux + dynamics
│   ├── replay_buffer.py      #   Ring buffer with PER + entity features
│   ├── league.py             #   AlphaStar league (PFSP, exploiters)
│   ├── imitation.py          #   BC warm-start + value warm-start
│   ├── curriculum.py         #   Training curriculum stages
│   ├── distributed.py        #   DDP / multi-GPU coordination
│   ├── domain_randomization.py#  Domain randomization for robustness
│   ├── self_play.py          #   Legacy single-game self-play
│   ├── self_play_v2.py       #   V2 self-play with entity features
│   └── trainer.py            #   Legacy trainer
│
├── eval/                     # Evaluation
│   ├── baseline_agents.py    #   Wait/Random/Heuristic/Meta/Search agents
│   ├── ladder.py             #   Bradley-Terry Elo round-robin
│   ├── tournament.py         #   Online Elo tournament
│   └── evaluator.py          #   Evaluation harness
│
├── realtime/                 # Real-time play
│   ├── perception.py         #   YOLOv8 + OCR game state extraction
│   ├── pipeline.py           #   End-to-end capture→decide→act loop
│   └── controller.py         #   ADB touch controller
│
├── verification/             # Cross-engine verification
│   ├── conformance.py        #   Stat conformance vs oracle
│   ├── oracle_data.py        #   Load oracle fixture
│   └── name_map.py           #   Card name mapping crsim ↔ oracle
│
├── tests/                    # Test suite (226+ tests)
│   ├── test_interactions.py  #   24+ golden interaction scenarios
│   ├── test_encoding.py      #   State encoding invariants
│   ├── test_action_space.py  #   Action codec roundtrip
│   ├── test_cross_engine_*.py#   Oracle conformance tests
│   ├── test_transformer_net.py#  CRStarNet forward pass tests
│   ├── test_vectorized_selfplay.py
│   ├── test_league.py
│   ├── test_imitation.py
│   └── ...
│
├── scripts/                  # CLI entrypoints
│   ├── train_v2.py           #   Main training script
│   ├── evaluate.py           #   Run evaluation
│   ├── ladder.py             #   Run Elo ladder
│   ├── selfplay_bench.py     #   Benchmark self-play throughput
│   ├── warmstart.py          #   Value warm-start from matchup data
│   └── play.py               #   Real-time play script
│
├── cr_engine/                # Rust native engine (optional)
└── scroll_bridge/            # External integration bridge
```
