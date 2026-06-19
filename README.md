# ClashRoyale-Zero

**AlphaZero-style reinforcement learning agent for Clash Royale.**

Learn superhuman Clash Royale play entirely through self-play — no human data, no hand-crafted heuristics. Combines a high-fidelity game simulator, a deep residual neural network, and Monte Carlo Tree Search (MCTS).

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    SELF-PLAY LOOP                        │
│                                                          │
│  Simulator ←→ MCTS (800 sims) ←→ ResNet-20 (256 ch)    │
│       │                                 ↑               │
│       ↓                                 │               │
│  Replay Buffer (500K) ──→ Trainer (DDP, 2×A100)        │
└──────────────────────────────────────────────────────────┘
```

| Component         | Details                                              |
|-------------------|------------------------------------------------------|
| **Simulator**     | 18×32 grid, 0.5s ticks, 20 cards, flow-field pathfinding |
| **State**         | 44 spatial channels + 116 scalar features            |
| **Action space**  | 2305 (4 cards × 18×32 positions + wait)              |
| **Network**       | 20 SE-ResBlocks, 256 filters (~30M params)           |
| **MCTS**          | 800 simulations, c_puct=2.5, Dirichlet noise         |
| **Training**      | AdamW, cosine LR, FP16 AMP, 2048 batch               |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.

## Quick Start

### Install

```bash
# Clone
git clone https://github.com/AA-Alghamdi/clash-royale-zero.git
cd clash-royale-zero

# Install (Python 3.10+)
pip install -e ".[dev]"
```

### Run Tests

```bash
python -m pytest tests/ -v
# or run individually:
python tests/test_sim.py
python tests/test_model.py
python tests/test_mcts.py
```

### Train (Single GPU — development)

```bash
python scripts/train.py --mode single --device cuda:0 \
    --n-res-blocks 4 --n-filters 64 --mcts-sims 50 --batch-size 256
```

### Train (8× A100 — production)

```bash
# Option 1: Auto Ray
python scripts/train_distributed.py --auto-ray

# Option 2: Existing Ray cluster
ray start --head --num-gpus=8
python scripts/train_distributed.py
```

### Evaluate

```bash
# Against random baseline
python scripts/evaluate.py --checkpoint checkpoints/best.pt --baseline random

# Two checkpoints
python scripts/evaluate.py --checkpoint checkpoints/step_100000.pt \
                           --opponent checkpoints/step_050000.pt

# Watch a game (text rendering)
python scripts/evaluate.py --checkpoint checkpoints/best.pt --watch
```

### Play Interactively

```bash
python scripts/play.py --checkpoint checkpoints/best.pt --mode interactive
```

## 24-Hour Execution Plan (8× A100 80GB)

This is the battle plan for going from zero to a strong Clash Royale agent in 24 hours.

### Phase 0: Setup (Hour 0–1)

```bash
# On your 8×A100 machine:
pip install -e ".[dev]"
pip install ray[default] tensorboard

# Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Run tests
python -m pytest tests/ -v
```

### Phase 1: Warmup (Hours 1–4)

Starter decks only, 200 MCTS simulations. Goal: fill replay buffer, get basic learning signal.

```bash
python scripts/train_distributed.py --auto-ray \
    --mcts-sims 200 --n-sp-workers 6 --n-train-gpus 2 \
    --batch-size 2048 --max-steps 50000
```

**Expected**: ~50K training steps, buffer fills to ~100K positions.

### Phase 2: Random Decks (Hours 4–12)

Random 8-card decks from the 20-card pool, 400 MCTS sims.

```bash
python scripts/train_distributed.py --auto-ray \
    --mcts-sims 400 --max-steps 200000
```

**Expected**: ~150K additional steps. Agent learns basic strategy (elixir management, card counters).

### Phase 3: Full Training (Hours 12–22)

Full 800 MCTS simulations, continued random decks, lower learning rate.

```bash
python scripts/train_distributed.py --auto-ray \
    --mcts-sims 800 --max-steps 500000
```

**Expected**: ~300K additional steps. Agent develops advanced strategies (pushes, counter-attacks, elixir trades).

### Phase 4: Final Polish (Hours 22–24)

Evaluate, select best checkpoint, run targeted evaluation.

```bash
# Evaluate latest vs best
python scripts/evaluate.py --checkpoint checkpoints/final.pt \
    --opponent checkpoints/best_step_*.pt --n-games 200

# Watch it play
python scripts/evaluate.py --checkpoint checkpoints/best.pt --watch
```

### Expected Throughput

| Metric                  | Estimate          |
|-------------------------|-------------------|
| Games/minute (6 GPUs)   | ~200–1200        |
| Total games in 24h      | ~300K–1.7M       |
| Training steps           | ~500K            |
| Positions generated      | ~50M–100M        |
| Elo (vs random baseline) | ~2000+           |

## GPU Memory Budget

| Component         | Memory/GPU |
|-------------------|------------|
| Model (FP16)      | ~120 MB    |
| MCTS tree (800)   | ~200 MB    |
| Batch (2048)      | ~2 GB      |
| Replay buffer     | ~20 GB (shared CPU RAM) |
| **Total per GPU** | **~3 GB** (self-play), **~5 GB** (training) |

A100 80GB has abundant headroom. Consider increasing batch size or model size if training saturates early.

## Project Structure

```
clash-royale-zero/
├── ARCHITECTURE.md           # Detailed system design document
├── README.md                 # This file
├── pyproject.toml            # Project config & dependencies
├── config/
│   └── default.yaml          # Training hyperparameters
├── crsim/                    # Game simulator
│   ├── constants.py          # Arena geometry, timing, action space
│   ├── cards.py              # 20-card roster with stats
│   ├── entities.py           # Runtime entity representation
│   ├── pathfinding.py        # Flow-field pathfinding
│   └── game.py               # Core game engine
├── model/                    # Neural network
│   ├── features.py           # State → tensor encoding
│   └── network.py            # ResNet + policy/value heads
├── mcts/                     # Monte Carlo Tree Search
│   └── search.py             # MCTS with NN guidance
├── training/                 # Training pipeline
│   ├── replay_buffer.py      # Ring buffer for experience
│   ├── self_play.py          # Game generation workers
│   ├── trainer.py            # Network training with AMP
│   └── distributed.py        # Multi-GPU orchestration
├── eval/                     # Evaluation
│   └── evaluator.py          # Model comparison & Elo tracking
├── scripts/
│   ├── train.py              # Single-GPU training entrypoint
│   ├── train_distributed.py  # Multi-GPU with Ray
│   ├── evaluate.py           # Evaluation & visualization
│   └── play.py               # Interactive play
└── tests/
    ├── test_sim.py           # Simulator tests
    ├── test_model.py         # Neural network tests
    └── test_mcts.py          # MCTS tests
```

