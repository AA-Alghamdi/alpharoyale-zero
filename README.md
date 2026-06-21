# Clash Royale Zero

Clash Royale Zero is a unified research and engineering stack for building a
high-strength Clash Royale player: a real-time simulator, neural policy/search
training loop, gameplay dataset pipeline, live emulator bridge, and verification
harness in one repo.

The project is organized around one loop:

```
video/gameplay data -> structured trajectories -> simulator + search -> policy training
       ^                                                            |
       |                                                            v
 live emulator bridge <- perception + state shim <- evaluated policies and scripts
```

## What Is Included

| Area | What it provides |
|---|---|
| Simulator | `crsim/` models Clash Royale as an 18x32 real-time arena with elixir, towers, troops, spells, buildings, pathing, targeting, stun, splash, and card/evolution/champion metadata. |
| Search and policy | `mcts/` and `model/` implement Gumbel-MuZero-style search, action encoding, spatial/entity features, transformer-enhanced policy/value networks, and baseline policies. |
| Training | `training/`, `scripts/train*.py`, and `sim_batch.py` provide self-play, replay buffers, imitation warm-start hooks, curriculum, league/PFSP wiring, vectorized simulation, and distributed training entrypoints. |
| Dataset pipeline | `crpipe/` plus `tools/dataset/` discover public gameplay videos, download clips, sample frames, annotate game state, segment matches, and emit validated `GameRecord` JSON. |
| Live emulator bridge | `apps/emulator_bot/` captures BlueStacks frames, runs a pluggable perception backend, maps detections into simulator-style state, selects actions, and taps the Android emulator through ADB. |
| Verification | `tests/` and `verification/` cover simulator contracts, action-space invariants, encoding, baseline behavior, cross-engine stats, ladder ordering, and golden gameplay interactions. |
| Native engine path | `cr_engine/` contains the Rust-backed engine experiments and gamedata conversion assets for higher-throughput simulation work. |

## Repository Layout

```text
.
├── apps/emulator_bot/       # Live BlueStacks control, perception bridge, strategy scripts
├── config/                  # Training and dataset example configuration
├── cr_engine/               # Rust engine experiments and generated gamedata assets
├── crpipe/                  # Gameplay video -> GameRecord dataset pipeline
├── crsim/                   # Python Clash Royale simulator
├── data/                    # Python data package for training datasets
├── eval/                    # Baselines, evaluator, ladder, replay, tournament tools
├── mcts/                    # Search implementations
├── model/                   # Feature encoders and neural network models
├── realtime/                # Real-time control abstractions
├── samples/game_records/    # Small schema-valid dataset examples
├── scripts/                 # Training, evaluation, play, warm-start, and generation CLIs
├── scroll_bridge/           # External simulator bridge experiments
├── tests/                   # Main regression suite
├── tools/dataset/           # Dataset command-line tools
├── training/                # Self-play, league, curriculum, trainers, replay buffers
└── verification/            # Conformance helpers and reports
```

## Quick Start

```bash
git clone https://github.com/AA-Alghamdi/clash-royale-zero.git
cd clash-royale-zero
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the core checks:

```bash
python -m pytest tests/ -q
python tools/dataset/smoke_test.py
```

Inspect the dataset schema:

```bash
python tools/dataset/run.py schema
```

Train a small development policy:

```bash
python scripts/train.py --mode single --device cuda:0 \
  --n-res-blocks 4 --n-filters 64 --mcts-sims 50 --batch-size 256
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --baseline pro
```

Run the live emulator bridge from the monorepo:

```bash
cd apps/emulator_bot
python -m perception.cli frame.png
python strategy_bot.py 5
python proagent_play.py 1
```

## Current Status

The repository is a consolidated platform, not a packaged champion checkpoint.
The simulator, action representation, model interfaces, search stack, dataset
schema, live bridge, and test harness are present and runnable. The live bridge
can operate a BlueStacks match using scripted policies and record decision logs.

The next technical milestone is training stability: the infrastructure is ready
for larger runs, but the learned policy should be treated as experimental until
reward targets, imitation warm-start data, and simulator-to-live fidelity are
validated against stronger baselines.

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Dataset Pipeline](docs/DATASET_PIPELINE.md)
- [Live Emulator Bridge](docs/LIVE_EMULATOR.md)
- [Training Notes](docs/TRAINING_NOTES.md)
- [Verification](verification/README.md)

## Notes

This project is for research and automation experiments. Use emulator control,
video collection, and external services only where you have the right to do so
and in accordance with applicable terms.
