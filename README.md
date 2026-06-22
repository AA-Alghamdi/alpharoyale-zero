# AlphaRoyale-Zero

AlphaRoyale-Zero is a full-stack research system for building, evaluating, and
deploying Clash Royale agents. It combines a mechanics simulator, search and
policy models, a gameplay dataset pipeline, evaluation harnesses, and a live
BlueStacks control bridge into one coherent platform.

This repository is the foundation of a private project developing a
world-competitive Clash Royale player. Its purpose is to make that ambition
credible at the infrastructure level: real data, tested mechanics, search,
reinforcement learning, evaluation, and live execution all share the same
interfaces.

The central result is an end-to-end control loop: public gameplay can be turned
into validated match records, those records can seed training and simulator
checks, policies can be evaluated through the same action space, and selected
actions can be sent to a running Android emulator through calibrated taps.

```text
gameplay video -> validated GameRecord -> simulator/search -> policy training
       ^                                                     |
       |                                                     v
 live emulator bridge <- perception/state shim <- evaluated policy action
```

## Executive Summary

Most Clash Royale automation projects solve one narrow layer: a tap script, a
screen detector, a toy environment, or a training loop. This repo is built around
the harder problem: keeping every layer connected by shared contracts.

The same concepts appear everywhere:

- `GameRecord` for match data and annotated timelines.
- `CRGame` state for simulator mechanics and evaluation.
- A legal action mask over hand slots, board tiles, champion ability, and wait.
- Policy interfaces that can be backed by scripts, search, or learned models.
- Logs and tests that make failures traceable to data, mechanics, encoding,
  search, training, or live control.

That unification is the point of the project. It makes the system useful before
there is a champion checkpoint, because each subsystem can be improved and
verified without rewriting the rest of the stack.

## Reinforcement Learning Lineage

The learning stack deliberately echoes the AlphaGo, AlphaZero, and MuZero family
of systems at the architectural level:

- Search-guided policy improvement through MCTS, Gumbel search, and
  MuZero-style search variants.
- Policy/value networks that consume spatial arena planes, scalar game context,
  and entity tokens.
- Self-play generation, replay buffers, value targets, and action-distribution
  targets for iterative improvement.
- Imitation warm-starts from structured gameplay records before expensive
  self-play scale-up.
- League/PFSP opponent sampling, curriculum hooks, and baseline ladders for
  avoiding brittle progress against a single opponent.

The analogy is methodological, not a performance claim. The project imports the
playbook of search plus learned evaluation plus self-play, then adapts it to
Clash Royale's real-time, partially observed, spatial action setting.

## End-To-End Result

The project now demonstrates a complete path from raw gameplay evidence to live
emulator execution:

1. Discover, download, sample, and annotate Clash Royale gameplay videos.
2. Validate segmented matches as schema-checked `GameRecord` JSON.
3. Convert game state into spatial, scalar, and entity-token model features.
4. Search over the same legal action space used by training and evaluation.
5. Run scripted or learned policies against the simulator and baseline ladder.
6. Translate selected policy actions into BlueStacks coordinates and ADB input.

This is the project's main milestone: the repository is not just a collection of
experiments. It is a single research apparatus where dataset construction,
simulation, training, evaluation, and live deployment speak the same language.

## Results Snapshot

Latest validation of the consolidated repo:

| Check | Result |
|---|---:|
| Python modules tracked | 125 |
| Test modules tracked | 21 |
| Core test suite | 227 passed, 2 skipped |
| Lint | `ruff check .` passed |
| Dataset smoke test | passed, generated 2 validated games, 9 timesteps, and 4 extracted actions |
| Sample record | `samples/game_records/realyt1_sample.json` validates as `GameRecord` |
| GitHub Actions | passing on `main` |

Implemented capabilities:

| Area | Result |
|---|---|
| Simulator | 18x32 arena, elixir phases, towers, troops, spells, buildings, pathing, targeting, splash, stun, champion/evolution metadata, and legal placement masks. |
| Search | Standard, Gumbel, and MuZero-style search paths sharing the simulator action interface. |
| Models | Spatial planes, scalar features, entity tokens, transformer-enhanced policy/value networks, and action masking. |
| RL and training | Self-play, replay buffers, policy/value targets, imitation warm-start hooks, curriculum, league/PFSP opponent sampling, vectorized simulation, and distributed entrypoints. |
| Dataset | Video discovery, download, frame sampling, frame annotation, game segmentation, schema validation, and batch processing tools. |
| Live control | BlueStacks screenshots, perception shim, simulator-style state mapping, policy selection, ADB tap calibration, spell guards, and decision logs. |
| Verification | Tests for mechanics, feature encoding, model outputs, search legality, self-play, ladders, imitation fixtures, dataset schema, and cross-engine behavior. |

## Methodology

The project follows a systems-first methodology:

1. **Define shared contracts.** Match records, simulator state, model features,
   and live actions are represented explicitly so every layer can be audited.
2. **Make the simulator testable before scaling training.** Mechanics are split
   into small modules and covered by focused tests for interactions, timing,
   targeting, action legality, and feature encoding.
3. **Use search and scripted baselines as control instruments.** They provide
   repeatable opponents, sanity checks, and policy targets before expensive
   training runs.
4. **Ground training data in real gameplay timelines.** The dataset pipeline
   converts videos and recordings into structured records with confidence,
   provenance, decks, tower health, elixir, actions, and raw annotations.
5. **Keep live deployment behind the same policy interface.** The emulator
   bridge consumes simulator-style state and emits the same action objects used
   by evaluation.
6. **Validate each boundary independently.** Dataset smoke tests, schema checks,
   encoding tests, model tests, search legality tests, simulator golden tests,
   and CI prevent one layer from hiding another layer's failure.

## Repository Layout

```text
.
├── apps/emulator_bot/       # BlueStacks control, perception bridge, strategy scripts
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
├── scripts/                 # Training, evaluation, play, warm-start, generation CLIs
├── scroll_bridge/           # External simulator bridge experiments
├── tests/                   # Regression suite
├── tools/dataset/           # Dataset command-line tools
├── training/                # Self-play, league, curriculum, trainers, replay buffers
└── verification/            # Conformance helpers and reports
```

## Quick Start

```bash
git clone https://github.com/AA-Alghamdi/alpharoyale-zero.git
cd alpharoyale-zero
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

## Current Boundary

The repo is a consolidated platform, not a packaged champion model. The
simulator, action representation, model interfaces, search stack, dataset
schema, live bridge, and verification harness are present and runnable. The live
bridge can operate BlueStacks through scripted policies and produce decision
logs.

The next milestone is a stable learned policy suitable for the private
world-competitive player track: high-confidence imitation data, reward target
audits, larger self-play runs, and simulator-to-live fidelity checks against
stronger baselines.

## Documentation

- [Project Report](docs/PROJECT_REPORT.md)
- [Architecture](ARCHITECTURE.md)
- [Dataset Pipeline](docs/DATASET_PIPELINE.md)
- [Live Emulator Bridge](docs/LIVE_EMULATOR.md)
- [Training Notes](docs/TRAINING_NOTES.md)
- [Verification](verification/README.md)

## Usage Notes

This project is for research and automation experiments. Use emulator control,
video collection, and external services only where you have the right to do so
and in accordance with applicable terms.
