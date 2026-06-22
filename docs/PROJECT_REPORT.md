# Project Report

## Thesis

Clash Royale is difficult to automate well because the hard parts are coupled:
partial observation, real-time timing, imperfect screen perception, large
placement spaces, deck cycles, elixir economics, and long-horizon tactical
payoffs. A strong project cannot treat those as separate demos.

Clash Royale Zero approaches the problem as a full system. The repo builds one
shared control interface and then connects every major layer to it: gameplay
records, simulator state, action encoding, search, training, evaluation, and
live emulator execution.

The repository is the foundation of a private project developing a
world-competitive Clash Royale player. It is intentionally written as an
infrastructure base rather than a one-off bot: every subsystem exists to support
measurable policy improvement and eventual live deployment.

## Research Lineage

The project borrows the central pattern behind AlphaGo, AlphaZero, and MuZero:
combine search, learned policy/value functions, self-play, and a rigorous
evaluation loop. Clash Royale changes the shape of the problem. Instead of a
turn-based board with perfect information, the agent faces real-time timing,
partial observation, elixir management, hidden deck cycle information, and a
large spatial placement/action space.

The repo adapts that lineage through:

- MCTS, Gumbel search, and MuZero-style search variants.
- Policy/value models over spatial planes, scalar features, and entity tokens.
- Self-play workers, replay buffers, value targets, and policy targets.
- Imitation warm-starts from structured gameplay timelines.
- Curriculum hooks, domain randomization, and league/PFSP opponent sampling.
- Evaluation ladders and scripted baselines that provide stable measurements
  before live deployment.

The reference to AlphaGo-style systems is methodological. It describes the
training architecture this repo enables, not a claim that the current checkpoint
has reached world-class play.

## The Milestone

The consolidated repo proves a closed loop:

```text
raw gameplay -> structured match data -> simulator/search -> evaluated policy
     ^                                                       |
     |                                                       v
decision logs <- live emulator control <- calibrated action translation
```

The important achievement is that each stage produces artifacts the next stage
can consume. A video-derived game record can be validated, inspected, and used as
training input. A simulator state can be encoded for a model or searched by MCTS.
The selected action can be checked against the same legality mask used in
training. The live bridge can translate that action into a calibrated emulator
tap and log the result.

That is the project's decisive result: it turns a collection of research ideas
into one testable apparatus.

## Methodology

### 1. Build a Shared Vocabulary

The system starts with explicit objects:

- `GameRecord` for external match timelines.
- Simulator game state for mechanics and evaluation.
- Model features split into spatial planes, scalar context, and entity tokens.
- A single action space for hand slots, board tiles, champion ability, and wait.

This makes boundaries visible. When something fails, the failure can be traced to
data extraction, mechanics, encoding, search, training, or live control.

### 2. Prove Mechanics Before Training Scale

The simulator is intentionally decomposed. Card metadata, entity state, actions,
pathing, hidden stats, game timing, targeting, and bridge behavior live in
separate modules. Tests cover small, high-value contracts such as legal actions,
elixir phases, tower health, targeting, encoding shapes, feature flips, and
baseline interactions.

### 3. Use Baselines as Instruments

Scripted policies are not treated as throwaway opponents. They are used as:

- Regression fixtures for mechanics.
- Sanity checks for ladder ordering.
- Stable opponents for policy evaluation.
- Live-control fallbacks while learned checkpoints are experimental.

This keeps progress measurable before the training loop is strong.

### 4. Turn Gameplay Into Structured Data

The dataset pipeline follows a reproducible flow:

```text
discover -> download -> sample frames -> annotate frames -> segment games -> validate
```

Every record stores confidence, source metadata, decks, tower health, elixir,
actions, raw frame annotations, and final result fields. Failed or uncertain
frames remain auditable instead of silently disappearing.

### 5. Keep Search, Training, and Deployment Aligned

Search, neural models, scripted policies, and the live emulator bridge all use
the same action concepts. This prevents the classic split where training learns
one interface, evaluation uses another, and deployment needs a third translation
layer.

For the private competitive-player track, this alignment is what makes
reinforcement learning practical: self-play actions, search-improved targets,
baseline evaluations, and live emulator taps remain comparable.

### 6. Validate the Boundaries

The repo validates behavior at several levels:

- Unit tests for mechanics and state encoding.
- Model and action-mask tests.
- Search legality tests.
- Self-play and training pipeline smoke tests.
- Dataset schema and smoke tests.
- Cross-engine conformance checks.
- Live bridge logging and tap calibration.

## Results

| Area | Current Result |
|---|---|
| Consolidation | One repo containing simulator, training, dataset, evaluation, live bridge, Rust engine experiments, docs, and CI. |
| Test coverage | 227 tests passed locally with 2 skipped optional cases. |
| Lint | `ruff check .` passed. |
| Dataset smoke | Offline smoke test passed and produced 2 validated games, 9 timesteps, and 4 extracted actions. |
| Dataset schema | `samples/game_records/realyt1_sample.json` validates against `crpipe.schema.GameRecord`. |
| CI | GitHub Actions passes lint, tests, and dataset smoke on `main`. |
| Live bridge | BlueStacks control path is implemented with screenshot capture, perception shim, action selection, calibrated ADB taps, spell guards, and run logs. |
| RL readiness | Self-play, replay buffers, policy/value targets, imitation warm-start hooks, Gumbel/MuZero-style search, curriculum, league sampling, and distributed entrypoints are present. |

## What Makes The Project Substantial

The repo is substantial because it joins three usually separate projects:

1. A research simulator with legal actions, card mechanics, feature encoders,
   search, reinforcement learning, and evaluation.
2. A data engine that converts real gameplay into structured, validated records.
3. A live emulator bridge that uses the same policy interface against a running
   Clash Royale client.

The value is not a single script. The value is the common interface across all
three.

## Honest Boundary

The project does not claim a finished champion policy. The current system is the
foundation required to train, audit, and deploy one inside the private
world-competitive player effort. The learned-policy path still needs larger
high-confidence datasets, reward target audits, simulator fidelity measurements,
and long self-play runs against stronger baselines.

That boundary is intentional. The repo is structured so the next work is
measurable rather than theatrical.

## Next Milestone

The next serious milestone is a reproducible policy benchmark:

1. Build a curated high-confidence imitation dataset.
2. Train a small policy until it beats scripted baselines in simulation.
3. Compare simulator decisions to live decision logs on matched states.
4. Run a fixed live-emulator evaluation protocol.
5. Publish a table of win rate, crown differential, elixir leakage, invalid
   action rate, and action latency.
