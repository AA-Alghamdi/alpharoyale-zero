# Architecture

Clash Royale Zero is designed as a closed-loop platform: collect structured
gameplay, improve policies in simulation, evaluate them against stable baselines,
then deploy the same decision interface to a live emulator.

The repository serves as the foundation for a private project developing a
world-competitive Clash Royale player. The architecture is therefore optimized
for measurable policy improvement, not just isolated automation demos.

## Design Thesis

The architecture is built around one constraint: every subsystem should share the
same strategic vocabulary. Video records, simulator states, model features,
search nodes, training targets, evaluator games, and live emulator taps all map
back to the same ideas of time, elixir, towers, entities, cards, legal actions,
and outcomes.

That design keeps the project from becoming separate demos. Dataset records can
be audited against simulator states. Search can produce targets for training.
Evaluation can reuse the same legal masks that protect live control. The live
bridge becomes a deployment surface for policies instead of a separate bot.

## System Map

```text
                        +--------------------------+
                        |      GameRecord data     |
                        |  videos, logs, rollouts  |
                        +------------+-------------+
                                     |
                                     v
+--------------------+      +--------+---------+      +----------------------+
| dataset pipeline   | ---> | training stack   | ---> | checkpoints/policies |
| crpipe + tools     |      | self-play/BC/RL  |      | neural + scripted    |
+--------------------+      +--------+---------+      +----------+-----------+
                                     ^                           |
                                     |                           v
                            +--------+---------+      +----------+-----------+
                            | CRSim simulator  | <--> | search/evaluation   |
                            | Python + Rust    |      | MCTS, ladder, replay|
                            +------------------+      +----------+-----------+
                                                               |
                                                               v
                                                    +----------+-----------+
                                                    | live emulator bridge |
                                                    | perception + ADB     |
                                                    +----------------------+
```

## Core Simulator

`crsim/` is the main environment implementation. It provides:

- 18x32 tile arena with mirrored player perspectives.
- 20 Hz timing model, elixir phases, overtime, and sudden death.
- Tower, troop, spell, building, champion, and evolution metadata.
- Target acquisition, pathing, collision-adjacent movement, splash, stun,
  retargeting, death cleanup, and win-condition checks.
- Legal-action masks for four-card hand placement, champion ability, and wait.

The simulator is intentionally transparent and testable. Card behavior is split
across small modules (`cards.py`, `entities.py`, `pathfinding.py`, `game.py`) so
individual mechanics can be validated without running a full training loop.

## State and Action Representation

The model input combines three views:

- Spatial planes for arena occupancy, towers, hazards, and ownership.
- Scalar features for elixir, phase, time, tower health, hand, and deck context.
- Entity tokens for units/buildings carrying identity, position, HP, movement,
  targeting, and ownership.

The main action space is:

```text
4 hand slots x 18 x 32 placement tiles + ability/wait actions
```

All training, search, emulator, and evaluation paths use this shared action
encoding so policies can move between simulation and live control with minimal
translation.

## Search and Policy

The learning side has two policy families:

- Scripted baselines in `eval/baseline_agents.py` for regression, live play, and
  stable evaluation opponents.
- Neural policies in `model/`, including spatial encoders and entity-token
  models with policy/value heads.

`mcts/` contains standard, Gumbel, and MuZero-style search variants. The current
development path favors fast Gumbel-style search for self-play and larger search
budgets for evaluation.

This follows the AlphaGo/AlphaZero/MuZero pattern in spirit: search improves
action selection, policy/value networks learn from search and outcomes, and
self-play turns the current agent into its own training curriculum. Clash Royale
adds real-time execution, partial observation, hidden deck-cycle information, and
spatial placement constraints, so the architecture keeps search, encoding,
evaluation, and live taps behind a shared action interface.

## Training Stack

`training/` contains:

- Self-play workers.
- Replay buffers.
- Policy/value target construction.
- Imitation warm-start scaffolding.
- Curriculum and domain-randomization hooks.
- League and PFSP-style opponent sampling.
- Distributed orchestration through Ray.

The stack is designed to scale from single-GPU experiments to multi-GPU workers,
while keeping smaller CPU or local smoke tests available for mechanics and data
format validation.

## Dataset Pipeline

`crpipe/` converts gameplay video into structured match records:

```text
discover -> download -> frame sample -> annotate -> segment -> GameRecord
```

The output schema captures provenance, decks, tower health, elixir, actions,
units, confidence, and frame-level raw annotations. These records are suitable
for imitation learning, policy analysis, and simulator-fidelity checks.

## Live Emulator Bridge

`apps/emulator_bot/` runs the same decision interface against BlueStacks:

```text
screenshot -> perception backend -> simulator-style state -> policy -> ADB tap
```

The bridge includes state logging, match/result handling, tap calibration, spell
guards, and robust screenshot retries for long unattended runs.

## Verification

The repo keeps tests close to the contracts that matter:

- Legal action encoding and decoding.
- Simulator mechanics and golden interactions.
- Feature encoding shapes and invariants.
- Baseline-policy behavior.
- Ladder/evaluator ordering.
- Optional cross-engine conformance checks.

The goal is to make policy training failures diagnosable: mechanics, encoding,
data, search, and trainer behavior can be tested independently before spending
large compute on a full run.

## Methodology In Practice

The system is developed in layers:

1. Lock down mechanics and action legality with small deterministic tests.
2. Encode simulator state into stable model features and masks.
3. Use scripted baselines and search to create repeatable evaluation signals.
4. Convert gameplay into `GameRecord` examples with confidence and provenance.
5. Warm-start and evaluate policies before scaling self-play.
6. Deploy only through the shared action interface used by the simulator.

This keeps the research loop observable. A bad match result can be decomposed
into data quality, simulator fidelity, feature representation, action masking,
search quality, policy training, or live input translation.
