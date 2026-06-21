# Training Notes

The repository contains the pieces required for full training, but stable strong
play depends on careful target construction, reward shaping, data quality, and
simulator fidelity. Treat training as an active engineering workflow rather than
a one-command solved result.

## What Is Ready

- Simulator contracts and action-space tests.
- Scripted baselines for regression and live control.
- Neural policy/value models and feature encoders.
- Self-play, replay buffer, trainer, distributed worker, and league scaffolding.
- Imitation warm-start hooks from structured `GameRecord` data.
- Evaluation tools for baseline games, checkpoint matches, replay, and ladders.

## Recommended Sequence

1. Run the mechanics and encoding tests.
2. Train on tiny deterministic fixtures until loss, value targets, and action
   entropy behave as expected.
3. Warm-start from high-confidence dataset records and live decision logs.
4. Run short self-play cycles against frozen scripted baselines.
5. Add league/PFSP sampling only after small runs improve monotonically.
6. Scale search budget, batch size, and worker count once the loop is stable.

## Commands

Small local trainer:

```bash
python scripts/train.py --mode single --device cuda:0 \
  --n-res-blocks 4 --n-filters 64 --mcts-sims 50 --batch-size 256
```

Distributed trainer:

```bash
python scripts/train_distributed.py --auto-ray
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --baseline pro
```

Generate or inspect imitation fixtures:

```bash
python scripts/warmstart.py --help
python run_bc_eval.py --help
```

## Stability Checklist

- Value targets are aligned with the final outcome and perspective.
- Reward shaping is potential-based or otherwise audited for exploitable loops.
- Action masks match placement legality, hand slots, and ability availability.
- Search targets are built from legal priors only.
- Dataset records have enough confidence to train on placements and timing.
- Simulator outcomes agree with baseline expectations before scaling compute.
