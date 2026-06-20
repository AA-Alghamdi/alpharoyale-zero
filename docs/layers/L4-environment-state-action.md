# L4 — Environment: State, Action, Reward

The backend-agnostic seam. `features.py` encodes state→tensor, `action_space.py` defines and
masks actions, and the env (`crsim.game.CRGame` / `ScrollBattleEnv` / Rust adapter) exposes
`reset/step/observe`. Everything above (MCTS, nets, training) talks only to this layer.

## Observation: 44 spatial channels (18×32) + 116 scalars

**Spatial (C × 18 × 32), C = 44:**

| Channels | Description |
|---|---|
| 0–19 | friendly unit density per card type (Gaussian splat σ=0.5 tiles, value = hp/max_hp) |
| 20–39 | enemy unit density per card type |
| 40 | friendly tower HP, normalized |
| 41 | enemy tower HP, normalized |
| 42 | valid placement mask for current player |
| 43 | static map features (river, bridges) |

**Scalars (116):** elixir (1) + regen-rate/phase (1) + hand one-hot ×4 (80) + next card (20) +
time remaining (1) + tower alive flags ×6 (6) + tower HP ×6 (6) + score/tower-diff (1).

**Perspective normalization:** state is always encoded from the *current player's* view — the
board is flipped vertically for player 1 so the network always sees its own side at the bottom.
This halves the effective state space and is essential for self-play symmetry. **It must be
implemented identically across all backends** or a crsim-trained net won't read Scroll states.

> Note: `RESEARCH_DEEP_DIVE.md` also catalogs external envs that use a raw `128×128×3` image
> observation. Our structured 44+116 encoding is far more sample-efficient and is the right
> default; reserve the image encoding for the realtime/CV deployment path (L6).

## Action: 2305 discrete (+ masking)

```
Action = (card_index ∈ {0..3}, x ∈ {0..17}, y ∈ {0..31})  or  WAIT
Total  = 4 × 18 × 32 + 1 = 2305
```

`WAIT` (accumulate elixir) is a first-class action, not the absence of one — the policy must be
able to choose to hold. Illegal actions are masked **before** softmax (`logits[~mask] = -1e9`):
- not enough elixir for the card → mask all `(card, *, *)`
- position outside the player's placement zone, in the river, or on a tower → mask that cell

Masking (not penalizing) is what keeps the policy from wasting probability mass on illegal
moves and is critical for MCTS prior quality.

## Tick vs decision cadence

The engine advances every tick (2 Hz); the agent decides every `decision_interval_ticks`.
`env.step()` advances `ticks_per_step` engine ticks per agent action. Decoupling these is both
faithful (you can't meaningfully act every 0.5 s on every front) and a large throughput win —
fewer NN/MCTS evaluations per game. A full game is ~7200 logic ticks; at a coarse decision
interval the agent makes far fewer decisions than that.

## Reward design

- **Terminal (AlphaZero-pure):** z = +1 win / −1 loss / 0 draw, by crowns then tower HP. This
  is the value-head target and the cleanest signal.
- **Shaped (bootstrap):** crown deltas, tower-HP deltas, elixir-efficiency (à la OpenAI Five's
  "surgically crafted" rewards). Use shaping early to get gradient, then **anneal toward the
  sparse terminal signal** so the agent optimizes the real objective, not a proxy.
- Keep shaping zero-sum/symmetric so self-play stays a proper game.

## Recommended contract additions

- **`snapshot()` / `restore()`** — cheap on crsim/cr_engine (clone state), harder on Scroll
  (would need an engine save/restore offset). Unlocks proper MCTS rollouts and bit-exact crash
  reproduction. Without it, MCTS on the real engine must re-simulate from root each time.
- **`legal_action_mask()`** — expose the 2305-bool mask through the contract so MCTS and the
  policy share one definition of legality.

## Determinism
crsim/cr_engine and the real engine are deterministic given seed + input sequence. Seed every
episode and log the action stream; any game (and any crash) can then be replayed bit-for-bit —
which is what makes `snapshot/restore`, the L2 calibration harness, and debugging tractable.
