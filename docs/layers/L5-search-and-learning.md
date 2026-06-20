# L5 — Search & Learning

The top layer: search (MCTS family), the networks, the training loop, and evaluation. It talks
only to the L4 contract, so it is backend-agnostic.

## Why this game is hard for AlphaZero (and how the design copes)

| Property | Go/Chess | Clash Royale | Design response |
|---|---|---|---|
| Turns | alternating | simultaneous, real-time | discretize to 0.5 s ticks; treat each decision point as a simultaneous-move game |
| Action space | ~361 | ~2305 | factored `(card,x,y)`; masking; pointer/conv policy head |
| Observability | perfect | partial (opponent hand/elixir) | IS-MCTS + opponent model |
| Time | none | continuous (elixir, timer) | scalar features + phase channel |

The honest tension: **vanilla AlphaZero assumes turn-based, perfect-info, deterministic
lookahead** — CR is none of those. The repo addresses this with (a) a *family* of search
algorithms, not just one, and (b) opponent modeling.

## Search (`mcts/`)

- `search.py` — vanilla PUCT MCTS (800 sims, c_puct≈2.5, Dirichlet root noise). The baseline.
- `gumbel_search.py` — **Gumbel MuZero** (Gumbel-Top-k sampling without replacement;
  guaranteed policy improvement even at ~16 sims). This is the right default for a 2 Hz game:
  matches AlphaZero quality at a fraction of the search budget.
- `is_mcts.py` — **Information-Set MCTS** for the hidden-information reality (opponent hand
  unknown): search over determinizations / information sets rather than a single known state.
- `muzero_search.py` — learned-dynamics search (plan in latent space; no perfect simulator
  needed at inference).

**Recommendation:** make **Gumbel MuZero the production searcher** and layer KataGo tricks
(playout-cap randomization, policy-target pruning, dynamic cPUCT, global pooling,
auxiliary-target training) for ~order-of-magnitude compute savings. Keep `is_mcts.py` for
hidden-info correctness and `search.py` as a reference/oracle for tests.

## Networks (`model/`)

- `network.py` — ResNet-20, 256 filters, SE blocks, ~30M params. Spatial trunk (Conv 3×3 →
  20× SE-ResBlock) + scalar MLP, concatenated into policy head (→ 2305 logits) and value head
  (→ scalar). The standard AlphaZero two-headed net.
- `transformer_net.py` — AlphaStar-style **entity transformer**: encode the variable-length
  on-field unit list with self-attention; better for unit-heavy states than fixed conv
  channels. Worth A/B-ing against the ResNet trunk.
- `opponent_model.py` — predicts the opponent's policy/hand; feeds IS-MCTS and improves value
  estimates under hidden information.
- `features.py` / `action_space.py` — the L4 encoders (kept here so net and env agree).

## Training (`training/`)

- `self_play.py` / `self_play_v2.py` — trajectory generators; store `(obs, π_visit, z)`.
- `replay_buffer.py` — 500K-game ring buffer (≈20 GB CPU RAM).
- `trainer.py` / `trainer_v2.py` — AdamW, cosine LR, FP16 AMP, batch 2048; loss =
  `CE(policy, π) + MSE(value, z) + L2`. `train_v2.py` targets CUDA (runs on CPU in CI).
- `curriculum.py` — starter decks → random 8-card decks (mirrors the staged 24h plan).
- `imitation.py` — warm-start from human replays (Supercell API battlelogs, Kaggle 37.9M
  matches, HF TV-Royale frames) before self-play, à la AlphaStar's imitation→RL transition.
- `distributed.py` — Ray/DDP multi-GPU (self-play workers + trainer GPUs).

## Self-play league (the "Zero" loop, done right)

Maintain a **pool of past checkpoints** and sample opponents (prioritize recent + hard) rather
than always playing the latest-vs-latest. This is the AlphaStar league idea and is the main
defense against **strategy collapse / cycling** (rock-paper-scissors deck dynamics make naive
self-play unstable). Track **Elo** (`eval/evaluator.py`) for promotion decisions and
`eval/tournament.py` for round-robins; `eval/replay.py` for inspecting games.

## Eval (`eval/`)
- `evaluator.py` — head-to-head + Elo vs random/baseline/checkpoints.
- `tournament.py` — multi-agent round-robin for league standings.
- `replay.py` — record/replay games for debugging and regression.
- Gate every promotion on Elo improvement with enough games for significance; watch for
  non-transitive cycles (A>B>C>A) — a single linear Elo can hide them, so keep the round-robin.

## Compute reality
- CI is CPU-only; keep a tiny-config smoke run (few res blocks, few sims, short game) green in
  CI, and gate heavy runs behind GPU.
- For real-time inference (L6 deployment), prefer a **search-light or search-free** policy
  (Gumbel few-sim, or pure learned policy à la OpenAI Five) — 800-sim MCTS won't hit a 2 Hz
  budget on a live client.
