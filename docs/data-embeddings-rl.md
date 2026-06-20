# Game Mapping, Embeddings, RL & the Data Flywheel

How I'd represent Clash Royale to a network, what gets embedded, how learning is structured,
and how every data source (self-play, human replays, card data, the real game) feeds in. This
is the "model + data" companion to `architecture.md` (which covers the systems/backends).

---

## 1. Mapping the game itself

CR state factors into four parts; encode each with the representation it deserves, then fuse.

| Part | Contents | Encoding |
|------|----------|----------|
| **Board (spatial)** | unit/building positions, towers, projectiles, area effects | `C×18×32` tensor (CNN) |
| **Entities (set)** | variable-length list of game objects, each with rich features | list of vectors (Transformer) |
| **Scalars** | own elixir, clock, phase (1×/2×/3×), crowns, tower HP/alive, deck cycle | vector → MLP |
| **Hidden** | opponent hand, opponent elixir, opponent deck, next card | inferred by opponent model |

Use **both** the spatial tensor *and* the entity set (AlphaStar/OpenAI Five do this — they're
complementary): the grid is right for placement/locality; the entity set is right for unit
interactions and variable counts. Fuse them with a **scatter connection** (write each entity's
embedding back into its grid cell) so the conv trunk sees per-unit semantics, not just density.

```
Board tensor (44×18×32) ─► SpatialEncoder (Conv + SE-ResBlocks) ─┐
Entity list (N×feat) ─► EntityEncoder (Transformer) ─► scatter ──┤─► fuse ─► Core (LSTM/Transformer)
Scalars (116) ─► ScalarEncoder (MLP) ───────────────────────────┘            │ carries battle context
Hand (4×card_emb)+next, inferred-opp-deck-emb ──────────────────────────────►┘
```

The **Core is recurrent** on purpose: elixir leaked, the push that's building, what the
opponent just cycled — CR rewards memory across ticks. OpenAI Five used a 4096-d LSTM; an
LSTM or small Transformer-XL core over decision steps is the right call here.

---

## 2. Embeddings (the part that makes or breaks generalization)

### (a) Card embeddings — make them **stat-derived**, not just a learned id table
Each card → vector. Two ways to build it; **use the hybrid**:

```
card_emb(c) = MLP_stats( features(c) )   ⊕   id_table[c]
features(c) = [cost, hp, dmg, dps, hit_speed, range, sight, speed, deploy,
               target_mode, flying, splash, spawn_count, rarity, mechanic_flags...]
```

Why the stat-MLP term matters: it means a **brand-new card the network has never seen still
gets a sensible embedding from its stats** (a 3-elixir fast melee troop lands near other fast
melee troops). This is what lets the modern-card converter (L1) add cards and have the agent
play them *near-zero-shot* — a pure learned-id table cannot generalize to unseen ids. Champions
/ evolutions add a few extra flag dimensions (ability cost, evo cycle) that are 0 for classic
cards.

Card embeddings are reused everywhere: hand encoding, deck encoding, the card action head, and
as the identity component of on-field unit embeddings.

### (b) Unit / entity embeddings — grounded in card identity
```
entity_emb = card_emb(type) ⊕ [hp_frac, x, y, owner, state(charge/attack/stun),
                                target_id, level, shield, remaining_lifetime]
```
So "Musketeer on the field" is literally derived from the "Musketeer card", and the net
transfers knowledge between hand and board. Position uses a learned/Fourier spatial embedding.

### (c) Deck / archetype embeddings — for matchup & opponent reasoning
`deck_emb = attention_pool({card_emb(c) : c in deck})`. Used for: your-deck conditioning;
**inferred opponent deck** (built incrementally from revealed cards, completed by the opponent
model); and offline meta analysis from the 37.9M-match dataset (deck → winrate prior).

### (d) State embedding = the Core's hidden vector
The fused, recurrent summary of the battle so far. This is the vector MCTS/MuZero plans over
and the value head reads.

> Net effect: identity flows **card → unit → deck → state**, all in one shared embedding
> space, which is what gives sample efficiency and new-card generalization.

---

## 3. RL setup

### Action head — factored & autoregressive (handles the 2305 space cleanly)
```
1) action_type ∈ {WAIT, PLAY}
2) if PLAY: card  ← softmax over the 4 hand card embeddings        (mask: not enough elixir)
3)          position ← pointer/deconv over 18×32, conditioned on the chosen card
                                                                    (mask: illegal placement)
```
Autoregressive (AlphaStar-style) so *where* depends on *what* (you place a Giant at the back,
a Fireball on a cluster). Masking before softmax, never penalties.

### Learner — AlphaZero/MuZero with the search the game can afford
- **Self-play** produces `(state, search_policy π, outcome z)`.
- **Gumbel MuZero** as the production searcher: near-AlphaZero strength at ~16 sims vs 800 —
  essential for a 2 Hz real-time budget — and it learns a **dynamics model**, so it can plan in
  latent space without a perfect simulator at inference (good, since Scroll snapshot/restore is
  expensive).
- **IS-MCTS + opponent model** for the **imperfect information**: search over determinizations
  of the opponent's hidden hand/deck, weighted by the opponent model's belief.
- **Loss** = `CE(policy, π) + MSE(value, z) + MuZero reward/dynamics consistency + aux heads`.
  Auxiliary heads (KataGo-style, free signal, big stability win): predict **opponent's next
  card**, **tower-HP deltas**, **elixir leak**, **game length**.

### Population / league self-play (not naive latest-vs-latest)
CR decks are rock-paper-scissors → naive self-play cycles and collapses. Maintain a
**population**: main agents + *exploiters* (trained to beat the current main) + frozen past
selves; sample opponents with priority. Evaluate with **Elo + round-robin** (watch for
non-transitive A>B>C>A cycles a single Elo hides). Curriculum: starter deck → fixed decks →
random decks → full pool (expanded by the converter).

---

## 4. The data flywheel — how everything feeds in

Different data plays different roles; they are not interchangeable:

```
                 ┌──────────────── card/stat data (CSV/JSON) ────────────────┐
                 │ feeds: stat-based card embeddings  +  the simulators       │
                 ▼                                                            ▼
 human replays ──► IMITATION / OFFLINE RL warm-start ──► POLICY ──► SELF-PLAY ──► replay buffer ──► TRAIN ──┐
 (frames, decks)          (Decision Transformer)            ▲     (crsim / cr_engine / Scroll)              │
                                                            └──────────────── new checkpoint ◄─────────────┘
 Scroll (real engine) ──► ground-truth traces ──► CALIBRATE crsim/cr_engine (shrink sim-to-real)
 real game via CV ──────► EVAL + slow online fine-tune
```

| Source | Granularity | Role | Plugs into |
|--------|-------------|------|------------|
| **Self-play** (crsim/cr_engine) | full (state,action,outcome) | primary online RL, bulk volume | replay buffer → trainer |
| **Self-play** (Scroll) | full, 100% fidelity | fidelity-critical training + final eval (post-P0) | replay buffer + calibration |
| **Supercell API battlelogs** | deck + outcome | deck/matchup priors, opponent-deck embeddings, value prior | embeddings, opponent model |
| **Kaggle 37.9M matches** | deck + outcome | meta/matchup table, value warm-start | offline value prior |
| **HF TV-Royale frames (1.88TB), wty-yy** | frame-level (≈10fps) | extract (state,action) via CV → **behavioral cloning / offline RL** warm-start | imitation → policy |
| **Card/stat CSV-JSON** | static | authentic physics + **stat card embeddings** (new-card generalization) | sims + embeddings |
| **Real game (CV/perception)** | frame-level, slow (~1–2 FPS) | deployment + eval + small online fine-tune | eval loop, not bulk training |

Two flywheel principles:
1. **Warm-start, then self-play.** Bootstrap embeddings + policy from human data (imitation /
   offline RL on replays + deck priors), then let self-play exceed the data ceiling — exactly
   AlphaStar's imitation→RL transition, and what beat the 8000-pt AI in KataCR.
2. **Fidelity ladder.** Train cheaply on crsim/cr_engine; use Scroll's ground-truth traces to
   **calibrate** the cheap sims (replay identical actions, diff tower-HP/outcome/checksum);
   final-eval on Scroll / the real client. The calibration loop is what keeps a
   crsim-trained policy from breaking on the real engine.

---

## 5. How a new card (or new game version) enters the system

This ties L1's converter to the model and is the cleanest test of the design:
1. Converter emits the card's stat rows (+ referential closure) into the sims' data files.
2. Stat-based **card embedding** is produced from those stats → the policy can represent it
   immediately (near-zero-shot), no retrain required to *use* it.
3. Self-play with the new card in the pool **specializes** the embedding and learns its
   interactions; the league prevents the new card from destabilizing existing strategies.
4. Champions/evolutions: simulatable on crsim/cr_engine today (`heroes.py`/`evolutions.py`);
   on Scroll they're stubbed until the engine is patched (out of scope for P0).

So "different games/data feeding in" reduces to: **data shapes embeddings + sims; embeddings +
sims shape self-play; self-play shapes the policy; the policy is validated back against the
real engine.** Each loop tightens the others.
