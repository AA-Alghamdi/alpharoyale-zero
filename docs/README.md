# ClashRoyale-Zero — Design Docs

Architecture synthesis, per-layer deep dives, and the current critical-path task spec.
Grounded in the implementation on the feature branches (`feature-v2`, `devin/*`); `main` is a
stub. RVAs/offsets are the working hypothesis for `libg.so` from CR v1.3.2.

- [`architecture.md`](architecture.md) — whole-system architecture, the three-backend design,
  the unifying environment contract, and the critical path.
- **Layer deep dives** (`layers/`):
  - [`L1-game-data.md`](layers/L1-game-data.md) — authentic CSV/JSON data, level scaling, the
    modern-card converter, champions/evolutions.
  - [`L2-simulation-backends.md`](layers/L2-simulation-backends.md) — `crsim` (Python),
    `cr_engine` (Rust), `scroll_bridge` (real engine), and the hybrid strategy.
  - [`L3-scroll-bridge-and-re.md`](layers/L3-scroll-bridge-and-re.md) — Scroll JSON/TCP
    protocol, the reverse-engineering findings, and the full offset table.
  - [`L4-environment-state-action.md`](layers/L4-environment-state-action.md) — observation
    (44+116), action space (2305), masking, perspective flip, reward.
  - [`L5-search-and-learning.md`](layers/L5-search-and-learning.md) — MCTS family, networks,
    training/league, evaluation.
- [`data-embeddings-rl.md`](data-embeddings-rl.md) — how the game is mapped to tensors, the
  embedding design (stat-based card embeddings → unit → deck → state), the RL/search setup, and
  the data flywheel (which data source plays which role).
- [`path-to-world-champion.md`](path-to-world-champion.md) — the staged program to build the
  best player and scale toward beating the world champion: method commitments, phased plan with
  go/no-go gates, how to measure progress (exploitability), compute staging, and honest-stop
  criteria.
- [`p0-engine-fix-spec.md`](p0-engine-fix-spec.md) — **the current P0 blocker**: a
  self-contained task spec to make Scroll boot a real battle (arena + towers) and step 200k
  ticks.

**TL;DR.** An AlphaZero/MuZero self-play agent with three interchangeable sim backends behind
one env contract (Python `crsim` for dev, Rust `cr_engine` for throughput, `scroll_bridge` →
real `libg.so` for 100% fidelity). The single blocker to fidelity is P0: `start_mission` never
constructs the arena (`[battle+0xb0]`), so towers never spawn and the next tick null-derefs.
Fix by restoring the engine's own `load_battle_state` call rather than hand-building objects.
