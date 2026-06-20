# L3 — Scroll Bridge & Reverse Engineering

This is the 100%-fidelity backend and the home of the current P0 blocker. It connects Python
to **Scroll**, a Rust server that reuses Supercell's compiled `libg.so` (CR v1.3.2) instead of
reimplementing physics — so it *is* the real game.

## Topology

```
┌──────────────────┐      JSON over TCP        ┌───────────────────────────┐
│ Python RL agent  │  ◄────  port 9340  ────►  │ Scroll headless server     │
│ scroll_bridge/   │                           │ (Rust, libserver/headless) │
│   client.py      │                           │        ▼                   │
│   env.py         │                           │   libg.so (real CR logic)  │
└──────────────────┘                           └───────────────────────────┘
   training host                                  ARM device / redroid on x86
```

`redroid` (Android-in-Docker) provides the ARM runtime on an x86 EC2 box; `libg.so` is
extracted from the v1.3.2 APK; Scroll is cross-compiled for `armv7-linux-androideabi` and
installed into the container, exposing port 9340.

## Wire protocol (`client.py` / `env.py`)

JSON request/response over a persistent, lock-guarded TCP socket (thread-safe for multiple
self-play workers).

```
new_battle(deck_p0, deck_p1)            -> battle_id
step(battle_id, commands_p0, commands_p1, ticks) -> GameState
destroy(battle_id)
ping()
```

- `CardPlay = {spell_id, x, y}` is the command unit.
- `GameState` carries: `tick`, `max_tick`, `game_over`, `winner`, `checksum`, per-player
  `elixir / king_hp / princess_left_hp / princess_right_hp / crowns / hand[4] / next_card`,
  and an `entities` list. This maps directly onto the L4 observation encoder.
- Per step, Scroll: injects commands via `LogicCommandManager::add_command()` → calls
  `LogicGameMode::update_one_tick()` N times → reads state from `libg.so` memory → returns it.

`ScrollBattleEnv` mirrors `crsim.game.CRGame` on purpose, so the env is a drop-in backend swap.
Default deck (from Scroll's `save.rs`): Knight `26000000`, Archers `26000001`, Bomber
`26000013`, Fireball `28000001`, Arrows `28000000`, Giant `26000003`, Prince `26000004`, Baby
Dragon `26000005`.

## What works vs what needs RE

**Working:** battle creation vs NPC, tick-by-tick `update_one_tick()`, checksum/desync
detection, deck loading, full v1.3.2 card roster from CSVs, the Python TCP client.

**Needs RE (find offsets in `libg.so`):** entity-state iteration (`GameObjectManager` →
`LogicGameObject` list → pos/HP/type), command injection (`SpellCardCommand` vtable), elixir
field, hand/deck during battle, win flags. Plus the **P0 battle-bootstrap** below.

Two viable state-extraction routes: (a) walk `GameObjectManager` (offset 40) and read each
`LogicGameObject`; or (b) parse `LogicGameMode::encode()`'s `ChecksumEncoder` byte stream,
which already serializes the entire state — often cheaper than chasing per-field offsets.

## The P0 blocker, precisely

Goal: bootstrap a battle that survives `begin_battle` and steps **200,000 ticks** with towers
present and no SIGSEGV. The RE established, in order (each ruling out a wrong theory):

1. **Data is complete** — `locations.csv` has all 37 columns incl. `FileName@23`. The earlier
   "missing column" theory was wrong; drop the workarounds it spawned.
2. **Tilemap id is correct** — `set_location` looks up by `[location+0x10] % 1000000`;
   `init_tilemap` registers by `location.parent.global_id % 1000000`. Both = **191**
   (`0x114a18(x)` decodes to `x % 1000000`). The tilemap is found: `[battle+0x28]` non-null.
3. **Real defect**: after `set_location`, **`[battle+0xb0]` (the arena object) is NULL**.
   `set_location → 0xee360` builds the tilemap + grids but never constructs the arena.
4. **Crash**: `begin_battle` enters side/crowns processing (`0x114784` / `0x11ade8`), reads the
   king-tower object at `[side+0x44]` (null), and faults (fault offset `0x508`; crash PC
   `0x000f32af` is inside the ARM→x86 translator → it's a *guest* null-deref, not a translator
   bug).
5. **Why towers are null**: tower entities are wired in `begin_battle`'s LATE block (`0xee708+`),
   **hard-gated on `[battle+0xb0]` non-null**. Arena null ⇒ block skipped ⇒ `[ctrl+0x44]`
   (king), `[ctrl+0x48]`/`[ctrl+0x4c]` (princess) stay null ⇒ next tick's over-check calls
   `isAlive` ⇒ `ldr [tower+0x8]` on null ⇒ SIGSEGV.
6. **Who builds the arena**: only `load_battle_state` (`0x11a428 → eed64`). `start_mission`
   never calls it — **even though the home analog `load_home_state` (`0x11A274`) IS wired**.

So this is no longer a mystery; it's a bounded engineering task: get the engine to construct
its own arena + per-side tower entities before `begin_battle`. See `docs/p0-engine-fix-spec.md`.

## Offset / symbol table (working hypothesis, v1.3.2 `libg.so`)

| Symbol / slot | RVA / offset | Role |
|---|---|---|
| `update_one_tick` (`LogicGameMode`) | `0x11A718` | per-tick advance (Scroll already calls) |
| `LogicCommandManager::add_command` | `0xF0914` | inject a command |
| `load_home_state` | `0x11A274` | home analog — **already wired** ✅ |
| `load_battle_state` | `0x11a428` → `eed64` | builds arena + per-side towers — **never called** |
| `start_mission` | (Scroll-written) | hand-written battle boot; skips the above |
| `begin_battle` | … → `0xee708` | LATE block wires towers, gated on `[battle+0xb0]` |
| `set_location` | → `0xee360` | builds tilemap+grids, NOT arena |
| `arena_init` (ctor candidate) | `0xFFC0C` | allocates; `str r1,[r0],#0x20` |
| `0xef0c8` | `0xef0c8` | last call in `set_location` worker (arena-ctor candidate) |
| `createGameObjectByData` | `0x10FC68` | entity factory; ctor `0x1121A4`; HP at `+0x8` |
| side/crowns processing | `0x114784`, `0x11ade8` | reads `[side+0x44]` king tower |
| `0x114a18(x)` | — | `x % 1000000` (id decode) |

| Struct offset | Meaning |
|---|---|
| `[battle+0]` | world/level object (null vtable in this build → likely the deeper root) |
| `[battle+0x28]` | tilemap pointer (non-null, found) |
| `[battle+0xb0]` | **arena object (NULL = root cause)** |
| `[location+0x10]` | location id used by `set_location` lookup |
| `[side+0x44]` | king-tower object (null → crash) |
| `[ctrl+0x44]` / `+0x48` / `+0x4c` | king / princess-left / princess-right towers |
| `[tower+0x8]` | tower HP field deref'd by `isAlive` (faults when tower null) |

## RE practice notes
- **Attribute faults to guest RVAs**: crash PCs land in the translator's `<anonymous>` code.
  Maintain a guest↔host PC map so every fault maps back to a `libg.so` RVA.
- **Avoid linear sweeps**: Thumb/data interleaving desyncs linear disassembly over the 200KB+
  text (this bit the RE repeatedly). Prefer function-local disassembly and resolve
  indirect/vtable calls explicitly.
- **Pin the binary**: every offset above is specific to this v1.3.2 build; a re-extract
  invalidates the table.
