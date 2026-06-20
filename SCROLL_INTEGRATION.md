# Scroll Integration — Real CR Engine for RL Training

## Overview

[Scroll](https://git.xeondev.com/Supercell/Scroll) is an experimental Clash Royale server that reuses `libg.so` — the actual compiled C++ game engine from Supercell's APK. Instead of reimplementing game physics, Scroll calls the real engine's `update_one_tick()` function directly, giving us **100% faithful simulation** at **thousands of ticks/sec**.

This document explains how to set up Scroll as a headless RL training environment and wire it into our ClashRoyale-Zero training pipeline.

## Architecture

```
┌─────────────────┐         JSON/TCP          ┌──────────────────────┐
│  Python RL Agent │  ◄──────────────────────► │  Scroll Headless     │
│  (MCTS + NN)     │       port 9340           │  Battle Server       │
│                  │                           │  (Rust)              │
│  scroll_bridge/  │                           │                      │
│    client.py     │                           │  headless/server.rs  │
│    env.py        │                           │       ▼              │
│                  │                           │  libg.so (real CR)   │
└─────────────────┘                           └──────────────────────┘
     Training GPU(s)                              ARM device / redroid
```

**Data flow per step:**
1. Python sends `{"type": "step", "battle_id": 1, "commands_p0": [...], "ticks": 1}`
2. Scroll injects commands via `LogicCommandManager::add_command()`
3. Scroll calls `LogicGameMode::update_one_tick()` N times
4. Scroll reads game state from libg.so memory structures
5. Scroll responds with `{"type": "state", "state": {...}}`

## Setup

### Prerequisites

- Android device with `armeabi-v7a` support, **OR** redroid (Android-in-Docker) on x86 Linux
- Clash Royale v1.3.2 APK (for `libg.so`)
- Rust 1.85+ with `armv7-linux-androideabi` target
- Android NDK

### Step 1: Extract libg.so

```bash
# Download CR v1.3.2 APK from an archive site
# (search: "com.supercell.clashroyale 1.3.2 apk")
unzip com.supercell.clashroyale-v1.3.2.apk -d cr_apk
cp cr_apk/lib/armeabi-v7a/libg.so .
```

### Step 2: Set up redroid (Android-in-Docker)

```bash
# On an x86_64 EC2 instance (Ubuntu 22.04)
sudo apt-get install -y linux-modules-extra-$(uname -r)
sudo modprobe binder_linux devices="binder,hwbinder,vndbinder"

# Run redroid container
docker run -d --name redroid \
    --privileged \
    -p 5555:5555 \
    -p 9340:9340 \
    redroid/redroid:11.0.0-latest

# Connect via ADB
adb connect localhost:5555
```

### Step 3: Build Scroll with headless mode

```bash
cd Scroll

# Install cross-compilation toolchain
rustup target add armv7-linux-androideabi
cargo install cargo-ndk

# Build (needs ANDROID_NDK_HOME set)
export ANDROID_NDK_HOME=/path/to/android-ndk
cargo ndk -t armeabi-v7a build --release
```

### Step 4: Deploy to redroid

```bash
# Install server APK into the redroid container
adb -s localhost:5555 install scroll-server.apk

# Install patched client APK
adb -s localhost:5555 install scroll-client.apk

# Forward the headless server port
adb -s localhost:5555 forward tcp:9340 tcp:9340
```

### Step 5: Connect Python

```python
from scroll_bridge import ScrollClient, ScrollBattleEnv

# Direct client usage
client = ScrollClient("localhost", 9340)
client.connect()
assert client.ping()

battle_id = client.new_battle(
    deck_p0=[26000000, 26000001, 26000013, 28000001,
             28000000, 26000003, 26000004, 26000005],
    deck_p1=[26000000, 26000001, 26000013, 28000001,
             28000000, 26000003, 26000004, 26000005],
)

# Tick forward 100 steps
for _ in range(100):
    state = client.step(battle_id, ticks=1)
    print(f"Tick {state.tick}, P0 king HP: {state.p0_king_hp}")

client.destroy(battle_id)
client.close()

# Or use the Gymnasium-style env
env = ScrollBattleEnv("localhost", 9340)
env.connect()
state = env.reset()
while not env.done:
    state, reward, done, info = env.step()
env.close()
```

## What's Working vs What Needs RE Work

### Working Now
- ✅ Battle creation (training battles against NPC)
- ✅ Tick-by-tick simulation via `update_one_tick()`
- ✅ Checksum verification (detect desync)
- ✅ Deck loading from JSON
- ✅ Full card roster from CSV data tables
- ✅ Python TCP client with JSON protocol

### Needs Reverse Engineering
These require finding additional offsets in `libg.so`:

| Feature | What's Needed | Difficulty |
|---------|---------------|------------|
| **Entity state** | `GameObjectManager` (offset 40) → iterate `LogicGameObject` list → read position, HP, type | Medium |
| **Command injection** | Construct `LogicCommand` for `SpellCardCommand` with correct vtable | Medium |
| **Elixir reading** | Find elixir field in per-player battle state | Easy |
| **Hand/deck state** | Read current hand from `LogicSpellDeck` during battle | Easy-Medium |
| **Win detection** | Read crown counts / tower destroyed flags | Easy |

### RE Approach
1. Use a disassembler (Ghidra/IDA) on `libg.so`
2. Key classes to focus on:
   - `GameObjectManager` — entity iteration
   - `LogicCharacter` / `LogicBuilding` — unit/tower state
   - `LogicBattleCommand` / `SpellCharacterCommand` — card play commands
   - `LogicPlayerData` — elixir, deck, hand state
3. Alternatively, use the `encode()` path: `LogicGameMode::encode()` serializes the full state into a `ChecksumEncoder` (byte stream) — we can parse this stream format to extract everything

## Fallback: Hybrid Approach

Until the RE work is complete, use our **Python simulator** (crsim/) for MCTS rollouts and **Scroll** for:
1. **Validation** — compare Python sim outputs vs Scroll checksums to tune our sim
2. **Data generation** — run thousands of NPC training battles to collect trajectory data
3. **Final evaluation** — test trained agents against the real engine

```python
# Hybrid: Python sim for fast MCTS, Scroll for ground truth
from crsim.game import CRGame
from scroll_bridge import ScrollBattleEnv

# Train with Python sim (fast, no setup)
game = CRGame()
# ... MCTS + training loop ...

# Validate against real engine
env = ScrollBattleEnv("localhost", 9340)
env.connect()
# ... replay the same sequence of actions ...
# ... compare tower HP, game outcome, etc. ...
```

## Performance Expectations

| Metric | Python Sim | Scroll (ARM on x86) | Scroll (native ARM) |
|--------|-----------|---------------------|---------------------|
| Ticks/sec/core | 500-2000 | 5,000-20,000 | 20,000-100,000 |
| Games/min (full) | 5-20 | 50-200 | 200-1000 |
| Card fidelity | ~80% | 100% | 100% |
| Cards supported | 20 | ~70 (v1.3.2) | ~70 (v1.3.2) |

## Key Scroll Source Files

| File | Purpose |
|------|---------|
| `libserver/src/logic/mode.rs` | `LogicGameMode` — main game loop, `update_one_tick()` at `0x11A718` |
| `libserver/src/logic/battle.rs` | `LogicBattle` — battle setup, deck/location config |
| `libserver/src/logic/command/manager.rs` | `LogicCommandManager::add_command()` at `0xF0914` |
| `libserver/src/logic/spell.rs` | Deck/spell structures |
| `libserver/src/session/mod.rs` | `start_mission()` — how battles are initialized |
| `libserver/src/session/handlers/home.rs` | `on_end_client_turn_message()` — the tick loop pattern |
| `libserver/src/headless/` | **NEW** — our headless RL server addition |
