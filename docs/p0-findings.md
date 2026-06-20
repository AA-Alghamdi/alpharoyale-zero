# P0 findings — real-engine sourcing, verification, and the battle-bootstrap bug

**Status:** the P0 *artifact* blocker is resolved. The two things P0 needs that
were previously assumed unavailable — the real CR engine binary and the Scroll
server source — were both located on public archives, downloaded, and verified.
The remaining blocker is an **execution environment** to run 32-bit ARM code
(redroid/binder), which the current VM kernel cannot provide.

Nothing proprietary is committed here. `tools/p0/` fetches the binary from a
public archive and verifies it byte-for-byte; this document records what was
verified and what the fix is.

---

## 1. The engine binary (verified)

| | |
|---|---|
| Source APK | Clash Royale **v1.3.2**, from archive.org (`ClashRoyale1.3.2Lastapk.com`) |
| APK sha256 | `551eff29a8d151a147c9885fc936c71c667f7212935d22d23b001bf672329046` |
| `lib/armeabi-v7a/libg.so` | ELF32, **EM_ARM**, Thumb-2, 4,410,156 bytes |
| armeabi-v7a sha256 | `f73b70d714d1423c6a8ef520982ce49d870e2db027021ebdffdd755efece6a80` |
| `lib/x86/libg.so` | also present (6,089,424 bytes) — not the RE target |

The APK ships **only 32-bit** libs (`armeabi-v7a`, `x86`); there is no
`arm64-v8a`. All handoff RVAs are 32-bit ARM, so `armeabi-v7a/libg.so` is the
binary the reverse-engineering targets.

### Why we are sure it is the exact build the handoff RE describes

Five independent, falsifiable fingerprints from the handoff all match on this
binary (reproduce with `tools/p0/validate_libg.py`):

| # | Fingerprint | RVA | Disassembly (Thumb-2) |
|---|---|---|---|
| 1 | `begin_battle` null-arena hard-gate | `0xee708` | `ldr.w r0,[r4,#0xb0]` · `cmp r0,#0` · `beq.w #0xee854` |
| 2 | `x % 1000000` magic-number division | `0x114a18` | `movw r1,#0xde83` · `movt r1,#0x431b` · `smmul r1,r0,r1` |
| 3 | arena ctor backing-store write | `0xffc24` | `str r1,[r0],#0x20` |
| 4 | `load_battle_state` → arena/tower builder | `0x11a428` | calls `bl #0xeed64` at `0x11a4c6` |
| 5 | `load_home_state` / `load_battle_state` symmetry | `0x11a274` / `0x11a428` | both prologues set up `this+0x18` |

Fingerprint #1 is the crux of the P0 crash: the entire late tower-wiring block
in `begin_battle` is gated on `[battle+0xb0]` (the arena pointer) being
non-null; when it is null the block is skipped, towers stay null, and the next
tick dereferences a null tower → SIGSEGV. Fingerprint #2 is decisive on its own:
`0x431BDE83` is the textbook signed-division magic constant for `/1000000`.

---

## 2. The Scroll server source (located + reviewed)

The handoff referred to a "Scroll" server on an unreachable box. It is in fact a
**public** project: `https://git.xeondev.com/Supercell/Scroll` ("scroll-rs —
Experimental Clash Royale server emulator on top of libg.so (v1.3.2)"). Cloned
at tag `0.1` (`d897141`).

Scroll is a JNI Android native library (built via `cargo-ndk` for
`armeabi-v7a`, run on an ARM device / redroid) that reuses the client's own
`libg.so` `Logic*` classes as a server — "turning the client into a server." It
provides idiomatic Rust bindings to engine functions at fixed RVAs.

**Important:** the upstream is the *base*. The project's headless RL additions
(`libserver/src/headless/`, the JSON/TCP server on port 9340 described in
`SCROLL_INTEGRATION.md`) are **not** in the upstream — that glue is the part that
still has to be (re)written on top.

---

## 3. The P0 bug, confirmed in real source

`go_home()` wires the home path; `start_mission()` does **not** wire the
symmetric battle path. From `libserver/src/session/mod.rs`:

```rust
// go_home(): home path IS initialized via the engine's own loader
let logic_game_mode = LogicGameMode::new(false);
logic_game_mode.load_home_state(logic_client_home, logic_client_avatar, 0, -1, random_seed);

// start_mission(): battle is assembled by hand and NEVER load_battle_state'd
let battle_mode = LogicGameMode::new(true);
let battle = battle_mode.battle.get_mut().unwrap();
battle.set_location(npc.location, false, 0);   // 0xEE05C: builds tilemap+grids, NOT the arena
battle.battle_type = 1;
battle.npc_data = npc;
battle.arena_data = arena;
battle.set_spell_decks(player_deck, npc_deck);
battle_mode.add_player(player_avatar);
battle_mode.add_player(npc_avatar);
// <-- nothing here constructs [battle+0xb0] (the arena) or the per-side towers
```

And `libserver/src/logic/mode.rs` binds `load_home_state` (`0x11A274`),
`update_one_tick` (`0x11A718`), `add_player`, `encode`, … but **has no
`load_battle_state` binding at all**. So there is no code path that calls
`0x11a428` → `0xeed64`, the only function that builds the arena + towers. This
is exactly the handoff's diagnosis, now confirmed against both the binary and
the source.

---

## 4. Recovered signatures (from real callers inside libg.so)

`load_battle_state` is never called by Scroll, but the *client's own code* calls
it. Scanning `.text` for BL/BLX targets (`tools/p0/find_callers.py`) found the
real call sites and lets us read the argument setup:

```
load_battle_state @0x11a428 — 2 call sites, both pass this + 2 args:
   0x9793c:  r0=[r4+0x54](this)  r1=r5  r2=r8       -> load_battle_state(this, a, b)
   0x97a7a:  r0=r4(this)         r1=r7  r2=r5        -> load_battle_state(this, a, b)

load_home_state  @0x11a274 — 1 call site, this + 5 args (incl. a -1):
   0x97a06:  this, r8, sb, r7, [sp]=-1, [sp+4]=...   -> 6 args total
```

The `load_home_state` recovery (6 args, one of them `-1`) **exactly matches**
Scroll's existing binding `load_home_state(lgm, lch, lca, 0, -1, seed)`, which
validates the method. Therefore the `load_battle_state` arity — **`this + 2
args`** — is reliable.

Both `load_battle_state` call sites live inside the **network message handler**
(state-code dispatch, `operator new` for reply messages). That is strong
evidence `load_battle_state` is the client-side *"deserialize a server-sent
battle state"* path — i.e. it consumes a serialized battle stream, it is not a
parameterless "make an empty arena" call.

---

## 5. Corrected fix analysis

The handoff ranked three fixes. Given that `load_battle_state` is a
**stream/state deserializer** (§4), the ranking should be refined:

1. **Restore a `load_battle_state` call (cheapest *iff* an input stream is
   available).** Add the binding to `mode.rs` (`= 0x11A428 + 1`, signature
   `(lgm, a, b)`) and call it in `start_mission` like `go_home` calls
   `load_home_state`. This only works if we can supply the battle-state stream
   it expects — which the headless path does not currently have.
2. **Drive `load_battle_state` with a constructed stream (most likely the real
   fix).** Build the minimal battle-setup byte stream (location, arena id, the
   two decks, king/2-princess tower rows) and pass it. Still uses the engine's
   own builder, so the arena + towers are correct by construction. The work is
   reproducing the stream format (parse the `encode()` output at `0x11A8D4` to
   learn the layout, then invert it).
3. **Manual arena+tower build (last resort).** Allocate the arena → `[battle+0xb0]`
   (ctor candidates `0xffc0c` / `0xef0c8`), then build 3 tower entities/side via
   `createGameObjectByData` (`0x10fc68`, ctor `0x1121a4`, HP at `+0x8`). Highest
   ongoing cost; risks leaving the owning world/level object under-built.

**Gate (unchanged):** a battle bootstraps and survives `begin_battle` +
`update_one_tick` to 200k ticks with no crash, and arena + 3 towers/side are
non-null.

---

## 6. The remaining blocker: an ARM execution environment

Running `armeabi-v7a` `libg.so` needs either a real ARM device or **redroid**
(Android-in-Docker). redroid requires the host kernel's **binder** support. On
the current VM:

- kernel `5.15.200` (custom `devin-box`) — **no `/lib/modules/<uname -r>` dir**,
  **no `binder*.ko`**, **no `/dev/binder`**, and **no `binder` filesystem
  registered** in `/proc/filesystems`. `modprobe` cannot help (the module does
  not exist for this kernel), so redroid cannot start here even with sudo.
- `/dev/kvm` exists, but KVM only accelerates x86 guests; an ARM guest is full
  software emulation (slow), and `libg.so` needs the full Android/bionic runtime
  around it.

**Options to actually run P0 (need a decision):**

- **(A) Provision a redroid-capable host** — an x86_64 box whose kernel exposes
  binder (e.g. Ubuntu with `linux-modules-extra` + `modprobe binder_linux`), per
  `SCROLL_INTEGRATION.md` §"Set up redroid". Add it to the env blueprint or give
  SSH, and the full bootstrap (build Scroll via `cargo-ndk`, deploy to redroid,
  run the headless loop) can proceed here.
- **(B) Full ARM-Android system emulation on this VM** — possible but heavy and
  slow (no KVM accel for ARM); a fragile fallback, not a training substrate.
- **(C) Targeted CPU emulation (Unicorn) of `load_battle_state` in isolation** —
  enough to *prove* the fix (arena becomes non-null, a tick survives) without a
  full device, but requires reconstructing the data-tables/allocator environment
  the function depends on. Good for validation, not for volume.

---

## 7. Reproduce it

```bash
pip install pyelftools capstone

# 1. fetch + byte-verify + fingerprint-validate the engine binary
tools/p0/fetch_and_verify_engine.sh            # -> .p0-engine/libg-armeabi-v7a.so

# 2. clone the upstream Scroll server (pinned)
tools/p0/clone_scroll.sh                        # -> .p0-engine/Scroll

# 3. (optional) re-derive a function's signature from its real callers
python3 tools/p0/find_callers.py .p0-engine/libg-armeabi-v7a.so 0x11a428 0x11a274
```
