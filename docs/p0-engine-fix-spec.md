# P0 Task Spec — Scroll Battle Bootstrap (arena + towers)

**Status:** open · **Priority:** P0 (gates 100%-fidelity training & final eval) · **Owner:** _unassigned_

A self-contained task spec for getting Scroll's headless server to run a real battle. Read
`docs/layers/L3-scroll-bridge-and-re.md` first for the full RE context and offset table. All
RVAs are the working hypothesis for `libg.so` from CR **v1.3.2** — pin that exact binary.

---

## 1. Objective & win condition

**Objective:** boot a battle through Scroll such that `begin_battle` succeeds and the engine
steps a full game without crashing, with towers present and updating.

**Win condition (binary, no success claim without this exact evidence):**
> The benchmark steps **200,000 ticks**, **no SIGSEGV**, and the post-`begin_battle` dump shows
> `[battle+0xb0]` (arena) non-null and `[ctrl+0x44]/[ctrl+0x48]/[ctrl+0x4c]` (king + 2 princess
> towers) non-null with sane HP.

Capture the benchmark log line that proves all three. Keep this benchmark as a permanent
CI-style canary for the Scroll backend.

---

## 2. Root cause (established by prior RE)

`start_mission` (Scroll's hand-written battle boot) builds the tilemap + grids
(`set_location → 0xee360`) but **never constructs the arena object `[battle+0xb0]`**. Tower
entities are wired only in `begin_battle`'s LATE block (`0xee708+`), which is **hard-gated on
`[battle+0xb0]` being non-null**. So:

```
arena == null
   └─► tower-wiring block at 0xee708 skipped
          └─► [ctrl+0x44/0x48/0x4c] stay null
                 └─► next tick: isAlive → ldr [tower+0x8] on null → SIGSEGV (fault 0x508)
```

The arena + per-side towers are built only by **`load_battle_state` (`0x11a428 → eed64`)**,
which `start_mission` skips — **even though the home analog `load_home_state` (`0x11A274`) is
already wired** for home mode. Ruled out already: missing-CSV-column theory (data is complete,
37 cols) and tilemap-id mismatch (lookup id == reg id == 191; tilemap found at `[battle+0x28]`).

---

## 3. Environment / access needed

- The Scroll source (`libserver/`, `headless/`) + cross-compile toolchain
  (`armv7-linux-androideabi`, `cargo-ndk`, `ANDROID_NDK_HOME`).
- redroid container running on the x86 box, `libg.so` (v1.3.2) installed, port 9340 forwarded.
- The existing benchmark harness (`benchmark_diag*.rs` from the RE sessions) that dumps battle
  sub-object pointers and steps N ticks. **Restore/locate it** — it is the test rig for the
  win condition.
- Disassembler (Ghidra/IDA) with the v1.3.2 `libg.so` loaded and the offset table from L3.

> If any of the above (Scroll source, redroid, the benchmark, the exact APK/`libg.so`) is
> missing or different from what's documented, **stop and escalate** — do not rebuild or
> substitute it silently, since the offsets are build-specific.

---

## 4. Approaches, in priority order

Guiding principle: **prefer the engine's own construction paths over hand-poking guest
memory.** Manual construction inherits responsibility for every sub-object the engine normally
initializes, and the RE already shows a null-chasing pattern (tilemap → arena → towers →
sides). Approaches 1–2 let the engine build the long tail; reserve 3 for fields it never
builds headless.

### Approach 1 — restore the missing `load_battle_state` call (do this first)
The home path already calls `load_home_state` (`0x11A274`). The battle path should
symmetrically call `load_battle_state` (`0x11a428`).
1. Treat the home boot as a **Rosetta Stone**: disassemble the real `startMission`/battle-boot
   sequence and the working home-boot sequence side by side.
2. Find the call the battle path drops relative to home (the `load_battle_state` invocation and
   whatever sets up the stream/args it consumes).
3. Add that call to Scroll's `start_mission` (in `libserver/src/session/`), between player
   setup (`add_player`) and `begin_battle`.
4. Run the benchmark → check win condition.

**Why first:** one call may build the arena *and* towers *and* the `world`/level object
(`[battle+0]`, currently a null vtable — a strong hint the true root is an unconstructed world
that owns the arena). Highest leverage, most correct, smallest patch.

### Approach 2 — drive `load_battle_state` directly with a constructed stream
If the call can't simply be restored (e.g. it's reached only via an indirect/vtable path):
1. Find how the real engine builds the byte stream `load_battle_state`/`eed64` deserializes
   (its caller). Resolve the indirect call — **do not** rely on linear disassembly (Thumb/data
   desync).
2. Construct the stream: battle type + location id (191) + per-side king/2-princess tower rows.
3. Call `0x11a428` with it before `begin_battle`.
4. Run the benchmark → check win condition.

Still uses the engine's own construction logic; you only supply input.

### Approach 3 — manual arena + tower build (last resort)
Only if 1–2 are infeasible. Expect to chase further nulls.
1. Allocate the arena → store into `[battle+0xb0]` (ctor candidates `0xFFC0C`, or `0xef0c8`
   = last call in `set_location`'s worker; `0xFFC0C` does `str r1,[r0],#0x20`).
2. Build 3 tower entities per side via the factory `createGameObjectByData` (`0x10FC68`, ctor
   `0x1121A4`, HP at `+0x8`).
3. Store king → `[ctrl+0x44]`, princess-left → `[ctrl+0x48]`, princess-right → `[ctrl+0x4c]`,
   and the king-tower object → `[side+0x44]` (the field the crash read as null).
4. Also construct/verify the `world`/level object at `[battle+0]` (null vtable today) — manual
   arena build alone likely leaves this unbuilt.
5. Run the benchmark → check win condition.

---

## 5. Debug tooling to use throughout
- **Sub-object dump after each boot step**: print `[battle+0]`, `[battle+0x28]`,
  `[battle+0xb0]`, `[side+0x44]`, `[ctrl+0x44/0x48/0x4c]` and their vtables right before
  `begin_battle` and after each candidate fix. This is how each prior theory was confirmed/killed.
- **Guest↔host PC map**: crash PCs land in the translator's `<anonymous>` code (e.g.
  `0x000f32af`); map them back to `libg.so` RVAs so faults are attributable.
- **Function-local disassembly only**: never linear-sweep the 200KB+ `.text`; resolve
  indirect/vtable calls explicitly.

## 6. Acceptance & follow-up
- [ ] Benchmark log shows 200,000 ticks, no crash, arena + 3 towers/side non-null.
- [ ] The fix lives in Scroll source (not a one-off patched memory poke), committed with the
      RVA rationale in the message.
- [ ] Benchmark wired as a canary (run before declaring the Scroll backend usable).
- [ ] Then: wire entity/elixir/hand/win extraction (L3 "needs RE" list) so `ScrollBattleEnv`
      returns full `GameState`, and run the L2 calibration harness (crsim vs Scroll on
      identical action sequences).

## 7. Out of scope (do not let these compete with P0)
- Modern-card converter (champions/evolutions on Scroll) — downstream of a running battle.
- Realtime/CV deployment.
- Backend physics tuning — comes after Scroll produces trustworthy ground-truth traces.
