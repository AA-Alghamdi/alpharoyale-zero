#!/usr/bin/env python3
"""Dynamically validate the P0 battle-bootstrap fix by CPU-emulating the real CR
v1.3.2 engine (libg.so) with Unicorn.

Background (P0): start_mission() builds a battle via set_location() (tilemap +
grids only) and never calls load_battle_state(), so the arena field battle+0xb0
stays NULL. begin_battle()'s tower-wiring block is hard-gated on that field, so
towers are never wired and the next tick derefs null -> SIGSEGV.

This is a *validation* harness, not a runtime. We cannot reconstruct the engine's
gamedata/asset state on-box, so we prove the fix's MECHANISM (not a full battle):

  A  gate         begin_battle@0xee708 reads battle+0xb0 and branches: NULL skips
                  the tower-wiring block (the crash), non-null runs it.
  B  full builder running the arena builder@0xeed64 bare walls on uninitialised
                  gamedata -> a bare memory-poke can't drive it (must use the
                  engine's own init path, i.e. a real battle-setup stream).
  B2 arena core   the arena store (battle+0xb0) is gated by load_battle_state's
                  arg2: arg2==0 -> arena NULL (today's P0 state); arg2!=0 ->
                  operator new(0x1c) + ctor, pointer stored at battle+0xb0.
  B3 end-to-end   from the builder entry, with data-driven helpers stubbed and
                  arg2!=0, control reaches the store and battle+0xb0 is non-null.

Exit 0 if the core proofs (A, B2, B3) hold. Requires: unicorn, capstone,
pyelftools, and a fetched libg.so (tools/p0/fetch_and_verify_engine.sh).
"""
from __future__ import annotations

import sys

from uc_engine import SCRATCH, Engine
from unicorn import (
    UC_HOOK_MEM_FETCH_UNMAPPED,
    UC_HOOK_MEM_READ_UNMAPPED,
    UC_HOOK_MEM_WRITE,
    UC_HOOK_MEM_WRITE_UNMAPPED,
    UcError,
)
from unicorn.arm_const import UC_ARM_REG_PC, UC_ARM_REG_R8, UC_ARM_REG_R9

BATTLE = SCRATCH + 0x1000
ARENA_OFF = 0xB0

# begin_battle gate
GATE, FALLTHROUGH, SKIP = 0xEE708, 0xEE712, 0xEE854
# arena builder (the fn load_battle_state calls at 0x11a4c6)
BUILDER, BUILDER_END = 0xEED64, 0xEF04A
# arena-construction core inside the builder
CORE_ENTRY, CORE_STOP = 0xEEF04, 0xEEF26
# data-driven helpers the builder calls (stubbed for the isolated/e2e probes)
HELPERS = {
    0x1188FC, 0x118A7C, 0x11AD80, 0x11ADE8, 0x11F430, 0x11F604, 0x17C394,
    0x17C51C, 0x17CEA8, 0x17CF5C, 0x17E2D4, 0xEAE00, 0xEB8F4, 0xEB9F8,
    0xEBD18, 0xEE05C, 0xEE17C, 0xEFC90, 0xEFDD4, 0xEFF14, 0xF0168, 0xF0188,
}


def _hdr(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def exp_a_gate() -> bool:
    _hdr("A  begin_battle@0xee708 gate  (ldr [r4,#0xb0]; cmp #0; beq 0xee854)")

    def run(arena_value):
        eng = Engine()
        uc = eng.uc
        uc.mem_write(BATTLE + ARENA_OFF, arena_value.to_bytes(4, "little"))
        from unicorn.arm_const import UC_ARM_REG_LR, UC_ARM_REG_R4
        uc.reg_write(UC_ARM_REG_R4, BATTLE)
        uc.reg_write(UC_ARM_REG_LR, 0x4444_4444)
        pc = GATE
        for _ in range(8):
            uc.emu_start(pc | 1, 0, count=1)
            pc = uc.reg_read(UC_ARM_REG_PC)
            if pc in (FALLTHROUGH, SKIP):
                break
        return pc

    null_pc = run(0)
    nn_pc = run(SCRATCH + 0x8000)
    print(f"  arena = NULL          -> {null_pc:#x}  "
          f"({'SKIP: towers NOT wired = P0 crash' if null_pc == SKIP else '??'})")
    print(f"  arena = {SCRATCH+0x8000:#x}   -> {nn_pc:#x}  "
          f"({'FALL-THROUGH: tower-wiring runs' if nn_pc == FALLTHROUGH else '??'})")
    ok = null_pc == SKIP and nn_pc == FALLTHROUGH
    print(f"  => {'PROVEN: the crash gate diverges on battle+0xb0.' if ok else 'INCONCLUSIVE'}")
    return ok


def exp_b_full() -> bool:
    _hdr("B  arena builder@0xeed64 run bare  (expected to wall on gamedata)")
    eng = Engine(trace=True)
    uc = eng.uc
    fault = {}

    def on_fault(uc, access, addr, size, value, user):
        fault.update(pc=uc.reg_read(UC_ARM_REG_PC), addr=addr)
        return False

    uc.hook_add(
        UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED | UC_HOOK_MEM_FETCH_UNMAPPED,
        on_fault,
    )
    uc.mem_write(BATTLE, b"\x00" * 0x400)
    try:
        eng.call(BUILDER, r0=BATTLE, r1=SCRATCH + 0x10000, r2=SCRATCH + 0x20000, count=200000)
        note = "returned (unexpected for a bare run)"
    except UcError as e:
        note = str(e)
    last = eng.trace_log[-1] if eng.trace_log else (0, "?")
    print(f"  instructions executed: {len(eng.trace_log)}")
    print(f"  imports hit: {sorted(set(eng.import_calls))}")
    print(f"  stop: {note}")
    if fault:
        print(f"  walled at pc={fault['pc']:#x} accessing {fault['addr']:#x}")
    else:
        print(f"  walled at pc={last[0]:#x}: {last[1]}")
    print("  => EXPECTED: bare execution can't build the arena (needs initialised")
    print("     gamedata) -> the fix must drive the engine's own load path, not poke memory.")
    return True  # informational; not a pass/fail gate


def exp_b2_arena_core() -> bool:
    _hdr("B2 arena-construction core 0xeef04..0xeef1e  (arg2 = the lever)")

    def run(arg2):
        eng = Engine()
        uc = eng.uc
        for a in (0xEFC90, 0xEFF14, 0xEFDD4, 0xF0168):
            eng.stub(a, "identity")
        eng.stop_at(CORE_STOP, BUILDER_END)
        uc.mem_write(BATTLE, b"\x00" * 0x400)
        uc.reg_write(UC_ARM_REG_R9, BATTLE)   # sb = battle
        uc.reg_write(UC_ARM_REG_R8, arg2)     # arg2 (load_battle_state's r2)
        from unicorn.arm_const import UC_ARM_REG_LR
        uc.reg_write(UC_ARM_REG_LR, 0x4444_4444)
        try:
            uc.emu_start(CORE_ENTRY | 1, CORE_STOP, count=200)
        except UcError:
            pass
        return int.from_bytes(uc.mem_read(BATTLE + ARENA_OFF, 4), "little"), eng.import_calls

    a0, _ = run(0)
    a1, imp1 = run(SCRATCH + 0x20000)
    s0 = "arena NULL = P0 state" if a0 == 0 else "??"
    s1 = ("arena allocated via " + imp1[0]) if a1 and imp1 else "??"
    print(f"  arg2 = 0           -> battle+0xb0 = {a0:#x}   ({s0})")
    print(f"  arg2 = {SCRATCH+0x20000:#x}  -> battle+0xb0 = {a1:#x}   ({s1})")
    ok = a0 == 0 and a1 != 0 and "_Znwj" in imp1
    msg = "PROVEN: arg2 gates arena construction into battle+0xb0." if ok else "INCONCLUSIVE"
    print(f"  => {msg}")
    return ok


def exp_b3_end_to_end() -> bool:
    _hdr("B3 builder@0xeed64 end-to-end (helpers stubbed, arg2!=0)")
    eng = Engine(trace=True)
    uc = eng.uc
    for a in HELPERS:
        eng.stub(a, "identity")
    writes = []

    def on_write(uc, access, addr, size, value, user):
        if addr == BATTLE + ARENA_OFF:
            writes.append((uc.reg_read(UC_ARM_REG_PC), value))

    uc.hook_add(UC_HOOK_MEM_WRITE, on_write)
    uc.mem_write(BATTLE, b"\x00" * 0x400)
    note = "ok"
    try:
        eng.call(BUILDER, r0=BATTLE, r1=SCRATCH + 0x10000, r2=SCRATCH + 0x20000, count=300000)
    except UcError as e:
        note = str(e)
    final = int.from_bytes(uc.mem_read(BATTLE + ARENA_OFF, 4), "little")
    print(f"  instructions executed: {len(eng.trace_log)}   stop: {note}")
    print(f"  imports hit: {sorted(set(eng.import_calls))}")
    print(f"  store(s) to battle+0xb0: {[(hex(p), hex(v)) for p, v in writes]}")
    print(f"  FINAL battle+0xb0 = {final:#x}  "
          f"({'ARENA NON-NULL' if final else 'null'})")
    ok = final != 0 and any(p == 0xEEF1E for p, _ in writes)
    print(f"  => {'PROVEN: entry->arena store reached; arena non-null.' if ok else 'INCONCLUSIVE'}")
    return ok


def main() -> int:
    print(__doc__.split("\n\n")[0])
    results = {
        "A  gate": exp_a_gate(),
        "B2 arena core": exp_b2_arena_core(),
        "B3 end-to-end": exp_b3_end_to_end(),
    }
    exp_b_full()  # informational
    _hdr("SUMMARY")
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(results.values())
    print(f"\n{'P0 fix mechanism VALIDATED in emulation.' if ok else 'Validation incomplete.'}")
    print("Note: this proves function-local logic, not a full real-engine battle")
    print("(that still requires an ARM exec host with initialised gamedata).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
