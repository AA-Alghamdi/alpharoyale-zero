#!/usr/bin/env python3
"""Recover a function's call signature by finding its real callers inside
libg.so. Decodes the Thumb BL/BLX encoding at every 2-byte offset in .text
(robust against linear-disassembly desync), then dumps the argument-setup
context before each call.

Usage:
    python3 tools/p0/find_callers.py LIBG.so 0x11a428 [0x11a274 ...]

This is how load_battle_state's arity (this + 2 args) was recovered, and
cross-validated against load_home_state's known 6-arg binding.

Requires: pyelftools, capstone
"""
from __future__ import annotations

import struct
import sys

from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
from elftools.elf.elffile import ELFFile


def main() -> int:
    path = sys.argv[1]
    targets = {int(x, 16) for x in sys.argv[2:]}
    if not targets:
        print("usage: find_callers.py LIBG.so TARGET_RVA [TARGET_RVA ...]")
        return 2

    elf = ELFFile(open(path, "rb"))
    data = open(path, "rb").read()
    text = elf.get_section_by_name(".text")
    ts, tsz = text["sh_addr"], text["sh_size"]

    def vaddr_off(a):
        for s in elf.iter_segments():
            if s["p_type"] == "PT_LOAD" and s["p_vaddr"] <= a < s["p_vaddr"] + s["p_filesz"]:
                return a - s["p_vaddr"] + s["p_offset"]
        return None

    base = vaddr_off(ts)

    def bl_target(addr):
        o = base + (addr - ts)
        hw1, hw2 = struct.unpack_from("<HH", data, o)
        if (hw1 & 0xF800) != 0xF000 or (hw2 & 0xC000) != 0xC000:
            return None
        is_bl = (hw2 & 0x1000) != 0
        s = (hw1 >> 10) & 1
        imm10 = hw1 & 0x3FF
        j1 = (hw2 >> 13) & 1
        j2 = (hw2 >> 11) & 1
        imm11 = hw2 & 0x7FF
        i1 = 1 - (j1 ^ s)
        i2 = 1 - (j2 ^ s)
        off = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm11 << 1)
        if off & (1 << 24):
            off -= 1 << 25
        tgt = (addr + 4) + off
        if not is_bl:
            tgt &= ~3
        return tgt, ("BL" if is_bl else "BLX")

    hits = {t: [] for t in targets}
    for addr in range(ts, ts + tsz - 4, 2):
        r = bl_target(addr)
        if r is None:
            continue
        tgt, kind = r
        for t in targets:
            if tgt in (t, t | 1):
                hits[t].append((addr, kind))

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    for t in targets:
        print(f"\n==== callers of {t:#x}: {len(hits[t])} ====")
        for addr, kind in hits[t][:12]:
            start = addr - 28
            o = base + (start - ts)
            print(f"  -- {kind} @ {addr:#x} (arg-setup context):")
            for i in md.disasm(data[o:o + 32], start):
                mark = "   <== CALL" if i.address == addr else ""
                print(f"       {i.address:#010x}: {i.mnemonic:<8} {i.op_str}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
