#!/usr/bin/env python3
"""Validate that a `libg.so` is the Clash Royale v1.3.2 engine the P0 RE work
targets, by disassembling at the handoff RVAs and checking falsifiable
instruction fingerprints.

This does *not* require the binary to be committed — point it at a `libg.so`
you fetched yourself (see fetch_and_verify_engine.sh):

    python3 tools/p0/validate_libg.py path/to/libg-armeabi-v7a.so

Exit code 0 = all fingerprints matched (this is the expected engine).
Exit code 1 = one or more fingerprints failed.

Requires: pyelftools, capstone  (pip install pyelftools capstone)
"""
from __future__ import annotations

import hashlib
import sys

from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
from elftools.elf.elffile import ELFFile

# Expected sha256 of lib/armeabi-v7a/libg.so inside the v1.3.2 APK
# (sha256 551eff29... — see fetch_and_verify_engine.sh).
EXPECTED_LIBG_SHA256 = "f73b70d714d1423c6a8ef520982ce49d870e2db027021ebdffdd755efece6a80"


def disasm(md: Cs, data: bytes, base: int, vaddr: int, n: int):
    off = base + vaddr
    out = []
    for insn in md.disasm(data[off:off + 4 * (n + 2)], vaddr):
        out.append((insn.address, insn.mnemonic, insn.op_str))
        if len(out) >= n:
            break
    return out


def vaddr_to_off(elf: ELFFile, addr: int):
    for seg in elf.iter_segments():
        if seg["p_type"] != "PT_LOAD":
            continue
        if seg["p_vaddr"] <= addr < seg["p_vaddr"] + seg["p_filesz"]:
            return addr - seg["p_vaddr"] + seg["p_offset"]
    return None


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "libg-armeabi-v7a.so"
    with open(path, "rb") as f:
        data = f.read()
    elf = ELFFile(open(path, "rb"))

    sha = hashlib.sha256(data).hexdigest()
    note = ("(matches expected)" if sha == EXPECTED_LIBG_SHA256
            else "(differs from expected — fingerprints below are authoritative)")
    print(f"[elf] machine={elf['e_machine']} type={elf['e_type']} size={len(data):,}")
    print(f"[sha256] {sha} {note}")
    if elf["e_machine"] != "EM_ARM":
        print(f"  FAIL: expected EM_ARM, got {elf['e_machine']}")
        return 1

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)

    checks: list[tuple[str, int, callable]] = [
        # (name, rva, predicate over [(addr,mnem,ops),...])
        (
            "begin_battle null-arena gate (ldr [r4,#0xb0]; cmp; beq)",
            0xEE708,
            lambda d: d[0][1].startswith("ldr") and "[r4, #0xb0]" in d[0][2]
            and d[1][1] == "cmp" and d[2][1].startswith("beq"),
        ),
        (
            "x % 1000000 magic (movw 0xde83; movt 0x431b; smmul)",
            0x114A18,
            lambda d: "0xde83" in d[0][2] and "0x431b" in d[1][2] and d[2][1] == "smmul",
        ),
        (
            "arena ctor store (str r1, [r0], #0x20 within first 12 insns)",
            0xFFC0C,
            lambda d: any(m == "str" and o.replace(" ", "") == "r1,[r0],#0x20" for _, m, o in d),
        ),
        (
            "load_battle_state calls builder 0xeed64",
            0x11A428,
            lambda d: any(m.startswith("bl") and "0xeed64" in o for _, m, o in d),
        ),
        (
            "load_home_state prologue (push.w {...} ; this+0x18)",
            0x11A274,
            lambda d: d[0][1].startswith("push") and any("r4, #0x18" in o for _, _, o in d),
        ),
    ]

    all_ok = True
    for name, rva, pred in checks:
        # window length: the bl-target / str checks need a longer window
        n = 90 if rva == 0x11A428 else (16 if rva == 0xFFC0C else 8)
        base = vaddr_to_off(elf, rva)
        if base is None:
            print(f"  FAIL [{name}] @ {rva:#x}: not in any LOAD segment")
            all_ok = False
            continue
        d = disasm(md, data, base - rva, rva, n)
        ok = False
        try:
            ok = pred(d)
        except (IndexError, KeyError):
            ok = False
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name} @ {rva:#x}")
        if not ok:
            for a, m, o in d[:6]:
                print(f"         {a:#010x}: {m:<8} {o}")
            all_ok = False

    print()
    if all_ok:
        print("RESULT: all fingerprints matched — this is the CR v1.3.2 P0 engine.")
        return 0
    print("RESULT: fingerprint mismatch — NOT the expected engine build.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
