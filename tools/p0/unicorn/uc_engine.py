"""Minimal Unicorn loader for the Clash Royale v1.3.2 armeabi-v7a `libg.so`.

This is a *validation* substrate, not a runtime: it maps the engine binary into a
Unicorn CPU and lets us execute individual functions in isolation to check the
P0 battle-bootstrap mechanism (see validate_p0_fix.py). It deliberately does NOT
reconstruct the engine's gamedata/asset state, so only function-local logic can
be validated.

The binary is PIC with vaddr base 0, so we map at base 0 and R_ARM_RELATIVE
relocs need no fixup. Imported PLT functions are intercepted at their veneer
entry address (the ARM veneers would corrupt our Thumb context if executed) and
serviced in Python: a bump allocator behind operator-new/malloc, working mem*,
no-op delete/free, and a logged 0-return for everything else.

The proprietary binary is never committed; fetch it with
tools/p0/fetch_and_verify_engine.sh (writes .p0-engine/libg-armeabi-v7a.so).
"""
from __future__ import annotations

import os

from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
from elftools.elf.elffile import ELFFile
from unicorn import UC_ARCH_ARM, UC_HOOK_CODE, UC_MODE_THUMB, Uc
from unicorn.arm_const import (
    UC_ARM_REG_CPSR,
    UC_ARM_REG_LR,
    UC_ARM_REG_PC,
    UC_ARM_REG_R0,
    UC_ARM_REG_R1,
    UC_ARM_REG_R2,
    UC_ARM_REG_SP,
)

PAGE = 0x1000
STACK_TOP = 0xB000_0000
STACK_SIZE = 0x0010_0000
SCRATCH = 0xC000_0000          # caller-built objects (battle / LGM / fake arena)
SCRATCH_SIZE = 0x0020_0000
HEAP = 0xA000_0000             # bump allocator
HEAP_SIZE = 0x0040_0000

DEFAULT_LIBG = ".p0-engine/libg-armeabi-v7a.so"


def _align_up(x, a=PAGE):
    return (x + a - 1) & ~(a - 1)


def resolve_libg(path: str | None = None) -> str:
    cand = path or os.environ.get("LIBG_PATH") or DEFAULT_LIBG
    if os.path.isfile(cand):
        return cand
    # also try relative to the repo root (two dirs up from tools/p0/unicorn)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    alt = os.path.join(root, DEFAULT_LIBG)
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError(
        f"libg.so not found ({cand}). Fetch it first:\n"
        f"    tools/p0/fetch_and_verify_engine.sh\n"
        f"or set LIBG_PATH=/path/to/libg-armeabi-v7a.so"
    )


class Engine:
    def __init__(self, libg_path: str | None = None, trace: bool = False):
        self.path = resolve_libg(libg_path)
        with open(self.path, "rb") as fh:
            self.data = fh.read()
        self.elf = ELFFile(open(self.path, "rb"))
        self.uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
        self.md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
        self.trace = trace
        self.trace_log: list[tuple[int, str]] = []
        self.import_calls: list[str] = []
        self._heap_ptr = HEAP
        self._imports: dict[int, str] = {}     # plt veneer entry addr -> symbol
        self._stubs: dict[int, str] = {}       # addr -> "identity" | "ret0"
        self._stop_at: set[int] = set()
        self._map_segments()
        self._setup_regions()
        self._wire_imports()

    # ---- memory -------------------------------------------------------------
    def _map_segments(self):
        for seg in self.elf.iter_segments():
            if seg["p_type"] != "PT_LOAD":
                continue
            va, off = seg["p_vaddr"], seg["p_offset"]
            start, end = va & ~(PAGE - 1), _align_up(va + seg["p_memsz"])
            self.uc.mem_map(start, end - start)
            self.uc.mem_write(va, self.data[off:off + seg["p_filesz"]])

    def _setup_regions(self):
        self.uc.mem_map(STACK_TOP - STACK_SIZE, STACK_SIZE)
        self.uc.mem_map(SCRATCH, SCRATCH_SIZE)
        self.uc.mem_map(HEAP, HEAP_SIZE)
        self.uc.reg_write(UC_ARM_REG_SP, STACK_TOP - PAGE)

    def alloc(self, size: int) -> int:
        size = _align_up(max(size, 8), 8)
        p = self._heap_ptr
        self._heap_ptr += size
        self.uc.mem_write(p, b"\x00" * size)
        return p

    # ---- imports / PLT ------------------------------------------------------
    def _wire_imports(self):
        # ARM .plt = 0x14-byte resolver header + one 0xc-byte veneer per
        # .rel.plt entry. Intercept at the veneer entry rather than running the
        # ARM veneer (its blx/mode-switch corrupts our Thumb-only context).
        relplt = self.elf.get_section_by_name(".rel.plt")
        dynsym = self.elf.get_section_by_name(".dynsym")
        plt = self.elf.get_section_by_name(".plt")["sh_addr"]
        for i, r in enumerate(relplt.iter_relocations()):
            name = dynsym.get_symbol(r["r_info_sym"]).name
            self._imports[plt + 0x14 + i * 0xC] = name
        self.uc.hook_add(UC_HOOK_CODE, self._on_code)

    # ---- call helpers -------------------------------------------------------
    def stub(self, addr: int, mode: str = "identity"):
        """Short-circuit a direct-called helper: 'identity' returns `this` (r0),
        'ret0' returns 0. Used to isolate a function from data-driven callees."""
        self._stubs[addr & ~1] = mode

    def stop_at(self, *addrs: int):
        for a in addrs:
            self._stop_at.add(a & ~1)

    def ret(self, value: int = 0):
        # Emulate `bx lr` into a Thumb caller. After an ARM-targeting blx the CPU
        # is in ARM state; writing PC with bit0 set is what reliably restores
        # Thumb in Unicorn (toggling the CPSR T bit alone does not stick here).
        lr = self.uc.reg_read(UC_ARM_REG_LR)
        self.uc.reg_write(UC_ARM_REG_R0, value & 0xFFFFFFFF)
        self.uc.reg_write(UC_ARM_REG_CPSR, self.uc.reg_read(UC_ARM_REG_CPSR) | (1 << 5))
        self.uc.reg_write(UC_ARM_REG_PC, lr | 1)

    def call(self, addr: int, r0=0, r1=0, r2=0, ret_sentinel=0x4444_4444, count=500000):
        """Invoke a Thumb function at `addr`; returns when LR sentinel is hit."""
        self.stop_at(ret_sentinel)
        self.uc.reg_write(UC_ARM_REG_R0, r0)
        self.uc.reg_write(UC_ARM_REG_R1, r1)
        self.uc.reg_write(UC_ARM_REG_R2, r2)
        self.uc.reg_write(UC_ARM_REG_LR, ret_sentinel)
        self.uc.emu_start(addr | 1, ret_sentinel & ~1, count=count)
        return self.uc.reg_read(UC_ARM_REG_R0)

    def _on_code(self, uc, address, size, user):
        if self.trace:
            self.trace_log.append((address, self._disasm_one(address, size)))
        a = address & ~1
        if a in self._stop_at:
            uc.emu_stop()
            return
        name = self._imports.get(a)
        if name is not None:
            self._handle_import(name)
            return
        mode = self._stubs.get(a)
        if mode is not None:
            self.ret(uc.reg_read(UC_ARM_REG_R0) if mode == "identity" else 0)

    def _handle_import(self, name: str):
        uc = self.uc
        a0 = uc.reg_read(UC_ARM_REG_R0)
        a1 = uc.reg_read(UC_ARM_REG_R1)
        a2 = uc.reg_read(UC_ARM_REG_R2)
        self.import_calls.append(name)
        if name in ("_Znwj", "_Znaj", "malloc", "calloc"):
            self.ret(self.alloc(max(a0, 8)))
        elif name in ("_ZdlPv", "_ZdaPv", "free"):
            self.ret(0)
        elif name in ("__aeabi_memclr", "__aeabi_memclr4", "__aeabi_memclr8"):
            if a1 < 0x100000:
                uc.mem_write(a0, b"\x00" * a1)
            self.ret(a0)
        elif name in ("__aeabi_memset", "memset"):
            dst, n, c = (a0, a2, a1 & 0xFF) if name == "__aeabi_memset" else (a0, a2, a1 & 0xFF)
            if n < 0x100000:
                uc.mem_write(dst, bytes([c]) * n)
            self.ret(a0)
        elif name in ("__aeabi_memcpy", "__aeabi_memcpy4", "__aeabi_memcpy8", "memcpy", "memmove"):
            if a2 < 0x100000:
                uc.mem_write(a0, bytes(uc.mem_read(a1, a2)))
            self.ret(a0)
        else:
            self.ret(0)

    def _disasm_one(self, address, size):
        code = bytes(self.uc.mem_read(address, size))
        for i in self.md.disasm(code, address):
            return f"{i.mnemonic} {i.op_str}"
        return "?"
