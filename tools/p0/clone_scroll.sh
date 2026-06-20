#!/usr/bin/env bash
# Clone the upstream Scroll server (the public CR server emulator that loads
# libg.so). This is the base on which the P0 battle-bootstrap fix is applied.
#
# Upstream is read-only to us; the P0 fix is documented in docs/p0-findings.md.
#
# Usage:  tools/p0/clone_scroll.sh [OUTDIR]
set -euo pipefail

OUTDIR="${1:-.p0-engine/Scroll}"
SCROLL_URL="https://git.xeondev.com/Supercell/Scroll.git"
# Pinned to the tag 0.1 commit reviewed for the P0 findings.
SCROLL_COMMIT="d897141"

if [[ -d "$OUTDIR/.git" ]]; then
  echo "==> Scroll already cloned at $OUTDIR"
else
  echo "==> Cloning Scroll -> $OUTDIR"
  git clone "$SCROLL_URL" "$OUTDIR"
fi
git -C "$OUTDIR" checkout -q "$SCROLL_COMMIT" || true
echo "==> Scroll at $(git -C "$OUTDIR" rev-parse --short HEAD)"
echo "    Key files for P0:"
echo "      libserver/src/session/mod.rs   (start_mission — missing load_battle_state)"
echo "      libserver/src/logic/mode.rs    (LogicGameMode bindings)"
echo "      libserver/src/logic/battle.rs  (LogicBattle / set_location)"
