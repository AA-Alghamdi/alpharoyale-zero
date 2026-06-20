#!/usr/bin/env bash
# Reproducibly fetch + verify the Clash Royale v1.3.2 engine binary (libg.so)
# used for P0 (real-engine fidelity). The proprietary binary is NOT committed to
# this repo; this script downloads it from a public archive and verifies it byte
# for byte, then statically validates it against the P0 RE fingerprints.
#
# Usage:  tools/p0/fetch_and_verify_engine.sh [OUTDIR]
# Default OUTDIR: ./.p0-engine  (gitignored)
set -euo pipefail

OUTDIR="${1:-.p0-engine}"
APK_URL="https://archive.org/download/ClashRoyale1.3.2Lastapk.com/Clash_Royale_1.3.2_lastapk.com.apk"
APK_SHA256="551eff29a8d151a147c9885fc936c71c667f7212935d22d23b001bf672329046"
LIBG_ARM_SHA256="f73b70d714d1423c6a8ef520982ce49d870e2db027021ebdffdd755efece6a80"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTDIR"
apk="$OUTDIR/cr-1.3.2.apk"

echo "==> Downloading CR v1.3.2 APK (95.6 MB) -> $apk"
if [[ ! -f "$apk" ]]; then
  curl -fSL --retry 3 -o "$apk" "$APK_URL"
fi

echo "==> Verifying APK sha256"
echo "${APK_SHA256}  ${apk}" | sha256sum -c -

echo "==> Extracting libg.so (armeabi-v7a = the ARM RE target; x86 also extracted)"
python3 - "$apk" "$OUTDIR" <<'PY'
import sys, zipfile, os
apk, outdir = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(apk)
for arch in ("armeabi-v7a", "x86"):
    name = f"lib/{arch}/libg.so"
    data = z.read(name)
    out = os.path.join(outdir, f"libg-{arch}.so")
    open(out, "wb").write(data)
    print(f"   {out}  ({len(data):,} bytes)")
PY

echo "==> Verifying armeabi-v7a libg.so sha256"
echo "${LIBG_ARM_SHA256}  ${OUTDIR}/libg-armeabi-v7a.so" | sha256sum -c -

echo "==> Static fingerprint validation (capstone/pyelftools required)"
python3 "$here/validate_libg.py" "$OUTDIR/libg-armeabi-v7a.so"

echo
echo "OK: verified CR v1.3.2 engine at $OUTDIR/libg-armeabi-v7a.so"
