#!/usr/bin/env bash
# Download ONNX model weights for the CR Vision Dashboard.
# These are sourced from Pbatch/ClashRoyaleBuildABot and Chris-P-Bacon7/cr_robot_player.
set -euo pipefail

WEIGHTS_DIR="$(dirname "$0")/backend/weights"
mkdir -p "$WEIGHTS_DIR"

echo "=== Downloading Pbatch YOLOv10m unit model (30MB) ==="
if [ ! -f "$WEIGHTS_DIR/pbatch_units_M_480x352.onnx" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/Pbatch/ClashRoyaleBuildABot.git /tmp/pbatch_repo 2>/dev/null || true
    cd /tmp/pbatch_repo
    git sparse-checkout set clashroyalebuildabot/models
    cp clashroyalebuildabot/models/units_M_480x352.onnx "$WEIGHTS_DIR/pbatch_units_M_480x352.onnx"
    cp clashroyalebuildabot/models/side.onnx "$WEIGHTS_DIR/pbatch_side.onnx"
    cd -
    rm -rf /tmp/pbatch_repo
    echo "  Done."
else
    echo "  Already exists, skipping."
fi

echo "=== Downloading cr_robot_player YOLOv8s model (43MB) ==="
if [ ! -f "$WEIGHTS_DIR/crrp_train7_best.onnx" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/Chris-P-Bacon7/cr_robot_player.git /tmp/crrp_repo 2>/dev/null || true
    cd /tmp/crrp_repo
    git sparse-checkout set runs/detect/train7/weights
    cp runs/detect/train7/weights/best.onnx "$WEIGHTS_DIR/crrp_train7_best.onnx"
    cd -
    rm -rf /tmp/crrp_repo
    echo "  Done."
else
    echo "  Already exists, skipping."
fi

echo ""
echo "All weights downloaded to $WEIGHTS_DIR:"
ls -lh "$WEIGHTS_DIR"/*.onnx
