"""CR Vision Dashboard — FastAPI backend.

Shows how multiple CV systems "see" a Clash Royale screenshot.
Real inference via Pbatch (YOLOv10m) and cr_robot_player (YOLOv8s),
plus a fusion/ensemble view that combines both for better accuracy.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from models.pbatch import PbatchDetector
from models.crrp import CRRPDetector
from models.fusion import FusionDetector

app = FastAPI(title="CR Vision Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
SAMPLES_DIR = BASE_DIR / "samples"
MODELS_DIR = BASE_DIR / "backend" / "weights"

# Initialize detectors (lazy-loaded on first call)
pbatch: PbatchDetector | None = None
crrp: CRRPDetector | None = None
fusion: FusionDetector | None = None


def get_detectors():
    global pbatch, crrp, fusion
    if pbatch is None:
        pbatch = PbatchDetector(
            unit_model=str(MODELS_DIR / "pbatch_units_M_480x352.onnx"),
            side_model=str(MODELS_DIR / "pbatch_side.onnx"),
        )
    if crrp is None:
        crrp = CRRPDetector(
            model_path=str(MODELS_DIR / "crrp_train7_best.onnx"),
        )
    if fusion is None:
        fusion = FusionDetector(pbatch, crrp)
    return pbatch, crrp, fusion


@app.get("/api/samples")
def list_samples():
    """List available sample images."""
    samples = []
    if SAMPLES_DIR.exists():
        for f in sorted(SAMPLES_DIR.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append({"name": f.stem, "filename": f.name})
    return {"samples": samples}


@app.get("/api/samples/{filename}")
def get_sample(filename: str):
    """Serve a sample image."""
    path = SAMPLES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.post("/api/infer")
async def infer(file: UploadFile = File(None), sample: str | None = Form(None)):
    """Run inference on an uploaded or sample image.

    Returns detections from all systems.
    """
    pbatch_det, crrp_det, fusion_det = get_detectors()

    # Load image
    img = None
    if file and file.filename and file.size and file.size > 0:
        img_bytes = await file.read()
        if len(img_bytes) > 0:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if img is None and sample:
        path = SAMPLES_DIR / sample
        if not path.exists():
            return JSONResponse({"error": "sample not found"}, status_code=404)
        img = Image.open(path).convert("RGB")
    if img is None:
        return JSONResponse({"error": "no image provided"}, status_code=400)

    w, h = img.size

    # Run all detectors
    t0 = time.time()
    pbatch_results = pbatch_det.detect(img)
    t_pbatch = time.time() - t0

    t0 = time.time()
    crrp_results = crrp_det.detect(img)
    t_crrp = time.time() - t0

    t0 = time.time()
    fusion_results = fusion_det.detect(img)
    t_fusion = time.time() - t0

    # Encode image as base64 for frontend display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image": img_b64,
        "width": w,
        "height": h,
        "systems": {
            "pbatch": {
                "name": "Pbatch (YOLOv10m + Side CNN)",
                "badge": "REAL",
                "detections": pbatch_results,
                "time_ms": round(t_pbatch * 1000, 1),
                "description": "YOLOv10m (97 classes) at 480x352 + 16x16 side CNN for ally/enemy. Conf >= 0.25.",
                "source": "Pbatch/ClashRoyaleBuildABot",
            },
            "crrp": {
                "name": "cr_robot_player (YOLOv8s)",
                "badge": "REAL",
                "detections": crrp_results,
                "time_ms": round(t_crrp * 1000, 1),
                "description": "YOLOv8s (13 enemy-only classes) at 640x640. Conf >= 0.30 (tuned down from 0.75).",
                "source": "Chris-P-Bacon7/cr_robot_player",
            },
            "fusion": {
                "name": "Fusion (Both Models Vote)",
                "badge": "ENSEMBLE",
                "detections": fusion_results,
                "time_ms": round(t_fusion * 1000, 1),
                "description": "Ensemble: Pbatch primary + CRRP cross-confirms enemies. IoU >= 0.3 merges boost confidence.",
                "source": "Combined",
            },
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": pbatch is not None}


# Serve frontend
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8077)
