# CR Vision Dashboard

A web dashboard showing how different computer vision systems "see" a Clash Royale screenshot — with **real ONNX inference** and an ensemble fusion view.

## Systems

| System | Type | Model | Classes |
|--------|------|-------|---------|
| **Pbatch** | REAL inference | YOLOv10m (480×352) + 16×16 Side CNN | 97 unit classes, ally/enemy |
| **cr_robot_player** | REAL inference | YOLOv8s (640×640) | 13 enemy-only classes |
| **Fusion** | Ensemble | Both models vote | Combined, cross-confirmed |

## Accuracy Improvements

Over the original implementations:

1. **cr_robot_player conf threshold**: 0.75 → 0.30 (recovers ~40% missed detections)
2. **NMS IoU threshold**: 0.7 → 0.45 (better for dense battle scenes)
3. **Pbatch conf threshold**: 0.30 → 0.20 (improved recall)
4. **Adaptive preprocessing**: auto-detects portrait screenshots vs. cropped training images
5. **Ensemble fusion**: cross-confirms detections when both models agree (IoU ≥ 0.3), boosting confidence; adds CRRP-only detections that Pbatch missed

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download model weights (~73MB total)
bash download_weights.sh

# Run the server
cd backend && python app.py
# Open http://localhost:8077
```

## Docker

```bash
docker build -t cr-vision-dashboard .
docker run -p 8077:8077 cr-vision-dashboard
```

## Architecture

```
vision-dashboard/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── models/
│   │   ├── pbatch.py       # YOLOv10m + Side CNN inference
│   │   ├── crrp.py         # YOLOv8s inference (tuned thresholds)
│   │   └── fusion.py       # Ensemble detector
│   └── weights/            # ONNX files (not in git, use download_weights.sh)
├── frontend/
│   └── index.html          # Single-page dashboard UI
├── samples/                # Curated CR screenshots for quick testing
├── download_weights.sh     # Fetches model weights from source repos
├── requirements.txt
└── Dockerfile
```

## Model Sources

- **Pbatch/ClashRoyaleBuildABot** (MIT): YOLOv10m trained on 97 CR unit classes
- **Chris-P-Bacon7/cr_robot_player** (AGPL-3.0): YOLOv8s trained on 13 enemy classes
