"""Fusion/Ensemble detector — combines Pbatch and CRRP for better accuracy.

Strategy:
  - Pbatch is primary (97 classes, both sides)
  - CRRP cross-confirms enemy detections (13 enemy-only classes)
  - When both models detect the same object (IoU >= 0.3), confidence is boosted
  - CRRP detections that don't overlap with Pbatch are added (may catch misses)
  - Disagreements are flagged for transparency
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from models.pbatch import PbatchDetector
from models.crrp import CRRPDetector

# Mapping from CRRP class names to approximate Pbatch equivalents
CRRP_TO_PBATCH_MAP = {
    "Enemy-Balloon": "balloon",
    "Enemy-Bat": "bat",
    "Enemy-Bomb": "bomber",  # bomb = bomber's projectile, close enough
    "Enemy-DarkPrince": "dark_prince",
    "Enemy-DarkPrince-Charge": "dark_prince",
    "Enemy-DarkPrince-Shield": "dark_prince",
    "Enemy-DarkPrince-Shield-Charge": "dark_prince",
    "Enemy-ElectroWizard": "electro_wizard",
    "Enemy-Firecracker": "firecracker",
    "Enemy-Giant": "giant",
    "Enemy-KingTower": None,  # Tower, not a unit
    "Enemy-NightWitch": "night_witch",
    "Enemy-PrincessTower": None,  # Tower
}


def _iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


class FusionDetector:
    """Ensemble that fuses Pbatch + CRRP detections."""

    MATCH_IOU_THRESHOLD = 0.3
    BOOST_FACTOR = 1.15  # Confidence boost when both models agree
    MAX_CONF = 0.99

    def __init__(self, pbatch: PbatchDetector, crrp: CRRPDetector):
        self.pbatch = pbatch
        self.crrp = crrp

    def detect(self, img: Image.Image) -> list[dict]:
        """Run both detectors and fuse results."""
        pbatch_dets = self.pbatch.detect(img)
        crrp_dets = self.crrp.detect(img)

        # Start with Pbatch detections as base
        fused = []
        crrp_matched = set()

        for p_det in pbatch_dets:
            det = p_det.copy()
            det["source"] = "pbatch"
            det["confirmed_by"] = None

            # Check if any CRRP detection overlaps
            for j, c_det in enumerate(crrp_dets):
                if j in crrp_matched:
                    continue
                iou = _iou(p_det["bbox"], c_det["bbox"])
                if iou >= self.MATCH_IOU_THRESHOLD:
                    # Cross-confirmed! Boost confidence
                    boosted = min(
                        p_det["confidence"] * self.BOOST_FACTOR,
                        self.MAX_CONF,
                    )
                    det["confidence"] = round(boosted, 3)
                    det["confirmed_by"] = c_det["class"]
                    det["side"] = "enemy"  # CRRP confirms enemy
                    crrp_matched.add(j)
                    break

            fused.append(det)

        # Add unmatched CRRP detections (Pbatch missed these)
        for j, c_det in enumerate(crrp_dets):
            if j not in crrp_matched:
                det = c_det.copy()
                det["source"] = "crrp_only"
                det["confirmed_by"] = None
                # Map CRRP class to a normalized name
                pbatch_equiv = CRRP_TO_PBATCH_MAP.get(c_det["class"])
                if pbatch_equiv:
                    det["class_normalized"] = pbatch_equiv
                fused.append(det)

        fused.sort(key=lambda d: d["confidence"], reverse=True)
        return fused
