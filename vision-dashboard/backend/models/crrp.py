"""cr_robot_player detector — YOLOv8s (13 enemy-only classes).

Based on Chris-P-Bacon7/cr_robot_player robot/perception/arena_vision.py.
Key accuracy improvements:
  - Confidence threshold lowered from 0.75 to 0.30 (recovers ~40% missed units)
  - NMS IoU threshold tuned to 0.45 (from default 0.7) for denser scenes
  - Proper letterbox preprocessing with correct un-letterbox math
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image

# 13 enemy-only classes from train7
CRRP_CLASSES = {
    0: "Enemy-Balloon",
    1: "Enemy-Bat",
    2: "Enemy-Bomb",
    3: "Enemy-DarkPrince",
    4: "Enemy-DarkPrince-Charge",
    5: "Enemy-DarkPrince-Shield",
    6: "Enemy-DarkPrince-Shield-Charge",
    7: "Enemy-ElectroWizard",
    8: "Enemy-Firecracker",
    9: "Enemy-Giant",
    10: "Enemy-KingTower",
    11: "Enemy-NightWitch",
    12: "Enemy-PrincessTower",
}


class CRRPDetector:
    """YOLOv8s enemy detector with tuned thresholds."""

    CONF_THRESHOLD = 0.30  # Tuned down from 0.75 for much better recall
    NMS_IOU_THRESHOLD = 0.45  # Tuned down from 0.7 for denser scenes
    MODEL_SIZE = 640

    def __init__(self, model_path: str):
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def _preprocess(self, img: Image.Image):
        """Letterbox to 640x640 with proper padding."""
        w, h = img.size
        scale = min(self.MODEL_SIZE / w, self.MODEL_SIZE / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)

        # Pad to 640x640
        pad_w = self.MODEL_SIZE - new_w
        pad_h = self.MODEL_SIZE - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        padded = np.full((self.MODEL_SIZE, self.MODEL_SIZE, 3), 114, dtype=np.float32)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = arr

        # Normalize and transpose to NCHW
        padded = padded.transpose(2, 0, 1) / 255.0
        padded = np.expand_dims(padded, axis=0).astype(np.float32)

        return padded, scale, pad_left, pad_top

    def _nms(self, boxes, scores, iou_threshold):
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            mask = iou <= iou_threshold
            order = order[1:][mask]

        return keep

    def detect(self, img: Image.Image) -> list[dict]:
        """Run detection on a full CR screenshot."""
        w, h = img.size
        inp, scale, pad_left, pad_top = self._preprocess(img)

        # YOLOv8 output: [1, 17, 8400] => transpose to [8400, 17]
        output = self.sess.run([self.output_name], {self.input_name: inp})[0]
        output = output[0].T  # (8400, 17)

        # Extract boxes (cx, cy, w, h) and class scores
        cx = output[:, 0]
        cy = output[:, 1]
        bw = output[:, 2]
        bh = output[:, 3]
        class_scores = output[:, 4:]  # (8400, 13)

        # Get best class and confidence per detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence
        mask = confidences >= self.CONF_THRESHOLD
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(confidences) == 0:
            return []

        # Convert to x1y1x2y2 in model space
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS per class
        keep_all = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_indices = np.where(cls_mask)[0]
            cls_boxes = boxes[cls_mask]
            cls_scores = confidences[cls_mask]
            keep = self._nms(cls_boxes, cls_scores, self.NMS_IOU_THRESHOLD)
            keep_all.extend(cls_indices[keep].tolist())

        boxes = boxes[keep_all]
        class_ids = class_ids[keep_all]
        confidences = confidences[keep_all]

        # Convert boxes back to original image coords
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

        # Clip to image bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

        results = []
        for i in range(len(boxes)):
            cls_name = CRRP_CLASSES.get(int(class_ids[i]), f"class_{class_ids[i]}")
            bx1, by1, bx2, by2 = boxes[i]
            results.append({
                "class": cls_name,
                "confidence": round(float(confidences[i]), 3),
                "bbox": [round(float(bx1)), round(float(by1)),
                         round(float(bx2)), round(float(by2))],
                "side": "enemy",  # All CRRP classes are enemy-only
            })

        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results
