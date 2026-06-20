"""Pbatch/ClashRoyaleBuildABot detector — YOLOv10m (97 classes) + Side CNN.

Based on clashroyalebuildabot/detectors/unit_detector.py and side_detector.py.
Key accuracy improvements over original:
  - Confidence threshold lowered to 0.25 (from 0.30) for better recall
  - Proper letterbox preprocessing with exact un-letterbox math
  - Side CNN runs on every detection (not just possible allies)
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image

# 97 classes from Pbatch's YOLOv10m model
PBATCH_CLASSES = {
    0: "archer", 1: "archer_queen", 2: "balloon", 3: "bandit", 4: "barbarian",
    5: "barbarian_hut", 6: "bat", 7: "battle_healer", 8: "battle_ram",
    9: "bomb_tower", 10: "bomber", 11: "bowler", 12: "brawler", 13: "cannon",
    14: "cannon_cart", 15: "dark_prince", 16: "dart_goblin",
    17: "electro_dragon", 18: "electro_giant", 19: "electro_spirit",
    20: "electro_wizard", 21: "elite_barbarian", 22: "elixir_collector",
    23: "elixir_golem_large", 24: "elixir_golem_medium",
    25: "elixir_golem_small", 26: "executioner", 27: "fire_spirit",
    28: "firecracker", 29: "fisherman", 30: "flying_machine", 31: "furnace",
    32: "giant", 33: "giant_skeleton", 34: "giant_snowball", 35: "goblin",
    36: "goblin_cage", 37: "goblin_drill", 38: "goblin_hut",
    39: "golden_knight", 40: "golem", 41: "golemite", 42: "guard",
    43: "heal_spirit", 44: "hog", 45: "hog_rider", 46: "hungry_dragon",
    47: "hunter", 48: "ice_golem", 49: "ice_spirit", 50: "ice_wizard",
    51: "inferno_dragon", 52: "inferno_tower", 53: "knight", 54: "lava_hound",
    55: "lava_pup", 56: "little_prince", 57: "lumberjack", 58: "magic_archer",
    59: "mega_knight", 60: "mega_minion", 61: "mighty_miner", 62: "miner",
    63: "minion", 64: "minipekka", 65: "monk", 66: "mortar",
    67: "mother_witch", 68: "musketeer", 69: "night_witch", 70: "pekka",
    71: "phoenix_egg", 72: "phoenix_large", 73: "phoenix_small", 74: "prince",
    75: "princess", 76: "ram_rider", 77: "rascal_boy", 78: "rascal_girl",
    79: "royal_ghost", 80: "royal_giant", 81: "royal_guardian",
    82: "royal_hog", 83: "royal_recruit", 84: "skeleton",
    85: "skeleton_dragon", 86: "skeleton_king", 87: "sparky",
    88: "spear_goblin", 89: "tesla", 90: "tombstone", 91: "valkyrie",
    92: "wall_breaker", 93: "witch", 94: "wizard", 95: "x_bow", 96: "zappy",
}


class PbatchDetector:
    """YOLOv10m unit detector + 16x16 side CNN."""

    CONF_THRESHOLD = 0.20  # Tuned down from 0.30 for better recall
    MODEL_H = 480
    MODEL_W = 352
    SIDE_SIZE = 16
    # Crop the top 5% and bottom 20% of the image (arena region only)
    UNIT_Y_START = 0.05
    UNIT_Y_END = 0.80

    def __init__(self, unit_model: str, side_model: str):
        providers = ["CPUExecutionProvider"]
        self.unit_sess = ort.InferenceSession(unit_model, providers=providers)
        self.side_sess = ort.InferenceSession(side_model, providers=providers)

        unit_inp = self.unit_sess.get_inputs()[0]
        self.unit_input_name = unit_inp.name
        self.unit_output_name = self.unit_sess.get_outputs()[0].name

        side_inp = self.side_sess.get_inputs()[0]
        self.side_input_name = side_inp.name
        self.side_output_name = self.side_sess.get_outputs()[0].name

    def _preprocess(self, img: Image.Image):
        """Letterbox to 480x352, normalize. Applies arena crop only for portrait screenshots."""
        w, h = img.size
        # Only apply arena Y-crop if image is portrait (likely a full CR screenshot)
        if h / w > 1.3:
            crop_top = int(self.UNIT_Y_START * h)
            crop_bottom = int(self.UNIT_Y_END * h)
            cropped = img.crop((0, crop_top, w, crop_bottom))
        else:
            crop_top = 0
            cropped = img
        crop_w, crop_h = cropped.size

        # Compute resize maintaining aspect ratio
        ratio = crop_h / crop_w
        if ratio > self.MODEL_H / self.MODEL_W:
            new_h = self.MODEL_H
            new_w = int(self.MODEL_H / ratio)
        else:
            new_w = self.MODEL_W
            new_h = int(self.MODEL_W * ratio)

        resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)

        # Pad to model size
        dx = self.MODEL_W - new_w
        dy = self.MODEL_H - new_h
        pad_right = dx // 2
        pad_left = dx - pad_right
        pad_bottom = dy // 2
        pad_top = dy - pad_bottom

        arr = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=114,
        )

        # Transpose to CHW and normalize
        arr = arr.transpose(2, 0, 1) / 255.0
        arr = np.expand_dims(arr, axis=0).astype(np.float16)

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        return arr, padding, (crop_w, crop_h), crop_top

    def _fix_bboxes(self, preds, orig_w, orig_h, padding):
        """Convert model coords back to original image coords."""
        pad_left, pad_right, pad_top, pad_bottom = padding
        effective_w = self.MODEL_W - pad_left - pad_right
        effective_h = self.MODEL_H - pad_top - pad_bottom

        preds[:, 0] = (preds[:, 0] - pad_left) * orig_w / effective_w
        preds[:, 1] = (preds[:, 1] - pad_top) * orig_h / effective_h
        preds[:, 2] = (preds[:, 2] - pad_left) * orig_w / effective_w
        preds[:, 3] = (preds[:, 3] - pad_top) * orig_h / effective_h
        return preds

    def _classify_side(self, img: Image.Image, bbox):
        """Use 16x16 CNN to classify ally vs enemy."""
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            return "unknown"
        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((self.SIDE_SIZE, self.SIDE_SIZE), Image.Resampling.BICUBIC)
        arr = np.array(crop, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = self.side_sess.run([self.side_output_name], {self.side_input_name: arr})[0]
        return ("ally", "enemy")[np.argmax(pred[0])]

    def detect(self, img: Image.Image) -> list[dict]:
        """Run detection on a full CR screenshot."""
        w, h = img.size
        arr, padding, (crop_w, crop_h), crop_top = self._preprocess(img)

        # YOLOv10 output: [1, 300, 6] = [x1, y1, x2, y2, conf, cls]
        output = self.unit_sess.run(
            [self.unit_output_name], {self.unit_input_name: arr}
        )[0][0]  # shape: (300, 6)

        # Filter by confidence
        mask = output[:, 4] >= self.CONF_THRESHOLD
        preds = output[mask].copy()

        if len(preds) == 0:
            return []

        # Fix bounding boxes to cropped image coords
        preds[:, :4] = self._fix_bboxes(preds[:, :4].copy(), crop_w, crop_h, padding)

        results = []
        for det in preds:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            class_name = PBATCH_CLASSES.get(cls_id, f"class_{cls_id}")

            # Map bbox back to full image coords
            full_y1 = y1 + crop_top
            full_y2 = y2 + crop_top

            # Classify side
            side = self._classify_side(img, (x1, full_y1, x2, full_y2))

            results.append({
                "class": class_name,
                "confidence": round(float(conf), 3),
                "bbox": [round(float(x1)), round(float(full_y1)),
                         round(float(x2)), round(float(full_y2))],
                "side": side,
                "tile_x": round((x1 + x2) / 2 / crop_w * 18),
                "tile_y": round((y1 + y2) / 2 / crop_h * 32),
            })

        # Sort by confidence descending
        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results
