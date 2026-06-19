"""Real-time game state perception from screen capture.

Extracts game state from Clash Royale screenshots using a combination of:
  1. YOLOv8 object detection (troops, buildings, spells)
  2. OCR for elixir bar reading
  3. Template matching for card hand detection
  4. Color-based tower HP estimation

Based on KataCR's perception pipeline (arXiv:2504.04783) but adapted for
our Entity Transformer input format.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from crsim.cards import CardType
from crsim.constants import ARENA_H, ARENA_W, NUM_CARD_TYPES

logger = logging.getLogger(__name__)

# Screen coordinate mapping (1080×1920 reference resolution)
SCREEN_W = 1080
SCREEN_H = 1920

# Arena bounds in screen pixels (approximate)
ARENA_LEFT = 60
ARENA_RIGHT = 1020
ARENA_TOP = 310
ARENA_BOTTOM = 1440

# Elixir bar region
ELIXIR_BAR_Y = 1550
ELIXIR_BAR_LEFT = 130
ELIXIR_BAR_RIGHT = 950

# Card hand region
CARD_HAND_Y = 1700
CARD_SLOTS_X = [220, 420, 620, 820]
CARD_SLOT_W = 140
CARD_SLOT_H = 170


@dataclass
class DetectedEntity:
    """A detected entity from screen perception."""
    class_name: str
    card_type: CardType | None
    owner: int  # 0=friendly, 1=enemy, -1=unknown
    screen_x: float
    screen_y: float
    arena_x: float  # tile x
    arena_y: float  # tile y
    confidence: float
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)  # x1,y1,x2,y2
    estimated_hp_frac: float = 1.0


@dataclass
class PerceivedGameState:
    """Full game state extracted from screen capture."""
    entities: list[DetectedEntity] = field(default_factory=list)
    elixir: float = 5.0
    hand_cards: list[CardType | None] = field(default_factory=lambda: [None] * 4)
    next_card: CardType | None = None
    my_tower_hp: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    opp_tower_hp: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    game_time_remaining: float = 180.0
    is_overtime: bool = False
    timestamp: float = 0.0


# Class name → CardType mapping for YOLOv8 detections
CLASS_TO_CARD: dict[str, CardType] = {
    "knight": CardType.KNIGHT,
    "archers": CardType.ARCHERS,
    "musketeer": CardType.MUSKETEER,
    "giant": CardType.GIANT,
    "mini_pekka": CardType.MINI_PEKKA,
    "valkyrie": CardType.VALKYRIE,
    "wizard": CardType.WIZARD,
    "hog_rider": CardType.HOG_RIDER,
    "minions": CardType.MINIONS,
    "baby_dragon": CardType.BABY_DRAGON,
    "skeleton_army": CardType.SKELETON_ARMY,
    "goblin_barrel": CardType.GOBLIN_BARREL,
    "prince": CardType.PRINCE,
    "dark_prince": CardType.DARK_PRINCE,
    "pekka": CardType.PEKKA,
    "golem": CardType.GOLEM,
    "lava_hound": CardType.LAVA_HOUND,
    "balloon": CardType.BALLOON,
    "witch": CardType.WITCH,
    "ice_wizard": CardType.ICE_WIZARD,
    "electro_wizard": CardType.ELECTRO_WIZARD,
    "bandit": CardType.BANDIT,
    "mega_knight": CardType.MEGA_KNIGHT,
    "sparky": CardType.SPARKY,
    "guards": CardType.GUARDS,
    "goblins": CardType.GOBLINS,
    "spear_goblins": CardType.SPEAR_GOBLINS,
    "fire_spirits": CardType.FIRE_SPIRITS,
    "ice_spirit": CardType.ICE_SPIRIT,
    "cannon": CardType.CANNON,
    "inferno_tower": CardType.INFERNO_TOWER,
    "tombstone": CardType.TOMBSTONE,
    "tesla": CardType.TESLA,
    "elixir_collector": CardType.ELIXIR_COLLECTOR,
}


def screen_to_arena(sx: float, sy: float) -> tuple[float, float]:
    """Convert screen pixel coordinates to arena tile coordinates."""
    ax = (sx - ARENA_LEFT) / (ARENA_RIGHT - ARENA_LEFT) * ARENA_W
    ay = (sy - ARENA_TOP) / (ARENA_BOTTOM - ARENA_TOP) * ARENA_H
    ax = max(0.0, min(float(ARENA_W - 1), ax))
    ay = max(0.0, min(float(ARENA_H - 1), ay))
    return ax, ay


def arena_to_screen(ax: float, ay: float) -> tuple[int, int]:
    """Convert arena tile coordinates to screen pixel coordinates."""
    sx = int(ARENA_LEFT + (ax / ARENA_W) * (ARENA_RIGHT - ARENA_LEFT))
    sy = int(ARENA_TOP + (ay / ARENA_H) * (ARENA_BOTTOM - ARENA_TOP))
    return sx, sy


class GamePerception:
    """Extract game state from screen captures using CV models.

    Usage:
        perception = GamePerception()
        perception.load_models("models/yolov8_cr.pt")
        state = perception.process_frame(screenshot_array)
    """

    def __init__(self) -> None:
        self._yolo_model = None
        self._card_templates: dict[str, np.ndarray] = {}
        self._ready = False

    def load_models(
        self,
        yolo_weights: str = "models/yolov8_cr.pt",
        card_templates_dir: str = "models/card_templates/",
    ) -> None:
        """Load YOLOv8 model and card templates."""
        yolo_path = Path(yolo_weights)
        if yolo_path.exists():
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(str(yolo_path))
                logger.info("Loaded YOLOv8 model from %s", yolo_path)
            except ImportError:
                logger.warning(
                    "ultralytics not installed — using fallback detection"
                )
        else:
            logger.warning("YOLOv8 weights not found at %s", yolo_path)

        templates_path = Path(card_templates_dir)
        if templates_path.is_dir():
            for img_path in templates_path.glob("*.png"):
                try:
                    import cv2
                    tmpl = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if tmpl is not None:
                        self._card_templates[img_path.stem] = tmpl
                except ImportError:
                    break
            if self._card_templates:
                logger.info(
                    "Loaded %d card templates", len(self._card_templates)
                )

        self._ready = True

    def process_frame(self, frame: np.ndarray) -> PerceivedGameState:
        """Process a single screenshot and extract game state.

        Parameters
        ----------
        frame : ndarray, shape (H, W, 3) BGR
            Screenshot of the game.

        Returns
        -------
        PerceivedGameState with detected entities, elixir, hand, etc.
        """
        state = PerceivedGameState(timestamp=time.time())

        # 1. Detect entities via YOLOv8
        if self._yolo_model is not None:
            state.entities = self._detect_entities_yolo(frame)
        else:
            state.entities = self._detect_entities_fallback(frame)

        # 2. Read elixir bar
        state.elixir = self._read_elixir(frame)

        # 3. Detect hand cards
        state.hand_cards = self._detect_hand(frame)

        # 4. Estimate tower HP from health bars
        state.my_tower_hp, state.opp_tower_hp = self._estimate_tower_hp(frame)

        return state

    def _detect_entities_yolo(
        self, frame: np.ndarray,
    ) -> list[DetectedEntity]:
        """Detect entities using YOLOv8."""
        results = self._yolo_model(frame, verbose=False, conf=0.3)
        entities = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, "unknown")

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                ax, ay = screen_to_arena(cx, cy)

                # Determine owner: top half = enemy, bottom half = friendly
                owner = 0 if cy > (ARENA_TOP + ARENA_BOTTOM) / 2 else 1

                card_type = CLASS_TO_CARD.get(cls_name.lower())

                entities.append(DetectedEntity(
                    class_name=cls_name,
                    card_type=card_type,
                    owner=owner,
                    screen_x=cx,
                    screen_y=cy,
                    arena_x=ax,
                    arena_y=ay,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))

        return entities

    def _detect_entities_fallback(
        self, frame: np.ndarray,
    ) -> list[DetectedEntity]:
        """Fallback entity detection using color segmentation."""
        entities = []
        h, w = frame.shape[:2]

        try:
            import cv2
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect blue (friendly) entities
            blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
            # Detect red (enemy) entities
            red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
            red_mask = red_mask1 | red_mask2

            for owner, mask in [(0, blue_mask), (1, red_mask)]:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 200:
                        continue
                    mx, my, mw, mh = cv2.boundingRect(cnt)
                    cx = mx + mw / 2
                    cy = my + mh / 2

                    if cy < ARENA_TOP or cy > ARENA_BOTTOM:
                        continue

                    ax, ay = screen_to_arena(cx, cy)
                    entities.append(DetectedEntity(
                        class_name="unknown",
                        card_type=None,
                        owner=owner,
                        screen_x=cx,
                        screen_y=cy,
                        arena_x=ax,
                        arena_y=ay,
                        confidence=0.5,
                        bbox=(mx, my, mx + mw, my + mh),
                    ))
        except ImportError:
            pass

        return entities

    def _read_elixir(self, frame: np.ndarray) -> float:
        """Read elixir count from the purple elixir bar."""
        try:
            import cv2
            # Crop elixir bar region
            bar_region = frame[
                ELIXIR_BAR_Y - 10:ELIXIR_BAR_Y + 10,
                ELIXIR_BAR_LEFT:ELIXIR_BAR_RIGHT,
            ]
            hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
            # Purple mask for elixir
            purple_mask = cv2.inRange(hsv, (120, 50, 50), (160, 255, 255))
            fill_ratio = np.count_nonzero(purple_mask) / max(purple_mask.size, 1)
            return fill_ratio * 10.0
        except (ImportError, Exception):
            return 5.0

    def _detect_hand(self, frame: np.ndarray) -> list[CardType | None]:
        """Detect which cards are in the player's hand."""
        hand = [None] * 4
        if not self._card_templates:
            return hand

        try:
            import cv2
            for slot_idx, slot_x in enumerate(CARD_SLOTS_X):
                region = frame[
                    CARD_HAND_Y:CARD_HAND_Y + CARD_SLOT_H,
                    slot_x:slot_x + CARD_SLOT_W,
                ]
                best_match = None
                best_score = 0.0

                for card_name, tmpl in self._card_templates.items():
                    tmpl_resized = cv2.resize(tmpl, (CARD_SLOT_W, CARD_SLOT_H))
                    result = cv2.matchTemplate(
                        region, tmpl_resized, cv2.TM_CCOEFF_NORMED,
                    )
                    score = float(result.max())
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match = card_name

                if best_match:
                    hand[slot_idx] = CLASS_TO_CARD.get(best_match.lower())
        except (ImportError, Exception):
            pass

        return hand

    def _estimate_tower_hp(
        self, frame: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        """Estimate tower HP from health bar overlays.

        Returns (my_towers, opp_towers) each [king, left_princess, right_princess]
        as fractions [0, 1].
        """
        my_hp = [1.0, 1.0, 1.0]
        opp_hp = [1.0, 1.0, 1.0]

        # Tower screen positions (approximate for 1080×1920)
        tower_positions = {
            "my_king": (540, 1380),
            "my_left": (240, 1200),
            "my_right": (840, 1200),
            "opp_king": (540, 380),
            "opp_left": (240, 540),
            "opp_right": (840, 540),
        }

        try:
            import cv2
            for name, (tx, ty) in tower_positions.items():
                # Look for health bar above tower
                bar_region = frame[
                    max(0, ty - 30):ty - 10,
                    max(0, tx - 40):min(frame.shape[1], tx + 40),
                ]
                if bar_region.size == 0:
                    continue

                hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
                # Green health bar
                green_mask = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))
                # Red health bar
                red_mask = cv2.inRange(hsv, (0, 80, 80), (15, 255, 255))

                green_pixels = np.count_nonzero(green_mask)
                red_pixels = np.count_nonzero(red_mask)
                total = green_pixels + red_pixels

                hp_frac = green_pixels / max(total, 1) if total > 10 else 1.0

                if name.startswith("my_"):
                    idx = ["my_king", "my_left", "my_right"].index(name)
                    my_hp[idx] = hp_frac
                else:
                    idx = ["opp_king", "opp_left", "opp_right"].index(name)
                    opp_hp[idx] = hp_frac
        except (ImportError, Exception):
            pass

        return my_hp, opp_hp

    def perceived_to_features(
        self, state: PerceivedGameState, player: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert PerceivedGameState to neural network input tensors.

        Returns (spatial, scalar, entity_features, entity_mask) matching
        the format expected by CRStarNet.
        """
        from model.features import (
            ENTITY_FEATURE_DIM,
            MAX_ENTITY_SLOTS,
        )

        spatial = np.zeros(
            (2 * NUM_CARD_TYPES + 4, ARENA_H, ARENA_W), dtype=np.float32,
        )
        scalar = np.zeros(2 + 5 * NUM_CARD_TYPES + 14, dtype=np.float32)
        entity_feats = np.zeros(
            (MAX_ENTITY_SLOTS, ENTITY_FEATURE_DIM), dtype=np.float32,
        )
        entity_mask = np.zeros(MAX_ENTITY_SLOTS, dtype=bool)

        # Fill entity features from detections
        for i, det in enumerate(state.entities):
            if i >= MAX_ENTITY_SLOTS:
                break
            entity_mask[i] = True

            ct = det.card_type
            if ct is not None:
                entity_feats[i, 0] = float(ct) / NUM_CARD_TYPES
            entity_feats[i, 1] = 1.0 if det.owner == player else 0.0
            entity_feats[i, 2] = det.arena_x / ARENA_W
            entity_feats[i, 3] = det.arena_y / ARENA_H
            entity_feats[i, 4] = det.estimated_hp_frac
            entity_feats[i, 5] = det.confidence

            # Fill spatial channels
            if ct is not None and 0 <= int(ct) < NUM_CARD_TYPES:
                is_friendly = det.owner == player
                chan = int(ct) if is_friendly else NUM_CARD_TYPES + int(ct)
                ix = int(round(det.arena_x))
                iy = int(round(det.arena_y))
                ix = max(0, min(ARENA_W - 1, ix))
                iy = max(0, min(ARENA_H - 1, iy))
                spatial[chan, iy, ix] += det.estimated_hp_frac

        # Scalar features
        idx = 0
        scalar[idx] = state.elixir / 10.0
        idx += 1
        scalar[idx] = 1.0 if state.is_overtime else 0.0
        idx += 1

        # Hand cards
        for slot in range(4):
            ct = state.hand_cards[slot] if slot < len(state.hand_cards) else None
            if ct is not None and 0 <= int(ct) < NUM_CARD_TYPES:
                scalar[idx + int(ct)] = 1.0
            idx += NUM_CARD_TYPES

        # Next card
        if state.next_card is not None and 0 <= int(state.next_card) < NUM_CARD_TYPES:
            scalar[idx + int(state.next_card)] = 1.0
        idx += NUM_CARD_TYPES

        # Time
        scalar[idx] = max(0.0, state.game_time_remaining / 180.0)
        idx += 1

        # Tower status
        for towers in [state.my_tower_hp, state.opp_tower_hp]:
            for i in range(3):
                scalar[idx] = 1.0 if towers[i] > 0.01 else 0.0
                idx += 1
        # Tower HP
        for towers in [state.my_tower_hp, state.opp_tower_hp]:
            for i in range(3):
                scalar[idx] = towers[i]
                idx += 1

        # Crown diff
        my_crowns = sum(1 for hp in state.opp_tower_hp if hp <= 0.01)
        opp_crowns = sum(1 for hp in state.my_tower_hp if hp <= 0.01)
        scalar[idx] = (my_crowns - opp_crowns) / 3.0

        return spatial, scalar, entity_feats, entity_mask
