"""Real-device pipeline for sim-to-real deployment.

Architecture (adapted from KataCR):
    Phone Screen → scrcpy → PC Frame Buffer
                                  ↓
                  ┌────────────────┼────────────────┐
                  ↓                ↓                ↓
             YOLOv8 x2-3      PaddleOCR        ResNet
             (Troops,         (Tower HP,      (Card hand,
              Spells,          Timer)          Elixir)
              Buildings)
                  ↓                ↓                ↓
                  └────────────────┼────────────────┘
                                  ↓
                        StateBuilder (perception fusion)
                        → State vector (same as sim format)
                                  ↓
                        CRStarNet + Gumbel MuZero
                                  ↓
                        ADB tap at (screen_x, screen_y)

Reference: KataCR (github.com/wty-yy/KataCR) — the only CR AI that
actually beat strong opponents on a real device.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    ANDROID_PHONE = auto()
    BLUESTACKS = auto()
    LDPLAYER = auto()


@dataclass
class DeviceConfig:
    """Configuration for real-device connection."""

    device_type: DeviceType = DeviceType.BLUESTACKS
    adb_serial: str = "127.0.0.1:5555"  # BlueStacks default
    scrcpy_port: int = 27183
    screen_width: int = 720  # CR on mobile
    screen_height: int = 1280

    # Game area within screen (may need calibration)
    arena_left: int = 67
    arena_top: int = 215
    arena_right: int = 653
    arena_bottom: int = 1030

    # Card tray positions (x coordinates for 4 hand slots)
    card_tray_y: int = 1170
    card_slot_xs: list[int] = field(default_factory=lambda: [196, 306, 416, 526])

    # FPS target
    target_fps: float = 5.0


@dataclass
class PerceptionConfig:
    """Configuration for visual perception pipeline."""

    yolo_model_path: str = "models/yolov8_cr.pt"
    yolo_conf_threshold: float = 0.5
    ocr_model: str = "paddleocr"
    card_classifier_path: str = "models/card_resnet.pt"
    elixir_classifier_path: str = "models/elixir_resnet.pt"
    num_classes: int = 150  # KataCR uses 150 entity classes


class ADBController:
    """Sends tap commands to the game via ADB.

    Converts game-space coordinates (tiles) to screen-space pixels.
    """

    def __init__(self, config: DeviceConfig) -> None:
        self.config = config
        self._connected = False

    def connect(self) -> bool:
        try:
            result = subprocess.run(
                ["adb", "connect", self.config.adb_serial],
                capture_output=True, text=True, timeout=10,
            )
            self._connected = "connected" in result.stdout.lower()
            if self._connected:
                logger.info(f"Connected to {self.config.adb_serial}")
            return self._connected
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("ADB not available")
            return False

    def tap(self, screen_x: int, screen_y: int) -> None:
        """Tap at screen coordinates."""
        if not self._connected:
            logger.warning("Not connected to device")
            return
        subprocess.run(
            ["adb", "-s", self.config.adb_serial, "shell",
             "input", "tap", str(screen_x), str(screen_y)],
            timeout=5,
        )

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        """Swipe gesture (for card dragging)."""
        if not self._connected:
            return
        subprocess.run(
            ["adb", "-s", self.config.adb_serial, "shell",
             "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
            timeout=5,
        )

    def play_card(self, hand_slot: int, tile_x: float, tile_y: float) -> None:
        """Play a card from hand to a tile position.

        Converts tile coordinates to screen coordinates and performs
        a drag from the card slot to the target position.
        """
        cfg = self.config

        # Source: card slot position
        src_x = cfg.card_slot_xs[hand_slot]
        src_y = cfg.card_tray_y

        # Target: convert tile to screen
        arena_w = cfg.arena_right - cfg.arena_left
        arena_h = cfg.arena_bottom - cfg.arena_top
        dst_x = int(cfg.arena_left + (tile_x / 18.0) * arena_w)
        dst_y = int(cfg.arena_bottom - (tile_y / 32.0) * arena_h)  # y is inverted

        self.swipe(src_x, src_y, dst_x, dst_y, duration_ms=150)

    def screenshot(self) -> np.ndarray | None:
        """Capture a screenshot via ADB."""
        try:
            result = subprocess.run(
                ["adb", "-s", self.config.adb_serial, "exec-out",
                 "screencap", "-p"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                import io

                from PIL import Image
                img = Image.open(io.BytesIO(result.stdout))
                return np.array(img)
        except Exception:
            logger.exception("Screenshot failed")
        return None


class StateBuilder:
    """Fuses perception outputs into a state vector matching sim format.

    Takes outputs from YOLOv8, PaddleOCR, and ResNet classifiers,
    and produces the same state representation that our simulator uses.
    This allows the same CRStarNet model to work with both sim and real data.
    """

    def __init__(self, num_card_types: int = 121, max_entities: int = 128) -> None:
        self.num_card_types = num_card_types
        self.max_entities = max_entities

        # Class ID to CardType mapping (needs calibration per YOLO model)
        self.class_to_card: dict[int, int] = {}

    def build_state(
        self,
        detections: list[dict],  # [{class_id, x, y, w, h, conf}]
        tower_hps: dict[str, float],  # {p0_king, p0_left, p0_right, p1_king, p1_left, p1_right}
        hand_cards: list[int],  # 4 card type IDs
        elixir: float,
        tick: int,
    ) -> np.ndarray:
        """Build state vector from perception data.

        Returns the same format as CRGame.encode_state() so the
        trained model works without any architecture changes.
        """
        # Entity features: same format as sim
        entity_features = np.zeros((self.max_entities, 12), dtype=np.float32)

        # Tower entities (indices 0-5)
        tower_positions = [
            ("p0_king", 9.0, 1.0, True),
            ("p0_left", 4.0, 4.0, True),
            ("p0_right", 13.0, 4.0, True),
            ("p1_king", 9.0, 30.0, False),
            ("p1_left", 4.0, 27.0, False),
            ("p1_right", 13.0, 27.0, False),
        ]
        for i, (key, tx, ty, is_friendly) in enumerate(tower_positions):
            hp = tower_hps.get(key, 0)
            if hp > 0:
                entity_features[i] = [
                    tx / 18.0, ty / 32.0,  # normalized position
                    hp / 4008.0,  # normalized HP
                    1.0 if is_friendly else 0.0,  # team
                    0.0,  # card type (tower)
                    1.0,  # is_building
                    0.0,  # is_flying
                    0.0,  # speed
                    0.0, 0.0, 0.0, 1.0,  # padding + alive flag
                ]

        # Detected entities
        for j, det in enumerate(detections[:self.max_entities - 6]):
            idx = j + 6
            card_id = self.class_to_card.get(det["class_id"], 0)
            is_friendly = det.get("is_friendly", det["y"] > 0.5)

            entity_features[idx] = [
                det["x"], det["y"],  # normalized position
                1.0,  # HP unknown from vision, assume full
                1.0 if is_friendly else 0.0,
                card_id / self.num_card_types,
                0.0,  # is_building
                det.get("is_flying", 0.0),
                0.0, 0.0, 0.0, 0.0, 1.0,  # alive
            ]

        # Scalar features: elixir, hand, tick
        n_scalar = 20 + 4 * self.num_card_types
        scalars = np.zeros(n_scalar, dtype=np.float32)
        scalars[0] = elixir / 10.0
        scalars[1] = tick / 10800.0  # normalize by max ticks
        for i, card in enumerate(hand_cards[:4]):
            if 0 <= card < self.num_card_types:
                scalars[4 + i * self.num_card_types + card] = 1.0

        return np.concatenate([entity_features.flatten(), scalars])


class RealDevicePipeline:
    """End-to-end pipeline for playing CR on a real device.

    Integrates perception, decision-making, and control.
    """

    def __init__(
        self,
        model,
        device_config: DeviceConfig | None = None,
        perception_config: PerceptionConfig | None = None,
    ) -> None:
        self.model = model
        self.device_config = device_config or DeviceConfig()
        self.perception_config = perception_config or PerceptionConfig()
        self.controller = ADBController(self.device_config)
        self.state_builder = StateBuilder()
        self._yolo_model = None
        self._running = False

    def setup(self) -> bool:
        """Initialize all components."""
        if not self.controller.connect():
            logger.error("Failed to connect to device")
            return False

        # Load YOLO model (deferred to avoid import issues)
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.perception_config.yolo_model_path)
            logger.info("YOLO model loaded")
        except ImportError:
            logger.warning("ultralytics not installed — YOLO detection disabled")
        except Exception as e:
            logger.warning(f"Failed to load YOLO: {e}")

        return True

    def run_game_loop(self, max_steps: int = 10000) -> dict:
        """Run the main game loop: perceive → decide → act.

        Returns game statistics.
        """
        self._running = True
        stats = {"actions_taken": 0, "fps_history": []}
        target_dt = 1.0 / self.device_config.target_fps

        for step in range(max_steps):
            if not self._running:
                break

            t0 = time.time()

            # 1. Capture screenshot
            frame = self.controller.screenshot()
            if frame is None:
                continue

            # 2. Run perception
            detections = self._detect_entities(frame)
            tower_hps = self._read_tower_hps(frame)
            hand_cards = self._classify_hand(frame)
            elixir = self._read_elixir(frame)

            # 3. Build state
            state = self.state_builder.build_state(
                detections=detections,
                tower_hps=tower_hps,
                hand_cards=hand_cards,
                elixir=elixir,
                tick=step,
            )

            # 4. Get action from model
            import torch
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
                action = policy_logits.argmax(dim=-1).item()

            # 5. Execute action
            if action < 4 * 18 * 32:  # card play
                hand_slot = action // (18 * 32)
                remaining = action % (18 * 32)
                tile_x = remaining % 18
                tile_y = remaining // 18
                self.controller.play_card(hand_slot, tile_x + 0.5, tile_y + 0.5)
                stats["actions_taken"] += 1

            # 6. Timing
            elapsed = time.time() - t0
            stats["fps_history"].append(1.0 / max(elapsed, 0.001))
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        return stats

    def stop(self) -> None:
        self._running = False

    def _detect_entities(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO detection on frame."""
        if self._yolo_model is None:
            return []

        results = self._yolo_model(frame, conf=self.perception_config.yolo_conf_threshold)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2 / frame.shape[1]
                cy = (y1 + y2) / 2 / frame.shape[0]
                detections.append({
                    "class_id": int(box.cls[0]),
                    "x": cx,
                    "y": cy,
                    "w": (x2 - x1) / frame.shape[1],
                    "h": (y2 - y1) / frame.shape[0],
                    "conf": float(box.conf[0]),
                })
        return detections

    def _read_tower_hps(self, frame: np.ndarray) -> dict[str, float]:
        """Read tower HP values via OCR."""
        # Placeholder — needs PaddleOCR or custom HP reader
        return {
            "p0_king": 4008, "p0_left": 2534, "p0_right": 2534,
            "p1_king": 4008, "p1_left": 2534, "p1_right": 2534,
        }

    def _classify_hand(self, frame: np.ndarray) -> list[int]:
        """Classify cards in hand via ResNet."""
        # Placeholder — needs card classifier model
        return [0, 0, 0, 0]

    def _read_elixir(self, frame: np.ndarray) -> float:
        """Read elixir bar value."""
        # Placeholder — needs elixir reader
        return 5.0
