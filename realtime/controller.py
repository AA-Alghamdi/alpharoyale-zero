"""ADB-based touch controller for playing Clash Royale on a real device.

Handles:
  - Card selection and placement via drag gestures
  - Timing of actions to respect animation/cooldowns
  - Screen capture via ADB screencap
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass

import numpy as np

from crsim.constants import NUM_HAND_SLOTS
from realtime.perception import (
    CARD_HAND_Y,
    CARD_SLOT_H,
    CARD_SLOT_W,
    CARD_SLOTS_X,
    arena_to_screen,
)

logger = logging.getLogger(__name__)


@dataclass
class ADBConfig:
    device_serial: str = ""  # empty = first connected device
    screencap_method: str = "adb"  # "adb" or "scrcpy"
    touch_delay_ms: int = 50
    drag_duration_ms: int = 200


class ADBController:
    """Control Clash Royale via ADB commands."""

    def __init__(self, config: ADBConfig | None = None) -> None:
        self.config = config or ADBConfig()
        self._adb_prefix = self._build_adb_prefix()
        self._last_action_time = 0.0

    def _build_adb_prefix(self) -> list[str]:
        prefix = ["adb"]
        if self.config.device_serial:
            prefix.extend(["-s", self.config.device_serial])
        return prefix

    def _adb(self, *args: str) -> subprocess.CompletedProcess:
        """Run an ADB command."""
        cmd = self._adb_prefix + ["shell"] + list(args)
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )

    def is_connected(self) -> bool:
        """Check if ADB device is connected."""
        try:
            result = subprocess.run(
                self._adb_prefix + ["devices"],
                capture_output=True, text=True, timeout=5,
            )
            lines = result.stdout.strip().split("\n")
            return len(lines) > 1 and "device" in lines[1]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def capture_screen(self) -> np.ndarray | None:
        """Capture the current screen as a numpy array (BGR).

        Returns None if capture fails.
        """
        try:
            result = subprocess.run(
                self._adb_prefix + ["exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None

            import cv2
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        except (subprocess.TimeoutExpired, ImportError, Exception) as e:
            logger.warning("Screen capture failed: %s", e)
            return None

    def tap(self, x: int, y: int) -> None:
        """Tap at screen coordinates."""
        self._adb("input", "tap", str(x), str(y))
        time.sleep(self.config.touch_delay_ms / 1000)

    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int | None = None,
    ) -> None:
        """Swipe from (x1,y1) to (x2,y2)."""
        d = duration_ms or self.config.drag_duration_ms
        self._adb("input", "swipe", str(x1), str(y1), str(x2), str(y2), str(d))
        time.sleep(d / 1000 + self.config.touch_delay_ms / 1000)

    def play_card(self, slot: int, arena_x: float, arena_y: float) -> None:
        """Play a card from hand slot to arena position.

        Parameters
        ----------
        slot : int (0-3)
            Hand slot index
        arena_x, arena_y : float
            Target arena tile coordinates
        """
        if slot < 0 or slot >= NUM_HAND_SLOTS:
            logger.warning("Invalid card slot: %d", slot)
            return

        # Card center in screen coords
        card_cx = CARD_SLOTS_X[slot] + CARD_SLOT_W // 2
        card_cy = CARD_HAND_Y + CARD_SLOT_H // 2

        # Target in screen coords
        target_sx, target_sy = arena_to_screen(arena_x, arena_y)

        # Drag card to target
        self.swipe(card_cx, card_cy, target_sx, target_sy)
        self._last_action_time = time.time()

        logger.debug(
            "Played card slot %d at arena (%.1f, %.1f) → screen (%d, %d)",
            slot, arena_x, arena_y, target_sx, target_sy,
        )

    def time_since_last_action(self) -> float:
        """Seconds since last action was performed."""
        if self._last_action_time == 0:
            return float("inf")
        return time.time() - self._last_action_time


class ReplayActionLogger:
    """Log actions taken during real-time play for replay analysis."""

    def __init__(self) -> None:
        self.actions: list[dict] = []
        self._start_time = time.time()

    def log_action(
        self,
        slot: int,
        arena_x: float,
        arena_y: float,
        card_type: int | None = None,
        elixir: float = 0.0,
        value_est: float = 0.0,
    ) -> None:
        self.actions.append({
            "time": time.time() - self._start_time,
            "slot": slot,
            "arena_x": arena_x,
            "arena_y": arena_y,
            "card_type": card_type,
            "elixir": elixir,
            "value_est": value_est,
        })

    def save(self, path: str) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.actions, indent=2))
