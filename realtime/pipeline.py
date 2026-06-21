"""End-to-end real-time inference pipeline.

Orchestrates: screen capture → perception → neural net → action → ADB touch.
Target latency: <100ms per decision cycle.

Usage:
    pipeline = RealtimePipeline(model, device="cuda:0")
    pipeline.run()  # starts the game-playing loop
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from crsim.cards import CARD_DEFS
from crsim.constants import ARENA_H, ARENA_W, NUM_HAND_SLOTS
from model.opponent_model import OpponentBeliefState
from realtime.controller import ADBConfig, ADBController, ReplayActionLogger
from realtime.perception import GamePerception, PerceivedGameState

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    decision_interval: float = 0.5  # seconds between decisions
    min_elixir_to_play: float = 2.0  # don't play cards below this
    confidence_threshold: float = 0.3  # min confidence for detections
    temperature: float = 0.5  # action sampling temperature
    use_gumbel_search: bool = False  # if True, use MCTS; if False, raw NN
    n_search_sims: int = 16  # Gumbel MuZero simulations
    log_actions: bool = True
    yolo_weights: str = "models/yolov8_cr.pt"
    card_templates_dir: str = "models/card_templates/"


class RealtimePipeline:
    """Complete real-time play pipeline.

    Lifecycle:
        1. Initialize perception + controller + model
        2. Enter game loop
        3. Each tick: capture → perceive → decide → act
        4. Log actions for replay analysis
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: PipelineConfig | None = None,
        adb_config: ADBConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or PipelineConfig()
        self.device = torch.device(device)

        # Model
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        # Components
        self.perception = GamePerception()
        self.controller = ADBController(adb_config)
        self.belief = OpponentBeliefState()
        self.action_logger = ReplayActionLogger() if self.config.log_actions else None

        # State
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._running = False
        self._frame_count = 0
        self._decision_count = 0
        self._total_inference_time = 0.0

    def setup(self) -> bool:
        """Initialize all components. Returns True if ready."""
        # Load perception models
        self.perception.load_models(
            self.config.yolo_weights,
            self.config.card_templates_dir,
        )

        # Check ADB connection
        if not self.controller.is_connected():
            logger.error("No ADB device connected")
            return False

        logger.info("Pipeline ready — perception + ADB connected")
        return True

    def run(self, max_duration: float = 300.0) -> None:
        """Main game loop.

        Parameters
        ----------
        max_duration : float
            Maximum play time in seconds (default: 5 min match).
        """
        self._running = True
        start = time.time()

        logger.info("Starting real-time play loop (max %.0fs)", max_duration)

        while self._running and (time.time() - start) < max_duration:
            loop_start = time.time()

            # 1. Capture screen
            frame = self.controller.capture_screen()
            if frame is None:
                logger.warning("Screen capture failed, retrying...")
                time.sleep(0.1)
                continue

            self._frame_count += 1

            # 2. Perceive game state
            state = self.perception.process_frame(frame)

            # 3. Decide action
            action = self._decide(state)

            # 4. Execute action
            if action is not None:
                slot, ax, ay = action
                self.controller.play_card(slot, ax, ay)
                self._decision_count += 1

                if self.action_logger:
                    ct = state.hand_cards[slot] if slot < len(state.hand_cards) else None
                    self.action_logger.log_action(
                        slot=slot,
                        arena_x=ax,
                        arena_y=ay,
                        card_type=int(ct) if ct is not None else None,
                        elixir=state.elixir,
                    )

            # 5. Wait for next decision interval
            elapsed = time.time() - loop_start
            wait = max(0, self.config.decision_interval - elapsed)
            if wait > 0:
                time.sleep(wait)

        self._running = False
        self._report_stats()

    def stop(self) -> None:
        """Stop the play loop."""
        self._running = False

    def _decide(
        self, state: PerceivedGameState,
    ) -> tuple[int, float, float] | None:
        """Decide what action to take given perceived state.

        Returns (card_slot, arena_x, arena_y) or None for wait.
        """
        # Convert perception to tensors
        spatial, scalar, entity_feats, entity_mask = (
            self.perception.perceived_to_features(state)
        )

        sp_t = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
        sc_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)
        ef_t = torch.from_numpy(entity_feats).unsqueeze(0).to(self.device)
        em_t = torch.from_numpy(entity_mask).unsqueeze(0).to(self.device)

        # Encode belief state
        belief_vec = self.belief.encode(int(time.time() * 2))
        bf_t = torch.from_numpy(belief_vec).unsqueeze(0).to(self.device)

        # Neural network inference
        t0 = time.time()
        with torch.no_grad():
            if hasattr(self.model, "predict_autoregressive"):
                card_idx, x_pos, y_pos, _log_prob, extras = (
                    self.model.predict_autoregressive(
                        sp_t, sc_t,
                        entity_features=ef_t,
                        entity_mask=em_t,
                        hidden=self._hidden,
                        temperature=self.config.temperature,
                        belief_features=bf_t,
                    )
                )
                self._hidden = extras.get("hidden")
                value = float(extras.get("value", torch.tensor(0.0)).item())
            else:
                policy, value = self.model.predict(sp_t, sc_t)
                action_id = int(torch.multinomial(policy, 1).item())
                card_idx = action_id // (ARENA_W * ARENA_H)
                rem = action_id % (ARENA_W * ARENA_H)
                x_pos = rem // ARENA_H
                y_pos = rem % ARENA_H
                value = float(value.item())

        inference_ms = (time.time() - t0) * 1000
        self._total_inference_time += inference_ms

        logger.debug(
            "Inference: card=%d x=%d y=%d value=%.3f (%.1fms)",
            card_idx, x_pos, y_pos, value, inference_ms,
        )

        # Wait action
        if card_idx >= NUM_HAND_SLOTS:
            return None

        # Check if we have enough elixir
        card_type = state.hand_cards[card_idx] if card_idx < len(state.hand_cards) else None
        if card_type is not None:
            card_def = CARD_DEFS.get(card_type)
            if card_def and card_def.cost > state.elixir:
                logger.debug(
                    "Not enough elixir (%.1f) for card slot %d (cost %d)",
                    state.elixir, card_idx, card_def.cost,
                )
                return None

        # Don't play if elixir is too low overall
        if state.elixir < self.config.min_elixir_to_play:
            return None

        return card_idx, float(x_pos), float(y_pos)

    def _report_stats(self) -> None:
        avg_ms = (
            self._total_inference_time / max(self._decision_count, 1)
        )
        logger.info(
            "Pipeline stats: %d frames, %d decisions, avg inference %.1fms",
            self._frame_count,
            self._decision_count,
            avg_ms,
        )

    def save_replay(self, path: str) -> None:
        """Save action log for replay analysis."""
        if self.action_logger:
            self.action_logger.save(path)
            logger.info("Saved replay to %s", path)
