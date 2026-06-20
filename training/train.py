"""Main training orchestrator.

Ties together all training components:
  - AlphaStar League training
  - Curriculum learning
  - Domain randomization
  - Imitation warm-start
  - Parallel simulation
  - Opponent modeling
  - KataGo optimizations

Usage:
    python -m training.train --config config/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from training.curriculum import CurriculumManager
from training.domain_randomization import DomainRandomizer
from training.league import League, LeagueConfig

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CR AlphaZero agent")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--imitation-only", action="store_true", help="Only run imitation warm-start")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("=" * 60)
    logger.info("Clash Royale AlphaZero Training")
    logger.info("=" * 60)

    # Load config
    # TODO: load from YAML when config is finalized
    logger.info(f"Device: {args.device}")

    # Phase 0: Imitation warm-start (if data available)
    if args.imitation_only or (not args.resume and os.path.exists("data/imitation")):
        logger.info("Phase 0: Imitation learning warm-start")
        # Model would be instantiated here from config
        # train_imitation(model, ImitationConfig())

    # Initialize league
    league_config = LeagueConfig(
        n_main_agents=3,
        n_league_exploiters=2,
        n_main_exploiters=2,
        pfsp_p=1.0,
        exploiter_win_rate_threshold=0.7,
    )
    league = League(league_config)
    logger.info(f"League: {len(league.agents)} agents, {league.config.n_main_agents} main")

    # Initialize curriculum
    curriculum = CurriculumManager()
    logger.info(f"Starting curriculum phase: {curriculum.phase.name}")

    # Initialize domain randomization
    randomizer = DomainRandomizer(
        strength=0.1,
        adr_enabled=True,
        adr_min_strength=0.02,
        adr_max_strength=0.25,
    )
    logger.info(f"Domain randomization strength: {randomizer.strength}")

    # Delegate to the real training script
    logger.info("Delegating to scripts/train_v2.py training loop")
    from scripts.train_v2 import main as train_main
    train_main()


if __name__ == "__main__":
    main()
