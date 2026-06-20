"""V2 training orchestrator — full pipeline with all improvements.

Usage::

    # Single GPU (testing)
    python scripts/train_v2.py --device cuda:0

    # Full 8x A100 distributed training
    python scripts/train_v2.py --distributed --n-gpus 8

    # With imitation warm-start
    python scripts/train_v2.py --warmstart-data data/battles.jsonl --distributed

Phases:
    1. (Optional) Imitation warm-start on battle outcome data
    2. Self-play with Gumbel MuZero search
    3. Training with KataGo improvements + auxiliary heads
    4. Periodic evaluation against fixed opponents
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import numpy as np
import torch

from crsim.cards import CardType
from crsim.game import CRGame
from mcts.gumbel_search import GumbelConfig, GumbelMuZeroSearch
from model.transformer_net import CRStarNet
from training.curriculum import CurriculumManager
from training.domain_randomization import DomainRandomizer
from training.league import AgentType, League, LeagueConfig
from training.replay_buffer import ReplayBuffer
from training.self_play_v2 import SelfPlayV2Config, SelfPlayWorkerV2
from training.trainer_v2 import TrainerV2

logger = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_vs_random(
    model: torch.nn.Module,
    device: torch.device,
    n_games: int = 50,
    decision_interval: int = 8,
) -> float:
    """Quick evaluation: play n games vs random opponent, return win rate.

    Uses the same decision interval as self-play (search every N ticks, WAIT in
    between) so evaluation is tractable instead of running a full search on every
    one of the up-to-7200 ticks per game.
    """
    from crsim.constants import WAIT_ACTION
    from crsim.game import Action
    from mcts.gumbel_search import _action_id_to_action

    searcher = GumbelMuZeroSearch(
        model=model,
        config=GumbelConfig(n_simulations=16, temperature=0.1),
        device=device,
    )

    wait = lambda p: Action(player=p, hand_slot=-1)  # noqa: E731
    interval = max(1, decision_interval)

    wins = 0
    for _ in range(n_games):
        deck_p0 = list(np.random.choice(list(CardType), 8, replace=False))
        deck_p1 = list(np.random.choice(list(CardType), 8, replace=False))
        game = CRGame(deck_p0=deck_p0, deck_p1=deck_p1)

        while not game.done:
            if game.tick_count % interval != 0:
                game.step([wait(0), wait(1)])
                continue

            # Player 0: model
            action_id, _ = searcher.select_action(game, 0, deterministic=True)
            p0_action = _action_id_to_action(action_id, 0)

            # Player 1: random
            mask = game.get_valid_actions_mask(1)
            valid = np.where(mask)[0]
            p1_action_id = int(np.random.choice(valid)) if len(valid) > 0 else WAIT_ACTION
            p1_action = _action_id_to_action(p1_action_id, 1)

            game.step([p0_action, p1_action])

        if game.get_reward(0) > 0:
            wins += 1

    return wins / max(n_games, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ClashRoyale-Zero V2 Training")
    parser.add_argument("--device", default="cuda:0", help="Primary device")
    parser.add_argument("--distributed", action="store_true", help="Use Ray distributed training")
    parser.add_argument("--n-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs/v2")

    # Gumbel search config
    parser.add_argument("--gumbel-sims", type=int, default=16)
    parser.add_argument("--playout-cap-randomization", action="store_true", default=True)

    # Self-play config
    parser.add_argument("--self-play-games-per-batch", type=int, default=32)
    parser.add_argument("--n-self-play-workers", type=int, default=4)

    # Imitation warm-start
    parser.add_argument(
        "--warmstart-data", type=str, default=None,
        help="Path to battle data for warm-start",
    )
    parser.add_argument("--warmstart-epochs", type=int, default=5)

    # Evaluation
    parser.add_argument(
        "--eval-interval", type=int, default=10_000,
        help="Steps between evaluations",
    )
    parser.add_argument("--eval-games", type=int, default=50)

    # Decision cadence / game truncation (search every N ticks, not every tick)
    parser.add_argument(
        "--decision-interval", type=int, default=8,
        help="Run MCTS search every N game ticks (real CR bots act ~5x/sec)",
    )
    parser.add_argument(
        "--max-game-ticks", type=int, default=None,
        help="Optional hard cap on ticks per self-play game (for fast runs)",
    )
    parser.add_argument(
        "--warmup-positions", type=int, default=5_000,
        help="Fill the replay buffer to this many positions before training",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # === Phase 0: Create model ===
    model = CRStarNet(
        spatial_blocks=10,
        spatial_filters=128,
        core_hidden=512,
        core_layers=2,
    )
    n_params = count_parameters(model)
    logger.info("CRStarNet created: %.1fM parameters", n_params / 1e6)

    # === Phase 1: Imitation warm-start (optional) ===
    if args.warmstart_data:
        logger.info("=== Phase 1: Imitation warm-start ===")
        from training.imitation import train_value_warmstart

        train_value_warmstart(
            battles_path=args.warmstart_data,
            output_path=f"{args.checkpoint_dir}/value_warmstart.pt",
            epochs=args.warmstart_epochs,
            device=str(device),
            model=model,  # warm-start the actual training model in place
        )
        logger.info("Value warm-start complete")

    # === Phase 2: Self-play training ===
    logger.info("=== Phase 2: Self-play training with Gumbel MuZero ===")

    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    gumbel_cfg = GumbelConfig(
        n_simulations=args.gumbel_sims,
        playout_cap_randomization=args.playout_cap_randomization,
    )
    sp_cfg = SelfPlayV2Config(
        n_games=args.self_play_games_per_batch,
        gumbel_config=gumbel_cfg,
        decision_interval_ticks=args.decision_interval,
        max_ticks=args.max_game_ticks,
    )

    # Initialize curriculum, domain randomization, league
    curriculum = CurriculumManager()
    randomizer = DomainRandomizer(
        strength=0.1, adr_enabled=True,
        adr_min_strength=0.02, adr_max_strength=0.25,
    )
    league = League(LeagueConfig(
        n_main_agents=1, n_league_exploiters=0, n_main_exploiters=0,
    ))
    # Register self as main agent
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    initial_ckpt = f"{args.checkpoint_dir}/initial.pt"
    torch.save(model.state_dict(), initial_ckpt)
    main_agent = league.add_agent(AgentType.MAIN, initial_ckpt)

    logger.info("Curriculum phase: %s", curriculum.phase.name)
    logger.info("Domain randomization strength: %.2f", randomizer.strength)

    trainer = TrainerV2(
        model=model,
        replay_buffer=replay_buffer,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        policy_target_pruning=True,
    )

    if args.distributed:
        _run_distributed(model, replay_buffer, trainer, sp_cfg, args, device)
    else:
        _run_single_gpu(
            model, replay_buffer, trainer, sp_cfg, args, device,
            curriculum=curriculum, randomizer=randomizer,
            league=league, main_agent=main_agent,
        )


def _run_single_gpu(
    model: CRStarNet,
    replay_buffer: ReplayBuffer,
    trainer: TrainerV2,
    sp_cfg: SelfPlayV2Config,
    args: argparse.Namespace,
    device: torch.device,
    curriculum: CurriculumManager | None = None,
    randomizer: DomainRandomizer | None = None,
    league: League | None = None,
    main_agent=None,
) -> None:
    """Single-GPU training: alternate self-play and training."""
    worker = SelfPlayWorkerV2(
        model=model,
        replay_buffer=replay_buffer,
        config=sp_cfg,
        device=device,
        worker_id=0,
        curriculum=curriculum,
        randomizer=randomizer,
    )

    logger.info("Starting single-GPU training loop")
    logger.info("Generating initial self-play data...")

    # Fill replay buffer with initial data
    warmup = getattr(args, "warmup_positions", 5_000)
    warmup_batch = max(1, min(16, args.self_play_games_per_batch))
    while len(replay_buffer) < warmup:
        worker.run_batch(n_games=warmup_batch)
        logger.info("Buffer size: %d / %d", len(replay_buffer), warmup)

    # Alternate: self-play → train → self-play → train
    steps_since_eval = 0
    total_games = 0
    while trainer.step_count < args.max_steps:
        # Self-play batch
        worker.run_batch(n_games=8)
        total_games += 8

        # Train for a while
        for _ in range(100):
            if len(replay_buffer) >= args.batch_size:
                trainer.train_step()

        # Sync model weights back to self-play worker
        worker.update_model(trainer.get_model_state())

        steps_since_eval += 100
        if steps_since_eval >= args.eval_interval:
            win_rate = evaluate_vs_random(
                model, device, n_games=args.eval_games,
                decision_interval=args.decision_interval,
            )
            logger.info(
                "Eval @ step %d: %.1f%% win rate vs random",
                trainer.step_count, win_rate * 100,
            )

            # Update curriculum based on performance
            if curriculum is not None:
                phase_changed = curriculum.update(win_rate)
                if phase_changed:
                    logger.info("Curriculum advanced to: %s", curriculum.phase.name)

            # Update ADR
            if randomizer is not None:
                randomizer.step_adr(win_rate)

            # League snapshot
            if league is not None and main_agent is not None:
                league.update_win_rate(main_agent, main_agent, win_rate > 0.5)
                if total_games % league.config.snapshot_every_n_games < 8:
                    ckpt = f"{args.checkpoint_dir}/league_gen{main_agent.generation}.pt"
                    torch.save(model.state_dict(), ckpt)
                    league.snapshot_agent(main_agent, ckpt)

            steps_since_eval = 0

    trainer.save_checkpoint(tag="final")
    final_wr = evaluate_vs_random(
        model, device, n_games=100, decision_interval=args.decision_interval,
    )
    logger.info("Training complete! Final win rate vs random: %.1f%%", final_wr * 100)


def _run_distributed(
    model: CRStarNet,
    replay_buffer: ReplayBuffer,
    trainer: TrainerV2,
    sp_cfg: SelfPlayV2Config,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Distributed training with Ray: N-2 GPUs self-play, 2 GPUs training."""
    try:
        import ray
    except ImportError:
        logger.error("Ray not installed. Install with: pip install 'ray[default]'")
        logger.info("Falling back to single-GPU mode")
        _run_single_gpu(model, replay_buffer, trainer, sp_cfg, args, device)
        return

    n_gpus = args.n_gpus
    n_train_gpus = min(2, n_gpus)
    n_play_gpus = n_gpus - n_train_gpus

    ray.init(num_gpus=n_gpus)
    logger.info("Ray initialized: %d GPUs (%d train, %d self-play)",
                n_gpus, n_train_gpus, n_play_gpus)

    @ray.remote(num_gpus=1)
    class SelfPlayActor:
        def __init__(self, model_state: dict, config: SelfPlayV2Config, worker_id: int) -> None:
            self.device = torch.device("cuda:0")
            self.model = CRStarNet()
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()

            self.buffer = ReplayBuffer(max_size=50_000)
            self.worker = SelfPlayWorkerV2(
                model=self.model,
                replay_buffer=self.buffer,
                config=config,
                device=self.device,
                worker_id=worker_id,
            )

        def generate_games(self, n_games: int = 16) -> list:
            self.worker.run_batch(n_games=n_games)
            # Collect all entries from local buffer
            entries = []
            while len(self.buffer) > 0:
                batch = self.buffer.sample(min(len(self.buffer), 1000))
                entries.append(batch)
            return entries

        def update_model(self, state_dict: dict) -> None:
            self.model.load_state_dict(state_dict)
            self.model.eval()

    # Launch self-play actors
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    actors = [
        SelfPlayActor.remote(model_state, sp_cfg, worker_id=i)
        for i in range(n_play_gpus)
    ]

    logger.info("Launched %d self-play actors", len(actors))

    # Main training loop
    def collect_data() -> None:
        """Background thread: collect data from self-play actors."""
        while trainer.step_count < args.max_steps:
            futures = [actor.generate_games.remote(16) for actor in actors]
            results = ray.get(futures)
            for batches in results:
                for batch in batches:
                    spatial, scalar, policy, value = batch
                    for i in range(len(spatial)):
                        from training.replay_buffer import ReplayEntry
                        replay_buffer.push(ReplayEntry(
                            spatial=spatial[i],
                            scalar=scalar[i],
                            policy=policy[i],
                            value=float(value[i]),
                        ))

            # Sync model weights to actors
            state = trainer.get_model_state()
            for actor in actors:
                actor.update_model.remote(state)

    # Start data collection in background
    collector = threading.Thread(target=collect_data, daemon=True)
    collector.start()

    # Wait for initial data
    logger.info("Waiting for initial self-play data...")
    while len(replay_buffer) < 5_000:
        time.sleep(1.0)

    # Train
    trainer.train_loop(
        min_buffer_size=5_000,
        steps_between_checkpoints=5_000,
    )

    ray.shutdown()
    logger.info("Distributed training complete!")


if __name__ == "__main__":
    main()
