"""AlphaStar-style League Training.

Maintains a diverse league of agents to prevent strategy collapse
and ensure robust play against all opponent types.

Three agent types (from AlphaStar, Nature 2019):
  1. Main Agents: Train via PFSP against all past agents. These become
     the final deployed agents.
  2. League Exploiters: Train to beat ALL agents in the league. When
     they find an exploit, they get added as a "hard opponent."
  3. Main Exploiters: Train specifically to beat main agents. Forces
     main agents to patch weaknesses.

Reference: Vinyals et al. "Grandmaster level in StarCraft II using
multi-agent reinforcement learning" Nature 2019.
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AgentType(Enum):
    MAIN = auto()
    LEAGUE_EXPLOITER = auto()
    MAIN_EXPLOITER = auto()


@dataclass
class AgentRecord:
    """A snapshot of an agent's weights + metadata in the league."""

    agent_id: str
    agent_type: AgentType
    weights_path: str  # path to saved model checkpoint
    win_rates: dict[str, float] = field(default_factory=dict)  # vs other agents
    games_played: int = 0
    generation: int = 0
    is_active: bool = True  # still training?

    @property
    def mean_win_rate(self) -> float:
        if not self.win_rates:
            return 0.5
        return sum(self.win_rates.values()) / len(self.win_rates)


@dataclass
class LeagueConfig:
    """Configuration for league training."""

    n_main_agents: int = 3
    n_league_exploiters: int = 2
    n_main_exploiters: int = 2

    # PFSP (Prioritized Fictitious Self-Play) parameters
    pfsp_p: float = 1.0  # prioritization exponent (higher = more focus on hard opponents)

    # Exploiter reset threshold
    exploiter_win_rate_threshold: float = 0.7  # add to league and reset when win rate > 70%

    # Snapshot frequency
    snapshot_every_n_games: int = 1000

    # Self-play fraction for main agents
    main_self_play_fraction: float = 0.2  # 20% games vs current self, 80% vs league


class League:
    """AlphaStar-style training league.

    Manages a population of agents with different training objectives
    to ensure strategic diversity and robustness.
    """

    def __init__(self, config: LeagueConfig | None = None) -> None:
        self.config = config or LeagueConfig()
        self.agents: list[AgentRecord] = []
        self.frozen_agents: list[AgentRecord] = []  # historical snapshots

    def add_agent(self, agent_type: AgentType, weights_path: str) -> AgentRecord:
        record = AgentRecord(
            agent_id=str(uuid.uuid4())[:8],
            agent_type=agent_type,
            weights_path=weights_path,
        )
        self.agents.append(record)
        logger.info(f"Added {agent_type.name} agent {record.agent_id}")
        return record

    def snapshot_agent(self, agent: AgentRecord, weights_path: str) -> AgentRecord:
        """Save a frozen copy of an agent's current state to the league."""
        frozen = AgentRecord(
            agent_id=f"{agent.agent_id}_gen{agent.generation}",
            agent_type=agent.agent_type,
            weights_path=weights_path,
            win_rates=dict(agent.win_rates),
            games_played=agent.games_played,
            generation=agent.generation,
            is_active=False,
        )
        self.frozen_agents.append(frozen)
        agent.generation += 1
        return frozen

    def select_opponent_pfsp(self, agent: AgentRecord) -> AgentRecord:
        """Prioritized Fictitious Self-Play: sample opponent weighted by difficulty.

        Opponents that beat us more often are sampled more frequently.
        This is how AlphaStar's main agents train.
        """
        all_opponents = [a for a in self.agents + self.frozen_agents
                         if a.agent_id != agent.agent_id]
        if not all_opponents:
            return agent  # self-play fallback

        # Compute priority weights: harder opponents (lower win rate against) get more weight
        weights = []
        for opp in all_opponents:
            wr = agent.win_rates.get(opp.agent_id, 0.5)
            # Priority: (1 - win_rate)^p — we play more against opponents we lose to
            priority = (1.0 - wr + 1e-6) ** self.config.pfsp_p
            weights.append(priority)

        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        idx = np.random.choice(len(all_opponents), p=weights)
        return all_opponents[idx]

    def select_opponent_uniform(self, agent: AgentRecord) -> AgentRecord:
        """Uniform sampling over all agents (for league exploiters)."""
        all_opponents = [a for a in self.agents + self.frozen_agents
                         if a.agent_id != agent.agent_id]
        if not all_opponents:
            return agent
        return all_opponents[np.random.randint(len(all_opponents))]

    def select_opponent_main_only(self, agent: AgentRecord) -> AgentRecord:
        """Sample only from main agents (for main exploiters)."""
        main_agents = [a for a in self.agents
                       if a.agent_type == AgentType.MAIN and a.agent_id != agent.agent_id]
        if not main_agents:
            return self.select_opponent_uniform(agent)
        return main_agents[np.random.randint(len(main_agents))]

    def select_opponent(self, agent: AgentRecord) -> AgentRecord:
        """Select opponent based on agent type."""
        if agent.agent_type == AgentType.MAIN:
            # 20% self-play, 80% PFSP
            if np.random.random() < self.config.main_self_play_fraction:
                return agent
            return self.select_opponent_pfsp(agent)
        elif agent.agent_type == AgentType.LEAGUE_EXPLOITER:
            return self.select_opponent_uniform(agent)
        elif agent.agent_type == AgentType.MAIN_EXPLOITER:
            return self.select_opponent_main_only(agent)
        return self.select_opponent_pfsp(agent)

    def update_win_rate(self, agent: AgentRecord, opponent: AgentRecord, won: bool) -> None:
        """Update win rate tracking after a game."""
        opp_id = opponent.agent_id
        old_wr = agent.win_rates.get(opp_id, 0.5)
        # Exponential moving average
        alpha = 2.0 / (min(agent.games_played, 100) + 1)
        agent.win_rates[opp_id] = old_wr * (1 - alpha) + (1.0 if won else 0.0) * alpha
        agent.games_played += 1

    def check_exploiter_reset(self, agent: AgentRecord, model, initial_weights_path: str) -> bool:
        """Check if an exploiter should be reset (added exploit to league).

        Returns True if the agent was reset.
        """
        if agent.agent_type not in (AgentType.LEAGUE_EXPLOITER, AgentType.MAIN_EXPLOITER):
            return False

        if agent.mean_win_rate > self.config.exploiter_win_rate_threshold:
            # This exploiter found a strong strategy — save it and reset
            save_path = f"{agent.weights_path}_exploit_gen{agent.generation}"
            torch.save(model.state_dict(), save_path)
            self.snapshot_agent(agent, save_path)

            # Reset to initial supervised weights
            model.load_state_dict(torch.load(initial_weights_path, weights_only=True))
            agent.win_rates.clear()
            agent.games_played = 0
            logger.info(
                f"Exploiter {agent.agent_id} reset after {agent.mean_win_rate:.1%} win rate"
            )
            return True
        return False

    def get_league_stats(self) -> dict:
        """Summary statistics for logging."""
        stats = {
            "active_agents": len(self.agents),
            "frozen_snapshots": len(self.frozen_agents),
            "total_league_size": len(self.agents) + len(self.frozen_agents),
        }
        for agent in self.agents:
            key = f"{agent.agent_type.name}_{agent.agent_id}"
            stats[f"{key}_win_rate"] = agent.mean_win_rate
            stats[f"{key}_games"] = agent.games_played
        return stats
