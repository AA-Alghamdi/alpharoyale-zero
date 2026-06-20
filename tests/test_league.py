"""Tests for the AlphaStar-style training league.

Cover the pieces the trainer relies on: snapshotting (generation bookkeeping),
PFSP opponent prioritization (harder opponents sampled more), bounded EMA win
rates, exploiter reset (save exploit + restore initial weights), and the
self-play worker opponent-weight round-trip.
"""

from __future__ import annotations

import numpy as np
import torch

from model.network import CRZeroNet
from training.league import AgentType, League, LeagueConfig


def _net() -> CRZeroNet:
    return CRZeroNet(n_res_blocks=1, n_filters=8)


def test_add_and_snapshot(tmp_path):
    lg = League(LeagueConfig())
    a = lg.add_agent(AgentType.MAIN, str(tmp_path / "m.pt"))
    assert a in lg.agents
    assert a.generation == 0

    frozen = lg.snapshot_agent(a, str(tmp_path / "m_gen0.pt"))
    assert frozen in lg.frozen_agents
    assert frozen.is_active is False
    assert frozen.agent_id.endswith("_gen0")
    assert a.generation == 1  # snapshotting advances the live agent's generation


def test_pfsp_prioritizes_hard_opponents(tmp_path):
    lg = League(LeagueConfig(pfsp_p=2.0))
    main = lg.add_agent(AgentType.MAIN, str(tmp_path / "m.pt"))
    hard = lg.add_agent(AgentType.MAIN, str(tmp_path / "h.pt"))
    easy = lg.add_agent(AgentType.MAIN, str(tmp_path / "e.pt"))
    main.win_rates = {hard.agent_id: 0.1, easy.agent_id: 0.9}

    np.random.seed(0)
    counts = {hard.agent_id: 0, easy.agent_id: 0}
    for _ in range(3000):
        opp = lg.select_opponent_pfsp(main)
        counts[opp.agent_id] += 1
    # priority (1-wr)^2: hard=0.81, easy=0.01 -> hard sampled vastly more often
    assert counts[hard.agent_id] > 10 * counts[easy.agent_id]


def test_update_win_rate_stays_bounded(tmp_path):
    lg = League(LeagueConfig())
    a = lg.add_agent(AgentType.MAIN, str(tmp_path / "m.pt"))
    opp = lg.add_agent(AgentType.MAIN, str(tmp_path / "o.pt"))

    for _ in range(60):
        lg.update_win_rate(a, opp, True)
        assert 0.0 <= a.win_rates[opp.agent_id] <= 1.0
    assert a.win_rates[opp.agent_id] > 0.8

    peak = a.win_rates[opp.agent_id]
    for _ in range(60):
        lg.update_win_rate(a, opp, False)
        assert 0.0 <= a.win_rates[opp.agent_id] <= 1.0
    # EMA span caps at 100 games (alpha -> ~0.02), so decay is slow but clearly
    # downward from the post-win peak.
    assert a.win_rates[opp.agent_id] < peak - 0.3


def test_exploiter_reset_saves_and_restores(tmp_path):
    lg = League(LeagueConfig(exploiter_win_rate_threshold=0.7))
    initial = tmp_path / "init.pt"
    model = _net()
    torch.save(model.state_dict(), initial)

    ex = lg.add_agent(AgentType.LEAGUE_EXPLOITER, str(tmp_path / "ex.pt"))
    ex.win_rates = {"x": 0.9, "y": 0.85}
    ex.games_played = 500

    # diverge the model from the initial checkpoint
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    assert lg.check_exploiter_reset(ex, model, str(initial)) is True
    assert ex.games_played == 0
    assert ex.win_rates == {}
    assert len(lg.frozen_agents) == 1  # the exploit was snapshotted

    # model weights restored to the initial checkpoint
    init_sd = torch.load(initial, weights_only=True)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, init_sd[k])


def test_exploiter_no_reset_below_threshold(tmp_path):
    lg = League(LeagueConfig(exploiter_win_rate_threshold=0.7))
    ex = lg.add_agent(AgentType.LEAGUE_EXPLOITER, str(tmp_path / "ex.pt"))
    ex.win_rates = {"x": 0.5}
    assert lg.check_exploiter_reset(ex, _net(), "nonexistent.pt") is False


def test_main_agent_never_resets(tmp_path):
    lg = League(LeagueConfig())
    a = lg.add_agent(AgentType.MAIN, str(tmp_path / "m.pt"))
    a.win_rates = {"x": 0.99}
    assert lg.check_exploiter_reset(a, _net(), "x.pt") is False


def test_get_opponent_weights_roundtrip(tmp_path):
    lg = League(LeagueConfig(main_self_play_fraction=0.0))
    lg.add_agent(AgentType.MAIN, str(tmp_path / "m.pt"))
    opp_ckpt = tmp_path / "opp.pt"
    net = _net()
    torch.save(net.state_dict(), opp_ckpt)
    lg.add_agent(AgentType.MAIN, str(opp_ckpt))

    np.random.seed(0)
    weights = lg.get_opponent_weights(0)  # main has exactly one opponent
    assert weights is not None
    assert set(weights.keys()) == set(net.state_dict().keys())


def test_get_opponent_weights_single_agent_is_selfplay(tmp_path):
    lg = League(LeagueConfig(main_self_play_fraction=0.0))
    lg.add_agent(AgentType.MAIN, str(tmp_path / "m.pt"))
    assert lg.get_opponent_weights(0) is None  # no opponents -> self-play


def test_league_stats(tmp_path):
    lg = League(LeagueConfig())
    a = lg.add_agent(AgentType.MAIN, str(tmp_path / "m.pt"))
    lg.snapshot_agent(a, str(tmp_path / "g0.pt"))
    stats = lg.get_league_stats()
    assert stats["active_agents"] == 1
    assert stats["frozen_snapshots"] == 1
    assert stats["total_league_size"] == 2
