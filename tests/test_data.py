"""Tests for data collection and dataset modules."""

from __future__ import annotations

import json

import numpy as np

from data.dataset import BattleOutcomeDataset, TrajectoryDataset, save_trajectories
from data.scraper import _extract_battle_record


class TestBattleRecordExtraction:
    def test_valid_record(self):
        battle = {
            "battleTime": "20230101T120000.000Z",
            "type": "PvP",
            "team": [{
                "tag": "#ABC",
                "crownsEarned": 3,
                "startingTrophies": 6000,
                "cards": [{"id": i} for i in range(8)],
            }],
            "opponent": [{
                "tag": "#DEF",
                "crownsEarned": 1,
                "startingTrophies": 5500,
                "cards": [{"id": i + 100} for i in range(8)],
            }],
        }

        record = _extract_battle_record(battle)
        assert record is not None
        assert record["winner"] == 0
        assert record["crowns_p0"] == 3
        assert record["crowns_p1"] == 1
        assert len(record["deck_p0"]) == 8
        assert len(record["deck_p1"]) == 8

    def test_draw(self):
        battle = {
            "team": [{
                "tag": "#A", "crownsEarned": 0, "startingTrophies": 5000,
                "cards": [{"id": i} for i in range(8)],
            }],
            "opponent": [{
                "tag": "#B", "crownsEarned": 0, "startingTrophies": 5000,
                "cards": [{"id": i} for i in range(8)],
            }],
        }
        record = _extract_battle_record(battle)
        assert record is not None
        assert record["winner"] == -1

    def test_incomplete_deck(self):
        battle = {
            "team": [{"tag": "#A", "crownsEarned": 1, "cards": [{"id": 1}]}],
            "opponent": [{
                "tag": "#B", "crownsEarned": 0,
                "cards": [{"id": i} for i in range(8)],
            }],
        }
        record = _extract_battle_record(battle)
        assert record is None  # deck too short

    def test_empty_battle(self):
        record = _extract_battle_record({})
        assert record is None


class TestBattleOutcomeDataset:
    def test_load_and_getitem(self, tmp_path):
        battles_file = tmp_path / "battles.jsonl"
        records = []
        for i in range(10):
            records.append(json.dumps({
                "deck_p0": list(range(8)),
                "deck_p1": list(range(8, 16)),
                "trophies_p0": 5000 + i * 100,
                "trophies_p1": 5500,
                "winner": i % 2,
                "crowns_p0": 2 if i % 2 == 0 else 1,
                "crowns_p1": 1 if i % 2 == 0 else 2,
            }))
        battles_file.write_text("\n".join(records))

        dataset = BattleOutcomeDataset(str(battles_file), card_vocab_size=50)
        assert len(dataset) == 10

        sample = dataset[0]
        assert "features" in sample
        assert "winner" in sample
        assert sample["features"].shape == (102,)  # 50*2 + 2

    def test_max_samples(self, tmp_path):
        battles_file = tmp_path / "battles.jsonl"
        records = [json.dumps({
            "deck_p0": list(range(8)),
            "deck_p1": list(range(8, 16)),
            "winner": 0,
        }) for _ in range(100)]
        battles_file.write_text("\n".join(records))

        dataset = BattleOutcomeDataset(str(battles_file), max_samples=10)
        assert len(dataset) == 10


class TestTrajectoryDataset:
    def test_load_npz(self, tmp_path):
        n = 50
        spatial = np.random.randn(n, 44, 32, 18).astype(np.float32)
        scalar = np.random.randn(n, 116).astype(np.float32)
        policy = np.random.dirichlet([1.0] * 10, size=n).astype(np.float32)
        # Pad policy to match action space
        policy = np.pad(policy, ((0, 0), (0, 2305 - 10)), constant_values=0.0)
        value = np.random.uniform(-1, 1, size=n).astype(np.float32)

        save_trajectories(spatial, scalar, policy, value, str(tmp_path), "trajectory_000")

        dataset = TrajectoryDataset(str(tmp_path))
        assert len(dataset) == n

        sample = dataset[0]
        assert sample["spatial"].shape == (44, 32, 18)
        assert sample["scalar"].shape == (116,)
        assert sample["value"].shape == ()


class TestSaveTrajectories:
    def test_save_and_load(self, tmp_path):
        n = 20
        spatial = np.random.randn(n, 44, 32, 18).astype(np.float32)
        scalar = np.random.randn(n, 116).astype(np.float32)
        policy = np.random.randn(n, 2305).astype(np.float32)
        value = np.random.randn(n).astype(np.float32)

        save_trajectories(spatial, scalar, policy, value, str(tmp_path), "test_chunk")

        loaded = np.load(tmp_path / "test_chunk.npz")
        assert loaded["spatial"].shape == (n, 44, 32, 18)
        assert loaded["scalar"].shape == (n, 116)
