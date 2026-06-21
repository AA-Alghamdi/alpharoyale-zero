"""Fast bounded BC warm-start on the ProAgent teacher data, then evaluate the
resulting policy (policy-only, no MCTS, so it's quick on CPU) vs Random and
ProAgent. Full BC is a GPU job; this proves the warm-start + eval path works.
"""
import os

import numpy as np
import torch

from eval.baseline_agents import ProAgent, RandomAgent, play_match
from model.features import encode_state, extract_entity_features
from model.transformer_net import CRStarNet
from training.imitation import (
    ExpertDataset,
    ImitationConfig,
    _forward_policy_value,
    train_behavioral_cloning,
)

N_SAMPLES = 12000
ds = ExpertDataset("data/expert_pro.npz", max_samples=N_SAMPLES)
print(f"BC dataset: {len(ds)} samples", flush=True)

model = CRStarNet()
cfg = ImitationConfig(batch_size=512, num_epochs=1, learning_rate=1e-3)
train_behavioral_cloning(model, ds, cfg, "cpu")
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/bc_pro.pt")
print("saved checkpoints/bc_pro.pt", flush=True)


class ModelPolicyAgent:
    """Greedy policy from the network (masked argmax, no search)."""

    def __init__(self, m):
        self.m = m.eval()

    @torch.no_grad()
    def select_action(self, game, player):
        sp, sc = encode_state(game, player)
        ef = extract_entity_features(game, player)
        if isinstance(ef, tuple):
            ef, em = ef
        else:
            em = np.ones((np.asarray(ef).shape[0],), dtype=bool)
        mask = np.asarray(game.get_valid_actions_mask(player))
        sp = torch.from_numpy(np.asarray(sp)).unsqueeze(0).float()
        sc = torch.from_numpy(np.asarray(sc)).unsqueeze(0).float()
        mk = torch.from_numpy(mask).unsqueeze(0).bool()
        eft = torch.from_numpy(np.asarray(ef)).unsqueeze(0).float()
        emt = torch.from_numpy(np.asarray(em)).unsqueeze(0)
        logits, _ = _forward_policy_value(self.m, sp, sc, mk, eft, emt)
        lg = logits.squeeze(0).clone()
        lg[~mk.squeeze(0)] = -1e9
        return int(torch.argmax(lg))


agent = ModelPolicyAgent(model)
for name, opp in [("Random", RandomAgent()), ("ProAgent", ProAgent())]:
    a, b, d = play_match(agent, opp, n_games=40, seed=11)
    wr = (a + 0.5 * d) / max(a + b + d, 1)
    print(f"BC_PRO (policy-only) vs {name}: {a}-{b}-{d}  winrate {wr:.1%}", flush=True)
print("EVAL-DONE", flush=True)
