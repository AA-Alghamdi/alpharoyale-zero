"""Eval a CRStarNet checkpoint (policy-only, fast) vs Random + ProAgent."""
import sys

import numpy as np
import torch

from eval.baseline_agents import ProAgent, RandomAgent, play_match
from model.features import encode_state, extract_entity_features
from model.transformer_net import CRStarNet
from training.imitation import _forward_policy_value

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/bc_pro.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = CRStarNet(spatial_blocks=10, spatial_filters=128, core_hidden=512, core_layers=2).to(dev)
sd = torch.load(ckpt, map_location=dev)
m.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd)
m.eval()


class PolicyAgent:
    def __init__(self, model):
        self.m = model

    @torch.no_grad()
    def select_action(self, game, player):
        sp, sc = encode_state(game, player)
        ef = extract_entity_features(game, player)
        if isinstance(ef, tuple):
            ef, em = ef
        else:
            em = np.ones((np.asarray(ef).shape[0],), dtype=bool)
        mask = np.asarray(game.get_valid_actions_mask(player))
        sp = torch.from_numpy(np.asarray(sp)).unsqueeze(0).float().to(dev)
        sc = torch.from_numpy(np.asarray(sc)).unsqueeze(0).float().to(dev)
        mk = torch.from_numpy(mask).unsqueeze(0).bool().to(dev)
        eft = torch.from_numpy(np.asarray(ef)).unsqueeze(0).float().to(dev)
        emt = torch.from_numpy(np.asarray(em)).unsqueeze(0).to(dev)
        lg, _ = _forward_policy_value(self.m, sp, sc, mk, eft, emt)
        lg = lg.squeeze(0).clone()
        lg[~mk.squeeze(0)] = -1e9
        return int(lg.argmax())


ag = PolicyAgent(m)
for nm, opp in [("Random", RandomAgent()), ("ProAgent", ProAgent())]:
    a, b, d = play_match(ag, opp, n_games=40, seed=11)
    print(f"CKPT vs {nm}: {a}-{b}-{d}  winrate {(a + 0.5 * d) / (a + b + d):.0%}", flush=True)
print("EVAL-CKPT-DONE", flush=True)
