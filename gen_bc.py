"""Generate ProAgent BC data + behavioral-clone CRStarNet -> checkpoints/bc_pro.pt.
Run on the GPU pod before train_v2 --init-checkpoint."""
import os

import torch

from eval.baseline_agents import ProAgent
from model.transformer_net import CRStarNet
from training.imitation import (
    ExpertDataset,
    ImitationConfig,
    generate_expert_dataset,
    train_behavioral_cloning,
)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("generating ProAgent BC data...", flush=True)
generate_expert_dataset("data/expert_pro.npz", n_games=400,
                        expert=ProAgent(), opponent=ProAgent(), seed=0)
ds = ExpertDataset("data/expert_pro.npz")
print(f"BC dataset: {len(ds)} samples", flush=True)
# architecture MUST match train_v2's CRStarNet so the checkpoint loads
model = CRStarNet(spatial_blocks=10, spatial_filters=128,
                  core_hidden=512, core_layers=2).to(dev)
cfg = ImitationConfig(batch_size=1024, num_epochs=8, learning_rate=1e-3)
train_behavioral_cloning(model, ds, cfg, dev)
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/bc_pro.pt")
print("BC-CKPT-SAVED", flush=True)
