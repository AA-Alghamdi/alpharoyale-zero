"""Training infrastructure for Clash Royale AlphaZero.

Modules:
  - league: AlphaStar-style league training with PFSP
  - curriculum: Progressive difficulty curriculum
  - parallel_sim: CPU workers + GPU batch inference
  - domain_randomization: Sim-to-real robustness
  - opponent_model: Belief state for imperfect information
  - imitation: Warm-start from human data
  - meta_adaptation: Surgery + fine-tune for balance patches
  - real_device: KataCR-inspired perception + control pipeline
"""
