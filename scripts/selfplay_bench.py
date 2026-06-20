#!/usr/bin/env python3
"""Benchmark vectorized self-play throughput and the batching speedup.

Runs the same number of games at several ``n_envs`` (batch) sizes and reports
games/sec. Batching across environments amortizes the network forward pass, so
throughput should climb with batch size — the effect is largest on a GPU
(``--device cuda``), where a single batched forward replaces many small ones.

    python scripts/selfplay_bench.py --games 16 --device cpu
    python scripts/selfplay_bench.py --games 256 --device cuda --backend rust
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from model.network import CRZeroNet
from training.vectorized_selfplay import (
    VectorizedSelfPlay,
    VectorizedSelfPlayConfig,
    rust_backend_available,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--backend", type=str, default="python", choices=["python", "rust"])
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64]
    )
    args = parser.parse_args()

    if args.backend == "rust" and not rust_backend_available():
        print("rust backend requested but cr_engine_native is not built; "
              "build it with `cd cr_engine && maturin build --release "
              "--features python && pip install target/wheels/*.whl`")
        return

    net = CRZeroNet(n_res_blocks=args.res_blocks, n_filters=args.filters)
    net.eval()

    print(f"device={args.device} backend={args.backend} games={args.games} "
          f"net={args.res_blocks}x{args.filters} max_ticks={args.max_ticks}\n")
    print(f"{'n_envs':>8} {'games/sec':>12} {'speedup':>10}")
    base = None
    for n_envs in args.batch_sizes:
        cfg = VectorizedSelfPlayConfig(
            n_envs=n_envs, max_ticks=args.max_ticks, backend=args.backend
        )
        runner = VectorizedSelfPlay(net, cfg, device=args.device, seed=0)
        t0 = time.time()
        runner.generate(args.games)
        dt = time.time() - t0
        rate = args.games / dt
        base = base or rate
        print(f"{n_envs:>8} {rate:>12.2f} {rate / base:>9.2f}x")


if __name__ == "__main__":
    main()
