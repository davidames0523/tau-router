from __future__ import annotations

import argparse

from tau_lattice.nano import run_train


def main():
    p = argparse.ArgumentParser(description="Tau-Nano training demo with numpy / torch / mlx backends")
    p.add_argument("--k", type=int, default=55_440)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["numpy", "torch", "mlx"], default="numpy")
    args = p.parse_args()
    run_train(
        k=args.k,
        steps=args.steps,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        lr=args.lr,
        seed=args.seed,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
