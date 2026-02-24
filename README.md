# Tau-Router

Invariant-routed long-context memory using Tau basin partitioning (gcd/divisor routing).

Tau-Router is a Python package and benchmark suite for deterministic memory partitioning and retrieval. It includes:

- Tau-Basin Retrieval (fast same-basin retrieval)
- ANN benchmark mode (Tau vs brute-force, FAISS, HNSW)
- Basin-local attention (NumPy / PyTorch / MLX backends)
- Persistence + mmap memory store
- Tau-Nano toy training demo
- CLI tools and benchmark scripts

Branding: Tau-Router
Backward compatibility is preserved for both `tau_router` and `tau_lattice` imports.

---

## Why this exists

Tau-Router routes tokens/states into a fixed set of divisor basins using a deterministic invariant:

- Routing key: `g = gcd(x0, x1, k)`
- Number of basins: `tau(k)` (the divisor count of `k`)

This creates a structured, partitioned memory layout that can reduce retrieval and attention work by searching one basin instead of scanning everything.

---

## Install

pip install tau-router

Optional extras:

pip install "tau-router[ann]"    # hnswlib
pip install "tau-router[torch]"  # PyTorch backend
pip install "tau-router[mlx]"    # Apple Silicon MLX backend

FAISS is optional and platform-dependent (`faiss` / `faiss-cpu`).

---

## Quick start

1) Infinite-context style demo

tau-infinite --tokens 200000

2) Retrieval baseline comparison (Tau vs naive full scan)

tau-infinite --tokens 200000 --compare-baseline

3) ANN-style benchmark (Tau vs exact brute-force / FAISS / HNSW)

tau-infinite --tokens 200000 --compare-ann-baselines

4) Basin-local attention benchmark

tau-infinite --tokens 200000 --attention-tokens 4096 --attention-backend numpy
# or: torch / mlx

5) Persist / reload with mmap

tau-infinite --tokens 200000 --save-mmap ./tau_store
tau-infinite --load-mmap ./tau_store --compare-baseline

6) Tau-Nano toy training demo

tau-nano --backend numpy
# optional
tau-nano --backend torch
tau-nano --backend mlx

---

## Example benchmark claim (synthetic systems benchmark)

Tau’s ANN benchmark is a deterministic synthetic vector benchmark with an explicit Tau-friendly basin-orthogonal block. It is designed to test the systems mechanism (routing, partitioning, candidate reduction), not semantic language quality.

Typical outcomes from the benchmark:
- Exact Recall@k = 1.0 vs brute-force
- Significant candidate reduction vs global scan
- Strong query-time speedups vs brute-force and FAISS exact
- Competitive query time vs HNSW (with much lower/no heavy index build path)

---

## Python API

from tau_router import TauRouter, TauPartitionedMemory
from tau_router.core import generate_synthetic_states

router = TauRouter(55_440)
mem = TauPartitionedMemory(router)

tok, x0, x1 = generate_synthetic_states(10000, router.k, seed=0)
mem.append_batch(tok, x0, x1)

out = mem.retrieve_same_basin(27720, 83160, max_tokens=256)
print(out["positions"][:5])

Backward-compatible imports also work:

from tau_lattice import TauRouter

---

## Multi-scale benchmark workflow

Run multiple scales and save logs:

bash scripts/run_multiscale_bench.sh

Parse logs into CSV / Markdown:

python scripts/parse_bench_logs.py bench_runs/*.txt --csv bench_summary.csv --markdown bench_summary.md

---

## Repo structure

src/
  tau_lattice/   # main implementation (legacy namespace)
  tau_router/    # new namespace + compatibility wrapper
tests/
examples/
scripts/

Why both packages?
- `tau_router` is the new project/package name
- `tau_lattice` remains for backward compatibility with earlier demos and imports

---

## Development

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest

Build release artifacts:

python -m build

Upload to PyPI:

python -m twine upload dist/*

---

## Naming

- Project: Tau-Router
- Mechanism: Tau-Basin Retrieval
- Attention kernel: Tau-Attention

---

## License

MIT
