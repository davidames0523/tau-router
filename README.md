# Tau-Router

Invariant-routed long-context memory using Tau basin partitioning (gcd/divisor routing).

Tau-Router is a Python package and benchmark suite for deterministic memory partitioning and retrieval. It includes:

- Tau-Basin Retrieval (fast same-basin retrieval)
- ANN benchmark mode (Tau vs brute-force, FAISS, HNSW)
- Basin-local attention (NumPy / PyTorch / MLX backends)
- Persistence + mmap memory store
- Tau-Nano toy training demo
- CLI tools and benchmark scripts

Branding: **Tau-Router**  
Backward compatibility is preserved for both `tau_router` and `tau_lattice` imports.

---

## Why this exists

Tau-Router routes tokens/states into a fixed set of divisor basins using a deterministic invariant.

Routing key:

$$
\Large g = \gcd(x_0, x_1, k)
$$

Number of basins:

$$
\Large \tau(k)
$$

where $\tau(k)$ is the number of positive divisors of $k$.

This creates a structured, partitioned memory layout that can reduce retrieval and attention work by searching one basin instead of scanning everything.

---

## Mathematical foundation (conjectural)

This project is motivated by an **experimental number-theoretic conjecture**. The routing mechanism is implemented and benchmarked as an engineering system, but the underlying mathematical claims should currently be treated as **conjectural** unless/until formally proven.

### Dynamical system

We study the integer recurrence

$$
\Large x_{n+1} = \gcd(x_n, x_{n-1}) + k
$$

for a fixed positive integer constant $k$.

Given an initial pair $(x_0, x_1)$, define the invariant candidate

$$
\Large g = \gcd(x_0, x_1, k)
$$

### Conjecture (routing / basin structure)

For fixed $k$, every trajectory is conjectured to eventually enter a length-3 attractor cycle determined uniquely by a divisor $g \mid k$:

$$
\Large
(k+g,\,k+g)
\;\to\;
(k+g,\,2k+g)
\;\to\;
(2k+g,\,k+g)
\;\to\;
(k+g,\,k+g)
\;\to\;\cdots
$$

Under this conjecture:

1. The destination cycle is determined by

$$
\Large g = \gcd(x_0, x_1, k)
$$

2. The number of distinct attractor cycles equals the divisor-count function

$$
\Large \tau(k)
$$

where $\tau(k)$ is the number of positive divisors of $k$.

3. The state space is partitioned into disjoint basins indexed by divisors of $k$:

$$
\Large \mathcal{B}_g = \left\{(x_0, x_1) : \gcd(x_0, x_1, k) = g \right\}, \qquad g \mid k
$$

### Why this matters for the system

Tau-Router uses the divisor index $g$ as a deterministic routing key.

- Routing key: `g = gcd(x0, x1, k)`
- Number of partitions: `tau(k)`
- Memory partition: each token/state is stored in the basin indexed by `g`

This turns routing into a constant-time arithmetic operation (no learned router and no nearest-neighbor index build required for the Tau path).

### Retrieval scaling intuition

If a query routes to basin $\mathcal{B}_g$, retrieval only scans that basin rather than the full memory.

Let $N$ be the total number of stored items and $|\mathcal{B}_g|$ the basin size. Then the work is reduced from roughly

$$
\Large O(N)
$$

to

$$
\Large O(|\mathcal{B}_g|)
$$

If basin occupancy is reasonably balanced, then a rough average is

$$
\Large |\mathcal{B}_g| \approx \frac{N}{\tau(k)}
$$

which yields an expected candidate reduction of approximately

$$
\Large \tau(k)
$$

(often larger in practice for some query distributions).

### Attention partitioning (assumption used in synthetic benchmarks)

Some benchmark modes in this repository use a synthetic embedding construction with an explicit basin-orthogonal block, i.e., vectors from different basins are constructed to be orthogonal in that subspace. This is used to test the systems effect of partitioned retrieval/attention.

Formally, in the synthetic benchmark setup, for basin labels $g_q \neq g_x$ we enforce

$$
\Large \langle \phi(q), \phi(x) \rangle_{\text{basin-block}} = 0
$$

This does **not** claim that natural-language embeddings are inherently basin-orthogonal. It is a controlled benchmark assumption used to measure the routing/partitioning mechanism itself.

### Status and scope

- The recurrence behavior and basin structure are the motivating conjectural foundation.
- The software and benchmarks in this repo are valid as an engineering implementation of deterministic invariant-routed partitioning.
- Benchmark results should be interpreted as:
  - **Systems evidence** for partitioned retrieval/attention efficiency, and
  - **Not yet** a proof of semantic superiority on real language tasks.

If you use this project in research or production, please describe clearly which parts are:
1) conjectural math assumptions,
2) synthetic benchmark assumptions, and
3) empirically measured system performance.

---

## Install (optional)

FAISS is platform-dependent (`faiss` / `faiss-cpu`).

---

## Local setup (from source)

1) Clone and enter the repo

```bash
git clone https://github.com/davidames0523/tau-router.git
cd tau-router
```

2) Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Install in editable mode (so code changes apply immediately)

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[ann]"    # HNSW benchmark support
pip install -e ".[torch]"  # PyTorch backend
pip install -e ".[mlx]"    # Apple Silicon MLX backend
pip install -e ".[dev]"    # tests / dev tools
```

4) Verify the CLI commands are available

```bash
tau-infinite --help
tau-nano --help
```

5) Run a quick sanity demo

```bash
tau-infinite --tokens 200000
tau-nano
```

Notes:
- If `tau-infinite` is not found, make sure your virtual environment is activated.
- On macOS, if you reinstalled and commands seem stale, run `hash -r`.
- You do **not** need to build the package for local development (`pip install -e .` is enough).
- FAISS is optional and may require platform-specific installation (`faiss-cpu` or conda).

---

## Quick start

### 1) Infinite-context style demo

```bash
tau-infinite --tokens 200000
```

### 2) Retrieval baseline comparison (Tau vs naive full scan)

```bash
tau-infinite --tokens 200000 --compare-baseline
```

### 3) ANN-style benchmark (Tau vs exact brute-force / FAISS / HNSW)

```bash
tau-infinite --tokens 200000 --compare-ann-baselines
```

### 4) Basin-local attention benchmark

```bash
tau-infinite --tokens 200000 --attention-tokens 4096 --attention-backend numpy
# or: torch / mlx
```

### 5) Persist / reload with mmap

```bash
tau-infinite --tokens 200000 --save-mmap ./tau_store
tau-infinite --load-mmap ./tau_store --compare-baseline
```

### 6) Tau-Nano toy training demo

```bash
tau-nano --backend numpy
# optional
tau-nano --backend torch
tau-nano --backend mlx
```

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

```bash
from tau_router import TauRouter, TauPartitionedMemory
from tau_router.core import generate_synthetic_states

router = TauRouter(55_440)
mem = TauPartitionedMemory(router)

tok, x0, x1 = generate_synthetic_states(10000, router.k, seed=0)
mem.append_batch(tok, x0, x1)

out = mem.retrieve_same_basin(27720, 83160, max_tokens=256)
print(out["positions"][:5])
```

Backward-compatible imports also work:

```bash
from tau_lattice import TauRouter
```

---

## Multi-scale benchmark workflow

Run multiple scales and save logs:

```bash
mkdir -p bench_runs

for N in 50000 200000 1000000; do
  echo "=== Running N=$N ==="
  /usr/bin/time -l tau-infinite \
    --tokens "$N" \
    --compare-ann-baselines \
    --ann-sample-tokens "$N" \
    --ann-queries 25 \
    --ann-top-k 64 \
    | tee "bench_runs/tau_ann_${N}.txt"
done
```

Parse logs into CSV / Markdown (if `scripts/parse_bench_logs.py` is present):

```bash
python scripts/parse_bench_logs.py bench_runs/*.txt --csv bench_summary.csv --markdown bench_summary.md
```

---

## Repo structure

```bash
src/
  tau_lattice/   # main implementation (legacy namespace)
  tau_router/    # new namespace + compatibility wrapper
tests/
examples/
scripts/
```

Why both packages?
- `tau_router` is the new project/package name
- `tau_lattice` remains for backward compatibility with earlier demos and imports

---

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Build release artifacts (only needed for PyPI/distribution):

```bash
python -m build
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```

---

## Naming

- Project: Tau-Router
- Mechanism: Tau-Basin Retrieval
- Attention kernel: Tau-Attention

---

## License

MIT