from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from tau_lattice.attention import basin_local_attention, global_attention, attention_backend_available
from tau_lattice.baselines import (
    naive_retrieve_same_basin_flat,
    make_demo_embeddings,
    faiss_available,
    hnsw_available,
    make_tau_orthogonal_embeddings,
    exact_topk_batch_ip,
    tau_prefilter_exact_topk,
    faiss_flat_ip_search,
    hnsw_cosine_search,
    recall_at_k,
    sample_query_indices_by_basin,
)
from tau_lattice.core import (
    TauRouter,
    TauPartitionedMemory,
    TauMemmapMemory,
    explicit_cycle_check,
    generate_synthetic_states,
)


def _format_bytes_gb(x: float) -> str:
    return f"{x:.3f} GB"


def _print_build_stats(mem, elapsed_s: float | None):
    stats = mem.memory_stats()
    print("\n--- Build Stats ---" if elapsed_s is not None else "\n--- Load Stats ---")
    if elapsed_s is not None:
        print(f"Build time: {elapsed_s*1000:.2f} ms")
        if stats["total_tokens"]:
            print(f"Throughput: {stats['total_tokens']/max(elapsed_s,1e-9)/1e6:.2f} M tokens/s")
    print(f"Approx allocated memory: {_format_bytes_gb(stats['allocated_gb'])}")
    print(f"Non-empty basins: {stats['nonempty_basins']} / {stats['tau_k']}")
    if stats.get("memory_mapped"):
        print("Storage mode: memory-mapped (read-only)")
    print("Top loaded basins (basin_id, g, count):")
    for item in mem.top_loaded_basins(8):
        print("  ", item)


def _retrieval_probe(mem, router: TauRouter, compare_baseline: bool = False, max_tokens: int = 256):
    qx0, qx1 = 27720, 83160
    basin_id = router.route_scalar(qx0, qx1)
    g = router.g_of_basin(basin_id)

    t0 = time.perf_counter()
    out = mem.retrieve_same_basin(qx0, qx1, max_tokens=max_tokens)
    t1 = time.perf_counter()

    print("\n--- Retrieval Probe (same-basin) ---")
    print(f"Query routes to basin {basin_id} (g={g})")
    print(f"Retrieved {len(out['token_ids'])} tokens in {(t1-t0)*1000:.3f} ms")
    if len(out["positions"]):
        print(f"Newest position returned: {int(out['positions'][0])}")
        print(f"Oldest position returned: {int(out['positions'][-1])}")

    if compare_baseline:
        flat = mem.to_flat_arrays(sort_by_position=True)
        t2 = time.perf_counter()
        base = naive_retrieve_same_basin_flat(flat, router.k, g, max_tokens=max_tokens)
        t3 = time.perf_counter()
        print("\n--- Baseline Comparison (retrieval) ---")
        print(f"Naive full-scan retrieve: {(t3-t2)*1000:.3f} ms (scan all {len(flat['token_ids']):,} tokens)")
        speedup = ((t3 - t2) / max(t1 - t0, 1e-12)) if (t1 - t0) > 0 else float('inf')
        print(f"Tau basin retrieve speedup vs naive scan: {speedup:.1f}x")
        if len(base["positions"]) and len(out["positions"]):
            overlap = len(set(base["positions"].tolist()) & set(out["positions"].tolist()))
            print(f"Overlap with naive top-{max_tokens}: {overlap}/{min(len(base['positions']), len(out['positions']))}")


def _ann_retrieval_benchmark(
    mem,
    router: TauRouter,
    *,
    sample_tokens: int = 200_000,
    n_queries: int = 25,
    top_k: int = 64,
    local_dim: int = 32,
    basin_scale: float = 8.0,
    seed: int = 123,
    hnsw_ef_search: int = 256,
):
    """Rigorous vector retrieval benchmark against exact, FAISS, and HNSW.

    This benchmark intentionally constructs deterministic vectors with an explicit
    orthogonal basin block so the retrieval task matches the conjecture-backed
    systems assumption (cross-basin orthogonality). It then compares:
      - Exact brute-force global top-k (ground truth)
      - Tau prefilter + exact rerank within one basin
      - FAISS exact (IndexFlatIP)
      - HNSW ANN (hnswlib)
    """
    flat = mem.to_flat_arrays(sort_by_position=True)
    n_total = len(flat["token_ids"])
    if n_total == 0:
        print("\n--- ANN Baseline Benchmark ---\nNo tokens in memory.")
        return

    n = min(int(sample_tokens), n_total)
    take = np.arange(n_total - n, n_total, dtype=np.int64)
    token_ids = flat["token_ids"][take]
    basin_ids = flat["basin_ids"][take]
    x0 = flat["x0"][take]
    x1 = flat["x1"][take]

    # Build deterministic vectors whose geometry encodes the Tau orthogonality assumption.
    t0 = time.perf_counter()
    vecs = make_tau_orthogonal_embeddings(
        token_ids,
        basin_ids,
        tau_k=router.num_basins,
        local_dim=int(local_dim),
        basin_scale=float(basin_scale),
    )
    t1 = time.perf_counter()
    emb_build_ms = (t1 - t0) * 1000.0

    q_idx = sample_query_indices_by_basin(basin_ids, n_queries=int(n_queries), top_k=int(top_k), seed=int(seed))
    if q_idx.size == 0:
        print("\n--- ANN Baseline Benchmark ---\nUnable to sample any queries.")
        return
    queries = vecs[q_idx]

    # Ground truth (exact brute-force over the full sample).
    exact_idx, exact_scores, exact_batch_ms = exact_topk_batch_ip(vecs, queries, int(top_k))
    # Time a per-query exact loop as a more apples-to-apples systems baseline against Tau's
    # current Python/NumPy implementation (many small partitioned GEMMs instead of one large GEMM).
    t_exact0 = time.perf_counter()
    for qv in queries:
        scores = vecs @ qv
        kk = int(min(int(top_k), len(scores)))
        part = np.argpartition(scores, kth=scores.shape[0] - kk)[-kk:]
        _ = part[np.argsort(scores[part])[::-1]]
    t_exact1 = time.perf_counter()
    exact_loop_ms = (t_exact1 - t_exact0) * 1000.0

    # Tau prefilter exact rerank (same-basin only).
    tau_res = tau_prefilter_exact_topk(vecs, basin_ids, q_idx, int(top_k))
    k_eval = int(tau_res.get("k_eff", top_k))
    exact_idx_eval = exact_idx[:, :k_eval]
    tau_metrics = recall_at_k(tau_res["indices"], exact_idx_eval)
    tau_speedup = exact_loop_ms / max(float(tau_res["elapsed_ms"]), 1e-12)
    global_candidates = int(len(q_idx) * n)
    tau_candidates = int(tau_res.get("total_candidates", 0))
    candidate_reduction = (global_candidates / max(tau_candidates, 1)) if tau_candidates else float("inf")

    print("\n--- ANN Baseline Benchmark (Tau vs FAISS vs HNSW) ---")
    print("Task: cosine/IP top-k retrieval on deterministic vectors with an explicit Tau basin-orthogonal block")
    print(f"Sample size: {n:,} tokens | Queries: {len(q_idx)} | dim={vecs.shape[1]} (tau_k={router.num_basins} + local={local_dim})")
    print(f"Embedding build time (deterministic): {emb_build_ms:.2f} ms")
    print(f"Embedding matrix memory: {_format_bytes_gb(vecs.nbytes / (1024**3))}")
    print(f"Top-k target: {top_k} | Compared k: {k_eval}")
    print(f"Tau candidate set size (mean/min/max): {tau_res['mean_candidates']:.1f} / {tau_res['min_candidates']} / {tau_res['max_candidates']}")
    print(f"Candidate reduction vs global scan: {candidate_reduction:.1f}x ({global_candidates:,} scores -> {tau_candidates:,})")

    print("\nExact brute-force (ground truth over full sample):")
    print(f"  Query time (batch GEMM, used for ground truth labels): {exact_batch_ms:.2f} ms")
    print(f"  Query time (per-query loop): {exact_loop_ms:.2f} ms | avg/query: {exact_loop_ms/max(len(q_idx),1):.3f} ms")
    print(f"  Checksum: {float(np.sum(exact_scores)):.4f}")

    print("\nTau prefilter + exact rerank (same basin only):")
    print(f"  Query time (batch of {len(q_idx)}): {tau_res['elapsed_ms']:.2f} ms")
    print(f"  Avg/query: {tau_res['elapsed_ms']/max(len(q_idx),1):.3f} ms")
    print(f"  Speedup vs exact brute-force: {tau_speedup:.1f}x")
    print(
        f"  Recall@{k_eval}: {tau_metrics['mean_recall']:.3f} | "
        f"min recall: {tau_metrics['min_recall']:.3f} | exact-order match rate: {tau_metrics['exact_match_rate']:.3f}"
    )

    # FAISS exact baseline
    if faiss_available():
        try:
            faiss_res = faiss_flat_ip_search(vecs, queries, int(k_eval))
            faiss_metrics = recall_at_k(faiss_res["indices"], exact_idx_eval)
            print("\nFAISS (IndexFlatIP, exact):")
            print(f"  Build time: {faiss_res['build_ms']:.2f} ms")
            print(f"  Query time (batch of {len(q_idx)}): {faiss_res['query_ms']:.2f} ms")
            print(f"  Avg/query: {faiss_res['query_ms']/max(len(q_idx),1):.3f} ms")
            print(
                f"  Recall@{k_eval}: {faiss_metrics['mean_recall']:.3f} | "
                f"min recall: {faiss_metrics['min_recall']:.3f} | exact-order match rate: {faiss_metrics['exact_match_rate']:.3f}"
            )
            if faiss_res["query_ms"] > 0:
                print(f"  Tau vs FAISS query time ratio (FAISS/Tau): {faiss_res['query_ms']/max(tau_res['elapsed_ms'],1e-12):.1f}x")
        except Exception as e:
            print("\nFAISS (IndexFlatIP, exact):")
            print(f"  Failed to run FAISS benchmark: {e}")
    else:
        print("\nFAISS (IndexFlatIP, exact):")
        print("  Not available. Install an optional FAISS build (e.g., faiss-cpu) to run this comparison.")

    # HNSW ANN baseline
    if hnsw_available():
        try:
            hnsw_res = hnsw_cosine_search(vecs, queries, int(k_eval), ef_search=int(hnsw_ef_search))
            hnsw_metrics = recall_at_k(hnsw_res["indices"], exact_idx_eval)
            print("\nHNSW (hnswlib, ANN cosine):")
            print(f"  Build time: {hnsw_res['build_ms']:.2f} ms")
            print(f"  Query time (batch of {len(q_idx)}): {hnsw_res['query_ms']:.2f} ms")
            print(f"  Avg/query: {hnsw_res['query_ms']/max(len(q_idx),1):.3f} ms")
            print(
                f"  Recall@{k_eval}: {hnsw_metrics['mean_recall']:.3f} | "
                f"min recall: {hnsw_metrics['min_recall']:.3f} | exact-order match rate: {hnsw_metrics['exact_match_rate']:.3f}"
            )
            if hnsw_res["query_ms"] > 0:
                print(f"  Tau vs HNSW query time ratio (HNSW/Tau): {hnsw_res['query_ms']/max(tau_res['elapsed_ms'],1e-12):.1f}x")
            print(f"  HNSW ef_search: {int(hnsw_ef_search)}")
        except Exception as e:
            print("\nHNSW (hnswlib, ANN cosine):")
            print(f"  Failed to run HNSW benchmark: {e}")
    else:
        print("\nHNSW (hnswlib, ANN cosine):")
        print("  Not available. Install hnswlib to run this comparison.")

    # Small correctness cross-check on sampled queries for transparency.
    q_basins = basin_ids[q_idx]
    q_g = np.gcd(np.gcd(x0[q_idx], x1[q_idx]), np.int64(router.k))
    q_basin_via_g = np.array([router.route_scalar(int(a), int(b)) for a, b in zip(x0[q_idx], x1[q_idx])], dtype=np.int32)
    consistent = bool(np.all(q_basins == q_basin_via_g))
    unique_q_basins = int(np.unique(q_basins).size)
    print("\nSanity checks:")
    print(f"  Query basin ids consistent with router: {'PASS' if consistent else 'FAIL'}")
    print(f"  Queries span {unique_q_basins} basins | distinct gcd values sampled: {int(np.unique(q_g).size)}")


def _attention_benchmark(mem, backend: str = "numpy", attn_tokens: int = 4096, d_model: int = 32, causal: bool = False, compare_global: bool = True, full_basin_only: bool = False):
    if not attention_backend_available(backend):
        print(f"\n--- Attention Benchmark ---\nBackend '{backend}' unavailable on this machine (install optional dependency).")
        return

    flat = mem.to_flat_arrays(sort_by_position=True)
    n_total = len(flat["token_ids"])
    if n_total == 0:
        print("\n--- Attention Benchmark ---\nNo tokens in memory.")
        return

    if full_basin_only:
        idx = np.arange(n_total)
        compare_global = False
        label = f"full context ({n_total:,} tokens)"
    else:
        n = min(int(attn_tokens), n_total)
        idx = np.arange(n_total - n, n_total)
        label = f"sample of last {n:,} tokens"

    basin_ids = flat["basin_ids"][idx]
    token_ids = flat["token_ids"][idx]
    emb = make_demo_embeddings(token_ids, basin_ids, d_model=d_model)
    q = emb
    k = emb
    v = emb

    print("\n--- Basin-Local Attention Benchmark ---")
    print(f"Backend: {backend} | Mode: {label} | d_model={d_model}")
    t0 = time.perf_counter()
    out_tau, stats = basin_local_attention(q, k, v, basin_ids, backend=backend, causal=causal)
    t1 = time.perf_counter()
    print(f"Tau basin-local attention time: {(t1-t0)*1000:.2f} ms")
    print(f"Basins present: {stats['n_basins_present']} | max basin size: {stats['max_basin_size']} | mean basin size: {stats['mean_basin_size']:.1f}")
    print(f"Pair-op reduction vs global: {stats['pair_op_reduction']:.2f}x (N^2={stats['global_pair_ops']:,} vs Σ|B|²={stats['basin_pair_ops']:,})")
    print(f"Output checksum: {float(np.sum(out_tau)):.4f}")

    if compare_global:
        t2 = time.perf_counter()
        out_global = global_attention(q, k, v, backend=backend, causal=causal)
        t3 = time.perf_counter()
        diff = float(np.max(np.abs(out_tau - out_global)))
        print("\nGlobal baseline (same sample):")
        print(f"Global attention time: {(t3-t2)*1000:.2f} ms")
        print(f"Max |Tau - Global| on sample: {diff:.6f} (expected non-zero if multiple basins present)")
        speed = ((t3 - t2) / max(t1 - t0, 1e-12)) if (t1 - t0) > 0 else float('inf')
        print(f"Observed speedup (global / tau): {speed:.2f}x")


def run_demo(
    n_tokens: int = 10_000_000,
    k: int = 55_440,
    batch_size: int = 250_000,
    chunk_size: int = 65_536,
    seed: int = 0,
    do_retrieval_probe: bool = True,
    compare_baseline: bool = False,
    compare_ann_baselines: bool = False,
    ann_sample_tokens: int = 200_000,
    ann_queries: int = 25,
    ann_top_k: int = 64,
    ann_local_dim: int = 32,
    ann_basin_scale: float = 8.0,
    ann_seed: int = 123,
    ann_hnsw_ef_search: int = 256,
    attention_backend: str = "numpy",
    attention_tokens: int = 0,
    attention_d_model: int = 32,
    attention_causal: bool = False,
    full_basin_attention: bool = False,
    save_mmap: str | None = None,
    load_mmap: str | None = None,
):
    router = TauRouter(k)

    print("Tau-Router Infinite Context Demo")
    print(f"k={k}  tau(k)={router.num_basins} basins")
    print(f"cycle sanity check: {'PASS' if explicit_cycle_check(router) else 'FAIL'}")

    if load_mmap:
        mem = TauMemmapMemory.load(load_mmap)
        print(f"Loaded memory-mapped store: {load_mmap}")
        print(f"Target context: {mem.total_tokens:,} tokens (from disk)")
        print(f"Chunk size: {mem.chunk_size:,}")
        _print_build_stats(mem, elapsed_s=None)
    else:
        mem = TauPartitionedMemory(router, chunk_size=chunk_size, d_model=0)
        print(f"Target context: {n_tokens:,} tokens")
        print(f"Batch size: {batch_size:,}  Chunk size: {chunk_size:,}")
        t0 = time.perf_counter()
        processed = 0
        batch_idx = 0
        while processed < n_tokens:
            n = min(batch_size, n_tokens - processed)
            token_ids, x0, x1 = generate_synthetic_states(n, k, seed=seed + batch_idx)
            mem.append_batch(token_ids, x0, x1)
            processed += n
            batch_idx += 1
        t1 = time.perf_counter()
        _print_build_stats(mem, elapsed_s=(t1 - t0))

        if save_mmap:
            save_path = Path(save_mmap)
            mem.save(save_path, overwrite=True)
            print(f"\nPersisted memory store to: {save_path}")
            reloaded = TauMemmapMemory.load(save_path)
            print(f"Reload check: loaded {reloaded.total_tokens:,} tokens via mmap")

    if do_retrieval_probe:
        _retrieval_probe(mem, router, compare_baseline=compare_baseline)

    if compare_ann_baselines:
        _ann_retrieval_benchmark(
            mem,
            router,
            sample_tokens=ann_sample_tokens,
            n_queries=ann_queries,
            top_k=ann_top_k,
            local_dim=ann_local_dim,
            basin_scale=ann_basin_scale,
            seed=ann_seed,
            hnsw_ef_search=ann_hnsw_ef_search,
        )

    if attention_tokens > 0 or full_basin_attention:
        _attention_benchmark(
            mem,
            backend=attention_backend,
            attn_tokens=attention_tokens if attention_tokens > 0 else 4096,
            d_model=attention_d_model,
            causal=attention_causal,
            compare_global=not full_basin_attention,
            full_basin_only=full_basin_attention,
        )


def main():
    p = argparse.ArgumentParser(description="Tau-Router infinite-context retrieval and basin-local attention demo")
    p.add_argument("--tokens", type=int, default=10_000_000)
    p.add_argument("--k", type=int, default=55_440)
    p.add_argument("--batch-size", type=int, default=250_000)
    p.add_argument("--chunk-size", type=int, default=65_536)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--compare-baseline", action="store_true", help="Compare Tau same-basin retrieval to naive full-scan retrieval")
    p.add_argument("--compare-ann-baselines", action="store_true", help="Benchmark Tau vector retrieval vs exact brute-force, FAISS, and HNSW")
    p.add_argument("--ann-sample-tokens", type=int, default=200_000, help="Number of recent tokens to include in the ANN benchmark sample")
    p.add_argument("--ann-queries", type=int, default=25, help="Number of queries for ANN benchmark")
    p.add_argument("--ann-top-k", type=int, default=64, help="Top-k for ANN benchmark")
    p.add_argument("--ann-local-dim", type=int, default=32, help="Local feature dimensions appended after the tau_k orthogonal basin block")
    p.add_argument("--ann-basin-scale", type=float, default=8.0, help="Scale of the orthogonal basin block (higher enforces stronger basin separation)")
    p.add_argument("--ann-seed", type=int, default=123, help="Seed for query sampling in ANN benchmark")
    p.add_argument("--ann-hnsw-ef-search", type=int, default=256, help="hnswlib ef_search for ANN benchmark")
    p.add_argument("--attention-backend", choices=["numpy", "torch", "mlx"], default="numpy")
    p.add_argument("--attention-tokens", type=int, default=0, help="Run basin-local attention on a recent sample of this size")
    p.add_argument("--attention-d-model", type=int, default=32)
    p.add_argument("--attention-causal", action="store_true")
    p.add_argument("--full-basin-attention", action="store_true", help="Run basin-local attention on the full context (no global baseline)")
    p.add_argument("--save-mmap", type=str, default=None, help="Directory to persist memory store as .npy files")
    p.add_argument("--load-mmap", type=str, default=None, help="Load a previously saved memory store with mmap and skip build")
    args = p.parse_args()

    run_demo(
        n_tokens=args.tokens,
        k=args.k,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        seed=args.seed,
        do_retrieval_probe=not args.no_probe,
        compare_baseline=args.compare_baseline,
        compare_ann_baselines=args.compare_ann_baselines,
        ann_sample_tokens=args.ann_sample_tokens,
        ann_queries=args.ann_queries,
        ann_top_k=args.ann_top_k,
        ann_local_dim=args.ann_local_dim,
        ann_basin_scale=args.ann_basin_scale,
        ann_seed=args.ann_seed,
        ann_hnsw_ef_search=args.ann_hnsw_ef_search,
        attention_backend=args.attention_backend,
        attention_tokens=args.attention_tokens,
        attention_d_model=args.attention_d_model,
        attention_causal=args.attention_causal,
        full_basin_attention=args.full_basin_attention,
        save_mmap=args.save_mmap,
        load_mmap=args.load_mmap,
    )


if __name__ == "__main__":
    main()
