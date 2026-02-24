from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np


def naive_retrieve_same_basin_flat(
    flat: Dict[str, np.ndarray],
    k: int,
    query_g: int,
    *,
    max_tokens: int = 1024,
    window: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """Baseline retrieval that scans all tokens and filters by gcd == query_g."""
    x0 = flat["x0"]
    x1 = flat["x1"]
    pos = flat["positions"]
    tok = flat["token_ids"]
    g = np.gcd(np.gcd(x0, x1), np.int64(k))
    mask = (g == np.int64(query_g))
    if window is not None:
        mask &= (pos >= int(window[0])) & (pos <= int(window[1]))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {
            "positions": np.empty(0, dtype=np.int64),
            "token_ids": np.empty(0, dtype=np.int32),
            "x0": np.empty(0, dtype=np.int64),
            "x1": np.empty(0, dtype=np.int64),
        }
    # newest first
    idx = idx[np.argsort(pos[idx])[::-1]]
    idx = idx[: int(max_tokens)]
    return {
        "positions": pos[idx],
        "token_ids": tok[idx],
        "x0": x0[idx],
        "x1": x1[idx],
    }


def make_demo_embeddings(token_ids: np.ndarray, basin_ids: np.ndarray, d_model: int = 32) -> np.ndarray:
    """Deterministic small embeddings for attention benchmarking without training."""
    token_ids = np.asarray(token_ids, dtype=np.int64)
    basin_ids = np.asarray(basin_ids, dtype=np.int64)
    idx = np.arange(int(d_model), dtype=np.float32)[None, :] + 1.0
    t = (token_ids[:, None] % 997).astype(np.float32)
    b = (basin_ids[:, None] % 991).astype(np.float32)
    emb = np.sin(0.013 * idx * (t + 1)) + np.cos(0.017 * idx * (b + 3))
    emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= np.maximum(norms, 1e-6)
    return emb


# =========================
# ANN benchmark helpers (Tau vs exact vs FAISS vs HNSW)
# =========================


def faiss_available() -> bool:
    try:
        import faiss  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False



def hnsw_available() -> bool:
    try:
        import hnswlib  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False



def make_tau_orthogonal_embeddings(
    token_ids: np.ndarray,
    basin_ids: np.ndarray,
    tau_k: int,
    *,
    local_dim: int = 32,
    basin_scale: float = 8.0,
) -> np.ndarray:
    """Construct deterministic vectors with explicit basin orthogonality.

    Layout: [one-hot basin block of size tau_k | local features of size local_dim].
    With basin_scale >> local norm, global top-k under cosine/IP is guaranteed to lie
    in the query's basin, matching the conjecture-based orthogonality assumption.
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    basin_ids = np.asarray(basin_ids, dtype=np.int64)
    n = int(len(token_ids))
    local_dim = int(local_dim)
    tau_k = int(tau_k)
    d = tau_k + local_dim
    out = np.zeros((n, d), dtype=np.float32)
    if n == 0:
        return out

    rows = np.arange(n, dtype=np.int64)
    out[rows, basin_ids] = float(basin_scale)

    if local_dim > 0:
        idx = (np.arange(local_dim, dtype=np.float32)[None, :] + 1.0)
        t = (token_ids[:, None] % 4099).astype(np.float32)
        b = (basin_ids[:, None] % 997).astype(np.float32)
        r = (rows[:, None] % 8191).astype(np.float32)
        local = (
            np.sin(0.0073 * idx * (t + 3.0))
            + np.cos(0.0111 * idx * (b + 5.0))
            + 0.5 * np.sin(0.0059 * idx * (r + 7.0))
        )
        local = local.astype(np.float32)
        local_norm = np.linalg.norm(local, axis=1, keepdims=True)
        local = local / np.maximum(local_norm, 1e-6)  # unit local norm
        out[:, tau_k:] = local

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(norms, 1e-6)
    return out.astype(np.float32, copy=False)



def exact_topk_batch_ip(vectors: np.ndarray, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Exact top-k by inner product for a batch of queries. Returns (idx, scores, elapsed_ms)."""
    x = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    q = np.ascontiguousarray(queries.astype(np.float32, copy=False))
    n = x.shape[0]
    k = int(min(max(1, top_k), n))
    t0 = time.perf_counter()
    scores = q @ x.T  # [Q, N]
    part = np.argpartition(scores, kth=scores.shape[1] - k, axis=1)[:, -k:]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(part_scores, axis=1)[:, ::-1]
    idx = np.take_along_axis(part, order, axis=1)
    top_scores = np.take_along_axis(part_scores, order, axis=1)
    t1 = time.perf_counter()
    return idx.astype(np.int64, copy=False), top_scores.astype(np.float32, copy=False), (t1 - t0) * 1000.0



def tau_prefilter_exact_topk(
    vectors: np.ndarray,
    basin_ids: np.ndarray,
    query_indices: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    """Exact top-k by inner product, but only within the query's Tau basin.

    Implementation groups queries by basin to avoid Python overhead and to make the
    benchmark reflect the intended systems path (batched GEMMs on small partitions).
    """
    x = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    bas = np.asarray(basin_ids, dtype=np.int32)
    qidx = np.asarray(query_indices, dtype=np.int64)
    n = x.shape[0]
    if n == 0 or qidx.size == 0:
        return {
            "indices": np.empty((0, 0), dtype=np.int64),
            "scores": np.empty((0, 0), dtype=np.float32),
            "elapsed_ms": 0.0,
            "mean_candidates": 0.0,
            "max_candidates": 0,
            "min_candidates": 0,
        }

    unique_basins = np.unique(bas)
    basin_to_idx: Dict[int, np.ndarray] = {int(b): np.flatnonzero(bas == b) for b in unique_basins}
    k_req = int(max(1, top_k))
    min_candidates = min(int(len(basin_to_idx[int(bas[int(i)])])) for i in qidx)
    k_eff = min(k_req, min_candidates)

    out_idx = np.empty((len(qidx), k_eff), dtype=np.int64)
    out_scores = np.empty((len(qidx), k_eff), dtype=np.float32)
    cand_sizes: List[int] = []

    q_basins = bas[qidx]
    q_order = np.argsort(q_basins, kind="stable")
    qidx_sorted = qidx[q_order]
    qbas_sorted = q_basins[q_order]
    starts = np.flatnonzero(np.r_[True, qbas_sorted[1:] != qbas_sorted[:-1]])
    ends = np.r_[starts[1:], np.array([len(qidx_sorted)])]

    t0 = time.perf_counter()
    for s, e in zip(starts.tolist(), ends.tolist()):
        b = int(qbas_sorted[s])
        cand = basin_to_idx[b]
        cand_sizes.extend([int(len(cand))] * (e - s))
        q_rows = qidx_sorted[s:e]
        qv = x[q_rows]                       # [Qb, D]
        candv = x[cand]                      # [Cb, D]
        scores = qv @ candv.T               # [Qb, Cb]
        part = np.argpartition(scores, kth=scores.shape[1] - k_eff, axis=1)[:, -k_eff:]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(part_scores, axis=1)[:, ::-1]
        top_local = np.take_along_axis(part, order, axis=1)
        top_scores = np.take_along_axis(part_scores, order, axis=1)
        orig_rows = q_order[s:e]
        out_idx[orig_rows] = cand[top_local]
        out_scores[orig_rows] = top_scores
    t1 = time.perf_counter()

    return {
        "indices": out_idx,
        "scores": out_scores,
        "elapsed_ms": (t1 - t0) * 1000.0,
        "mean_candidates": float(np.mean(cand_sizes)) if cand_sizes else 0.0,
        "max_candidates": int(max(cand_sizes)) if cand_sizes else 0,
        "min_candidates": int(min(cand_sizes)) if cand_sizes else 0,
        "k_eff": int(k_eff),
        "total_candidates": int(sum(cand_sizes)) if cand_sizes else 0,
    }


def faiss_flat_ip_search(vectors: np.ndarray, queries: np.ndarray, top_k: int) -> Dict[str, Any]:
    """FAISS exact baseline (IndexFlatIP)."""
    import faiss  # type: ignore

    x = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    q = np.ascontiguousarray(queries.astype(np.float32, copy=False))
    d = int(x.shape[1])
    k = int(min(max(1, top_k), x.shape[0]))

    index = faiss.IndexFlatIP(d)
    t0 = time.perf_counter()
    index.add(x)
    t1 = time.perf_counter()
    sims, idx = index.search(q, k)
    t2 = time.perf_counter()
    return {
        "indices": idx.astype(np.int64, copy=False),
        "scores": sims.astype(np.float32, copy=False),
        "build_ms": (t1 - t0) * 1000.0,
        "query_ms": (t2 - t1) * 1000.0,
    }



def hnsw_cosine_search(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    *,
    ef_construction: int = 200,
    M: int = 32,
    ef_search: int = 256,
) -> Dict[str, Any]:
    """hnswlib ANN baseline over normalized vectors using cosine distance."""
    import hnswlib  # type: ignore

    x = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    q = np.ascontiguousarray(queries.astype(np.float32, copy=False))
    n, d = x.shape
    k = int(min(max(1, top_k), n))

    index = hnswlib.Index(space="cosine", dim=d)
    t0 = time.perf_counter()
    index.init_index(max_elements=n, ef_construction=int(ef_construction), M=int(M))
    labels = np.arange(n, dtype=np.int32)
    index.add_items(x, labels)
    index.set_ef(max(int(ef_search), k))
    t1 = time.perf_counter()
    idx, dists = index.knn_query(q, k=k)
    t2 = time.perf_counter()
    # hnswlib cosine returns distance = 1 - cosine_similarity for normalized vectors.
    scores = 1.0 - dists.astype(np.float32, copy=False)
    return {
        "indices": idx.astype(np.int64, copy=False),
        "scores": scores,
        "build_ms": (t1 - t0) * 1000.0,
        "query_ms": (t2 - t1) * 1000.0,
    }



def recall_at_k(pred_indices: np.ndarray, true_indices: np.ndarray) -> Dict[str, float]:
    """Compute mean/min recall@k and exact-match@k across query batches."""
    pred = np.asarray(pred_indices, dtype=np.int64)
    true = np.asarray(true_indices, dtype=np.int64)
    q = int(min(len(pred), len(true)))
    if q == 0:
        return {"mean_recall": 0.0, "min_recall": 0.0, "exact_match_rate": 0.0}
    recalls: List[float] = []
    exact = 0
    for i in range(q):
        p = [int(x) for x in pred[i].tolist() if int(x) >= 0]
        t = [int(x) for x in true[i].tolist() if int(x) >= 0]
        if not t:
            recalls.append(1.0)
            exact += 1
            continue
        inter = len(set(p) & set(t))
        recalls.append(inter / float(len(t)))
        if len(p) == len(t) and p == t:
            exact += 1
    return {
        "mean_recall": float(np.mean(recalls)),
        "min_recall": float(np.min(recalls)),
        "exact_match_rate": float(exact / q),
    }



def sample_query_indices_by_basin(
    basin_ids: np.ndarray,
    *,
    n_queries: int,
    top_k: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample queries from basins with at least top_k elements for fair comparisons."""
    bas = np.asarray(basin_ids, dtype=np.int32)
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(bas, return_counts=True)
    eligible = unique[counts >= int(max(1, top_k))]
    if eligible.size == 0:
        eligible = unique
    if eligible.size == 0:
        return np.empty(0, dtype=np.int64)

    basin_to_idx = {int(b): np.flatnonzero(bas == b) for b in eligible}
    # Cycle through shuffled basins for coverage, picking one random item from each.
    order = eligible.copy()
    rng.shuffle(order)
    picks: List[int] = []
    while len(picks) < int(n_queries):
        for b in order.tolist():
            arr = basin_to_idx[int(b)]
            picks.append(int(arr[rng.integers(0, len(arr))]))
            if len(picks) >= int(n_queries):
                break
        rng.shuffle(order)
    return np.asarray(picks, dtype=np.int64)
