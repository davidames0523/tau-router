from __future__ import annotations

from typing import Dict, Tuple
import math
import numpy as np


def attention_backend_available(backend: str) -> bool:
    b = backend.lower()
    if b == "numpy":
        return True
    if b == "torch":
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False
    if b == "mlx":
        try:
            import mlx.core as mx  # noqa: F401
            return True
        except Exception:
            return False
    return False


def _softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _global_attention_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False) -> np.ndarray:
    d = q.shape[-1]
    scores = (q @ k.T) / math.sqrt(max(1, d))
    if causal:
        mask = np.triu(np.ones((q.shape[0], q.shape[0]), dtype=bool), 1)
        scores = scores.copy()
        scores[mask] = -1e30
    probs = _softmax_numpy(scores, axis=-1)
    return probs @ v


def _basin_local_attention_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray, basin_ids: np.ndarray, causal: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    basin_ids = np.asarray(basin_ids, dtype=np.int32)
    n = q.shape[0]
    out = np.zeros_like(v, dtype=np.float32)
    unique, counts = np.unique(basin_ids, return_counts=True)
    basin_sq_ops = 0
    for b, c in zip(unique, counts):
        idx = np.flatnonzero(basin_ids == b)
        qb = q[idx]
        kb = k[idx]
        vb = v[idx]
        out[idx] = _global_attention_numpy(qb, kb, vb, causal=causal)
        basin_sq_ops += int(c) * int(c)
    stats = {
        "n_tokens": int(n),
        "n_basins_present": int(len(unique)),
        "global_pair_ops": int(n) * int(n),
        "basin_pair_ops": int(basin_sq_ops),
        "pair_op_reduction": (float(int(n) * int(n)) / float(max(1, basin_sq_ops))),
        "mean_basin_size": float(np.mean(counts)) if len(counts) else 0.0,
        "max_basin_size": int(np.max(counts)) if len(counts) else 0,
    }
    return out, stats


def _global_attention_torch(q_np, k_np, v_np, causal: bool = False):
    import torch

    q = torch.as_tensor(q_np, dtype=torch.float32)
    k = torch.as_tensor(k_np, dtype=torch.float32)
    v = torch.as_tensor(v_np, dtype=torch.float32)
    d = q.shape[-1]
    scores = (q @ k.transpose(0, 1)) / math.sqrt(max(1, int(d)))
    if causal:
        mask = torch.triu(torch.ones((q.shape[0], q.shape[0]), dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -1e30)
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    return out.detach().cpu().numpy()


def _basin_local_attention_torch(q_np, k_np, v_np, basin_ids_np, causal: bool = False):
    import torch

    q = torch.as_tensor(q_np, dtype=torch.float32)
    k = torch.as_tensor(k_np, dtype=torch.float32)
    v = torch.as_tensor(v_np, dtype=torch.float32)
    basin_ids = torch.as_tensor(basin_ids_np, dtype=torch.int64)
    n = int(q.shape[0])
    out = torch.zeros_like(v)
    unique, counts = torch.unique(basin_ids, return_counts=True)
    basin_sq_ops = 0
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(max(1, int(d)))
    for i in range(int(unique.numel())):
        b = unique[i]
        idx = torch.nonzero(basin_ids == b, as_tuple=False).squeeze(1)
        qb, kb, vb = q[idx], k[idx], v[idx]
        scores = (qb @ kb.transpose(0, 1)) * scale
        if causal:
            m = torch.triu(torch.ones((idx.numel(), idx.numel()), dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(m, -1e30)
        probs = torch.softmax(scores, dim=-1)
        out[idx] = probs @ vb
        c = int(counts[i].item())
        basin_sq_ops += c * c
    stats = {
        "n_tokens": n,
        "n_basins_present": int(unique.numel()),
        "global_pair_ops": n * n,
        "basin_pair_ops": int(basin_sq_ops),
        "pair_op_reduction": float(n * n) / float(max(1, basin_sq_ops)),
        "mean_basin_size": float(counts.float().mean().item()) if unique.numel() else 0.0,
        "max_basin_size": int(counts.max().item()) if unique.numel() else 0,
    }
    return out.detach().cpu().numpy(), stats


def _global_attention_mlx(q_np, k_np, v_np, causal: bool = False):
    import numpy as _np
    import mlx.core as mx

    q = mx.array(q_np.astype(_np.float32, copy=False))
    k = mx.array(k_np.astype(_np.float32, copy=False))
    v = mx.array(v_np.astype(_np.float32, copy=False))
    d = q.shape[-1]
    scores = mx.matmul(q, mx.transpose(k)) / math.sqrt(max(1, int(d)))
    if causal:
        m = _np.triu(_np.ones((int(q.shape[0]), int(q.shape[0])), dtype=_np.bool_), 1)
        scores = mx.where(mx.array(m), mx.array(-1e30, dtype=mx.float32), scores)
    scores = scores - mx.max(scores, axis=-1, keepdims=True)
    exp_scores = mx.exp(scores)
    probs = exp_scores / mx.sum(exp_scores, axis=-1, keepdims=True)
    out = mx.matmul(probs, v)
    mx.eval(out)
    return _np.array(out)


def _basin_local_attention_mlx(q_np, k_np, v_np, basin_ids_np, causal: bool = False):
    import numpy as _np
    import mlx.core as mx

    q = _np.asarray(q_np, dtype=_np.float32)
    k = _np.asarray(k_np, dtype=_np.float32)
    v = _np.asarray(v_np, dtype=_np.float32)
    basin_ids = _np.asarray(basin_ids_np, dtype=_np.int32)
    n = q.shape[0]
    out = _np.zeros_like(v, dtype=_np.float32)
    unique, counts = _np.unique(basin_ids, return_counts=True)
    basin_sq_ops = 0
    for b, c in zip(unique, counts):
        idx = _np.flatnonzero(basin_ids == b)
        q_m = mx.array(q[idx])
        k_m = mx.array(k[idx])
        v_m = mx.array(v[idx])
        d = q_m.shape[-1]
        scores = mx.matmul(q_m, mx.transpose(k_m)) / math.sqrt(max(1, int(d)))
        if causal:
            m = _np.triu(_np.ones((len(idx), len(idx)), dtype=_np.bool_), 1)
            scores = mx.where(mx.array(m), mx.array(-1e30, dtype=mx.float32), scores)
        scores = scores - mx.max(scores, axis=-1, keepdims=True)
        exp_scores = mx.exp(scores)
        probs = exp_scores / mx.sum(exp_scores, axis=-1, keepdims=True)
        o = mx.matmul(probs, v_m)
        mx.eval(o)
        out[idx] = _np.array(o)
        basin_sq_ops += int(c) * int(c)
    stats = {
        "n_tokens": int(n),
        "n_basins_present": int(len(unique)),
        "global_pair_ops": int(n) * int(n),
        "basin_pair_ops": int(basin_sq_ops),
        "pair_op_reduction": (float(int(n) * int(n)) / float(max(1, basin_sq_ops))),
        "mean_basin_size": float(_np.mean(counts)) if len(counts) else 0.0,
        "max_basin_size": int(_np.max(counts)) if len(counts) else 0,
    }
    return out, stats


def global_attention(q, k, v, *, backend: str = "numpy", causal: bool = False):
    b = backend.lower()
    if b == "numpy":
        return _global_attention_numpy(np.asarray(q), np.asarray(k), np.asarray(v), causal=causal)
    if b == "torch":
        return _global_attention_torch(q, k, v, causal=causal)
    if b == "mlx":
        return _global_attention_mlx(q, k, v, causal=causal)
    raise ValueError(f"Unknown backend: {backend}")


def basin_local_attention(q, k, v, basin_ids, *, backend: str = "numpy", causal: bool = False):
    b = backend.lower()
    if b == "numpy":
        return _basin_local_attention_numpy(q, k, v, basin_ids, causal=causal)
    if b == "torch":
        return _basin_local_attention_torch(q, k, v, basin_ids, causal=causal)
    if b == "mlx":
        return _basin_local_attention_mlx(q, k, v, basin_ids, causal=causal)
    raise ValueError(f"Unknown backend: {backend}")
