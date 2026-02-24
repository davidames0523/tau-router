from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import time
import numpy as np

from .core import TauRouter, generate_synthetic_states


def _softmax_cross_entropy_numpy(logits: np.ndarray, targets: np.ndarray):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), targets] + 1e-12).mean()
    grad = probs
    grad[np.arange(n), targets] -= 1.0
    grad /= n
    return float(loss), grad.astype(np.float32, copy=False)


class TauNanoNumpy:
    def __init__(self, vocab_size: int, num_basins: int, d_model: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 0.02
        self.vocab_size = int(vocab_size)
        self.num_basins = int(num_basins)
        self.d_model = int(d_model)
        self.token_emb = rng.normal(0, scale, size=(vocab_size, d_model)).astype(np.float32)
        self.basin_emb = rng.normal(0, scale, size=(num_basins, d_model)).astype(np.float32)
        self.W = rng.normal(0, scale, size=(d_model, vocab_size)).astype(np.float32)
        self.b = np.zeros((vocab_size,), dtype=np.float32)

    def step(self, token_ids: np.ndarray, basin_ids: np.ndarray, targets: np.ndarray, lr: float = 1e-1) -> float:
        token_ids = token_ids.astype(np.int64, copy=False)
        basin_ids = basin_ids.astype(np.int64, copy=False)
        targets = targets.astype(np.int64, copy=False)
        h = self.token_emb[token_ids] + self.basin_emb[basin_ids]
        logits = h @ self.W + self.b
        loss, dlogits = _softmax_cross_entropy_numpy(logits, targets)

        dW = h.T @ dlogits
        db = dlogits.sum(axis=0)
        dh = dlogits @ self.W.T
        d_token_emb = np.zeros_like(self.token_emb)
        d_basin_emb = np.zeros_like(self.basin_emb)
        np.add.at(d_token_emb, token_ids, dh)
        np.add.at(d_basin_emb, basin_ids, dh)

        self.W -= lr * dW
        self.b -= lr * db
        self.token_emb -= lr * d_token_emb
        self.basin_emb -= lr * d_basin_emb
        return float(loss)


class TauNanoTorch:
    def __init__(self, vocab_size: int, num_basins: int, d_model: int = 64, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(int(seed))
        self.torch = torch
        self.vocab_size = int(vocab_size)
        self.num_basins = int(num_basins)
        self.d_model = int(d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.basin_emb = nn.Embedding(num_basins, d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.opt = torch.optim.SGD(list(self.token_emb.parameters()) + list(self.basin_emb.parameters()) + list(self.proj.parameters()), lr=1.0)
        # Match numpy init scale roughly.
        with torch.no_grad():
            for p in list(self.token_emb.parameters()) + list(self.basin_emb.parameters()) + list(self.proj.parameters()):
                p.mul_(0.02)

    def step(self, token_ids: np.ndarray, basin_ids: np.ndarray, targets: np.ndarray, lr: float = 1.0) -> float:
        torch = self.torch
        for g in self.opt.param_groups:
            g["lr"] = float(lr)
        tok = torch.as_tensor(token_ids, dtype=torch.long)
        bas = torch.as_tensor(basin_ids, dtype=torch.long)
        tgt = torch.as_tensor(targets, dtype=torch.long)

        self.opt.zero_grad(set_to_none=True)
        h = self.token_emb(tok) + self.basin_emb(bas)
        logits = self.proj(h)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        self.opt.step()
        return float(loss.detach().cpu().item())


class TauNanoMLX:
    """
    Optional MLX backend (Apple Silicon). Uses MLX for forward matmuls/softmax and NumPy for sparse scatter updates.
    This keeps the implementation simple and dependency-light while providing a working MLX path.
    """

    def __init__(self, vocab_size: int, num_basins: int, d_model: int = 64, seed: int = 0):
        import mlx.core as mx

        self.mx = mx
        self.rng = np.random.default_rng(seed)
        scale = 0.02
        self.vocab_size = int(vocab_size)
        self.num_basins = int(num_basins)
        self.d_model = int(d_model)
        self.token_emb = mx.array(self.rng.normal(0, scale, size=(vocab_size, d_model)).astype(np.float32))
        self.basin_emb = mx.array(self.rng.normal(0, scale, size=(num_basins, d_model)).astype(np.float32))
        self.W = mx.array(self.rng.normal(0, scale, size=(d_model, vocab_size)).astype(np.float32))
        self.b = mx.array(np.zeros((vocab_size,), dtype=np.float32))

    def step(self, token_ids: np.ndarray, basin_ids: np.ndarray, targets: np.ndarray, lr: float = 1.0) -> float:
        mx = self.mx
        token_ids = np.asarray(token_ids, dtype=np.int64)
        basin_ids = np.asarray(basin_ids, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)

        tok_m = mx.array(token_ids)
        bas_m = mx.array(basin_ids)
        h = self.token_emb[tok_m] + self.basin_emb[bas_m]
        logits = mx.matmul(h, self.W) + self.b
        mx.eval(logits)
        logits_np = np.array(logits)
        loss, dlogits = _softmax_cross_entropy_numpy(logits_np, targets)

        h_np = np.array(h)
        dW = h_np.T @ dlogits
        db = dlogits.sum(axis=0)
        dh = dlogits @ np.array(self.W).T

        d_token_emb = np.zeros((self.vocab_size, self.d_model), dtype=np.float32)
        d_basin_emb = np.zeros((self.num_basins, self.d_model), dtype=np.float32)
        np.add.at(d_token_emb, token_ids, dh)
        np.add.at(d_basin_emb, basin_ids, dh)

        self.W = mx.array(np.array(self.W) - float(lr) * dW.astype(np.float32))
        self.b = mx.array(np.array(self.b) - float(lr) * db.astype(np.float32))
        self.token_emb = mx.array(np.array(self.token_emb) - float(lr) * d_token_emb)
        self.basin_emb = mx.array(np.array(self.basin_emb) - float(lr) * d_basin_emb)
        return float(loss)


@dataclass
class TrainResult:
    losses: List[float]
    elapsed_s: float
    backend: str


def make_batch(router: TauRouter, batch_size: int, vocab_size: int, seed: int):
    token_ids, x0, x1 = generate_synthetic_states(batch_size, router.k, seed=seed)
    basin_ids, _ = router.route_batch(x0, x1)
    token_ids = token_ids % vocab_size
    targets = (basin_ids.astype(np.int64) * 7 + 3) % vocab_size
    return token_ids.astype(np.int32), basin_ids.astype(np.int32), targets.astype(np.int32)


def run_train(
    k: int = 55_440,
    steps: int = 120,
    batch_size: int = 256,
    vocab_size: int = 64,
    d_model: int = 16,
    lr: float = 2.0,
    seed: int = 0,
    backend: str = "numpy",
    quiet: bool = False,
) -> TrainResult:
    router = TauRouter(k)
    b = backend.lower()
    if b == "numpy":
        model = TauNanoNumpy(vocab_size=vocab_size, num_basins=router.num_basins, d_model=d_model, seed=seed)
    elif b == "torch":
        model = TauNanoTorch(vocab_size=vocab_size, num_basins=router.num_basins, d_model=d_model, seed=seed)
    elif b == "mlx":
        try:
            model = TauNanoMLX(vocab_size=vocab_size, num_basins=router.num_basins, d_model=d_model, seed=seed)
        except Exception as e:
            raise RuntimeError("MLX backend requested but mlx is not installed/available.") from e
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if not quiet:
        print(f"Tau-Nano ({b} demo)")
        print(f"k={k} tau(k)={router.num_basins} basins | vocab={vocab_size} d_model={d_model}")

    losses: List[float] = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        tok, basin, tgt = make_batch(router, batch_size, vocab_size, seed=seed + step)
        loss = model.step(tok, basin, tgt, lr=lr)
        losses.append(float(loss))
        if not quiet and (step == 1 or step % 25 == 0):
            window = losses[-25:] if len(losses) >= 25 else losses
            print(f"step {step:4d} | loss={loss:.4f} | avg(last {len(window)}): {np.mean(window):.4f}")

    elapsed = time.perf_counter() - t0
    if not quiet:
        print(f"\nTrain time: {elapsed:.2f}s")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss:   {losses[-1]:.4f}")
        trend = "decreasing ✅" if losses[-1] < losses[0] else "not decreasing ⚠️"
        print(f"Loss trend: {trend}")
    return TrainResult(losses=losses, elapsed_s=elapsed, backend=b)
