from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import math
import numpy as np


# =========================
# Number-theoretic primitives
# =========================

def divisors(n: int) -> List[int]:
    if n <= 0:
        raise ValueError("k must be a positive integer")
    small: List[int] = []
    large: List[int] = []
    r = int(math.isqrt(n))
    for d in range(1, r + 1):
        if n % d == 0:
            small.append(d)
            q = n // d
            if q != d:
                large.append(q)
    return small + large[::-1]


def tau(n: int) -> int:
    return len(divisors(n))


def gcd3_scalar(a: int, b: int, c: int) -> int:
    return math.gcd(math.gcd(int(a), int(b)), int(c))


def gcd3_batch(x0: np.ndarray, x1: np.ndarray, k: int) -> np.ndarray:
    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have the same shape")
    x0_arr = np.asarray(x0)
    x1_arr = np.asarray(x1)
    if not np.issubdtype(x0_arr.dtype, np.integer) or not np.issubdtype(x1_arr.dtype, np.integer):
        raise TypeError("x0 and x1 must be integer arrays")
    k_arr = np.full_like(x0_arr, fill_value=int(k))
    return np.gcd(np.gcd(x0_arr, x1_arr), k_arr)


def cycle_for_divisor(k: int, g: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    if k % g != 0:
        raise ValueError(f"g={g} is not a divisor of k={k}")
    a = k + g
    b = 2 * k + g
    return ((a, a), (a, b), (b, a))


# =========================
# Tau Router
# =========================

@dataclass(frozen=True)
class TauConfig:
    k: int = 55_440
    chunk_size: int = 65_536
    dtype_token: str = "int32"
    dtype_state: str = "int64"
    dtype_pos: str = "int64"


class TauRouter:
    """Deterministic O(1)-style basin router via g = gcd(x0, x1, k)."""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = int(k)
        self._divisors = divisors(self.k)
        self._g_to_basin: Dict[int, int] = {g: i for i, g in enumerate(self._divisors)}
        self._basin_to_g = np.asarray(self._divisors, dtype=np.int64)
        self._cycle_lut = np.asarray(
            [[self.k + g, self.k + g, self.k + g, 2 * self.k + g, 2 * self.k + g, self.k + g] for g in self._divisors],
            dtype=np.int64,
        )

    @property
    def divisors(self) -> List[int]:
        return list(self._divisors)

    @property
    def num_basins(self) -> int:
        return len(self._divisors)

    def route_scalar(self, x0: int, x1: int) -> int:
        g = gcd3_scalar(x0, x1, self.k)
        return self._g_to_basin[g]

    def route_batch(self, x0: np.ndarray, x1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = gcd3_batch(x0, x1, self.k)
        basin_ids = np.searchsorted(self._basin_to_g, g)
        return basin_ids.astype(np.int32, copy=False), g.astype(np.int64, copy=False)

    def basin_of_g(self, g: int) -> int:
        return self._g_to_basin[int(g)]

    def g_of_basin(self, basin_id: int) -> int:
        return int(self._basin_to_g[int(basin_id)])

    def cycle_of_basin(self, basin_id: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        row = self._cycle_lut[int(basin_id)]
        return ((int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (int(row[4]), int(row[5])))

    def generated_weight_vector(self, basin_id: int, dim: int, dtype=np.float32) -> np.ndarray:
        if dim <= 0:
            raise ValueError("dim must be positive")
        row = self._cycle_lut[int(basin_id)].astype(np.float64)
        idx = np.arange(dim, dtype=np.float64)
        vec = (
            np.sin((idx + 1.0) * (row[0] % 997) * 0.001)
            + np.cos((idx + 1.0) * (row[3] % 991) * 0.0013)
            + ((row[5] % 31) / 31.0)
        )
        vec /= max(1e-6, np.linalg.norm(vec))
        return vec.astype(dtype, copy=False)


# =========================
# Basin-partitioned memory
# =========================

@dataclass
class BlockSummary:
    basin_id: int
    chunk_index: int
    start_pos: int
    end_pos: int
    count: int
    token_min: int
    token_max: int
    checksum: int


class _Chunk:
    __slots__ = ("capacity", "size", "positions", "token_ids", "x0", "x1", "embeddings", "d_model", "summaries")

    def __init__(self, capacity: int, d_model: int = 0):
        self.capacity = int(capacity)
        self.size = 0
        self.positions = np.empty(self.capacity, dtype=np.int64)
        self.token_ids = np.empty(self.capacity, dtype=np.int32)
        self.x0 = np.empty(self.capacity, dtype=np.int64)
        self.x1 = np.empty(self.capacity, dtype=np.int64)
        self.d_model = int(d_model)
        self.embeddings = np.empty((self.capacity, self.d_model), dtype=np.float32) if self.d_model > 0 else None
        self.summaries: List[BlockSummary] = []

    @property
    def free(self) -> int:
        return self.capacity - self.size

    def append(
        self,
        basin_id: int,
        chunk_index: int,
        positions: np.ndarray,
        token_ids: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> int:
        n = int(len(token_ids))
        if n == 0:
            return 0
        if n > self.free:
            raise ValueError("Chunk overflow")

        s0, s1 = self.size, self.size + n
        self.positions[s0:s1] = positions
        self.token_ids[s0:s1] = token_ids
        self.x0[s0:s1] = x0
        self.x1[s0:s1] = x1
        if self.embeddings is not None and embeddings is not None:
            self.embeddings[s0:s1, :] = embeddings

        pos_view = self.positions[s0:s1]
        tok_view = self.token_ids[s0:s1]
        checksum = int((np.int64(tok_view).sum() * 1315423911 + np.int64(pos_view).sum() * 2654435761) & 0xFFFFFFFF)
        self.summaries.append(
            BlockSummary(
                basin_id=int(basin_id),
                chunk_index=int(chunk_index),
                start_pos=int(pos_view[0]),
                end_pos=int(pos_view[-1]),
                count=n,
                token_min=int(tok_view.min()),
                token_max=int(tok_view.max()),
                checksum=checksum,
            )
        )
        self.size = s1
        return n

    def slice(self, start: int = 0, end: Optional[int] = None) -> Dict[str, np.ndarray]:
        e = self.size if end is None else min(int(end), self.size)
        s = max(0, int(start))
        out = {
            "positions": self.positions[s:e],
            "token_ids": self.token_ids[s:e],
            "x0": self.x0[s:e],
            "x1": self.x1[s:e],
        }
        if self.embeddings is not None:
            out["embeddings"] = self.embeddings[s:e]
        return out

    def bytes_used(self) -> int:
        total = int(self.positions.nbytes + self.token_ids.nbytes + self.x0.nbytes + self.x1.nbytes)
        if self.embeddings is not None:
            total += int(self.embeddings.nbytes)
        return total


class BasinPartition:
    def __init__(self, basin_id: int, g: int, chunk_size: int, d_model: int = 0):
        self.basin_id = int(basin_id)
        self.g = int(g)
        self.chunk_size = int(chunk_size)
        self.d_model = int(d_model)
        self.chunks: List[_Chunk] = []
        self.total_tokens = 0

    def _ensure_chunk(self) -> _Chunk:
        if not self.chunks or self.chunks[-1].free == 0:
            self.chunks.append(_Chunk(self.chunk_size, self.d_model))
        return self.chunks[-1]

    def append_batch(
        self,
        positions: np.ndarray,
        token_ids: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        n = int(len(token_ids))
        if n == 0:
            return
        cursor = 0
        while cursor < n:
            chunk = self._ensure_chunk()
            take = min(chunk.free, n - cursor)
            chunk_idx = len(self.chunks) - 1
            emb_slice = None if embeddings is None else embeddings[cursor : cursor + take]
            chunk.append(
                basin_id=self.basin_id,
                chunk_index=chunk_idx,
                positions=positions[cursor : cursor + take],
                token_ids=token_ids[cursor : cursor + take],
                x0=x0[cursor : cursor + take],
                x1=x1[cursor : cursor + take],
                embeddings=emb_slice,
            )
            cursor += take
            self.total_tokens += take

    def summaries(self) -> List[BlockSummary]:
        out: List[BlockSummary] = []
        for ch in self.chunks:
            out.extend(ch.summaries)
        return out

    def bytes_allocated(self) -> int:
        return sum(ch.bytes_used() for ch in self.chunks)

    def concat_arrays(self) -> Dict[str, np.ndarray]:
        if self.total_tokens == 0:
            out = {
                "positions": np.empty(0, dtype=np.int64),
                "token_ids": np.empty(0, dtype=np.int32),
                "x0": np.empty(0, dtype=np.int64),
                "x1": np.empty(0, dtype=np.int64),
            }
            if self.d_model > 0:
                out["embeddings"] = np.empty((0, self.d_model), dtype=np.float32)
            return out
        pos = np.concatenate([c.positions[: c.size] for c in self.chunks])
        tok = np.concatenate([c.token_ids[: c.size] for c in self.chunks])
        x0 = np.concatenate([c.x0[: c.size] for c in self.chunks])
        x1 = np.concatenate([c.x1[: c.size] for c in self.chunks])
        out = {"positions": pos, "token_ids": tok, "x0": x0, "x1": x1}
        if self.d_model > 0:
            out["embeddings"] = np.concatenate([c.embeddings[: c.size] for c in self.chunks], axis=0) if self.chunks else np.empty((0, self.d_model), dtype=np.float32)
        return out


class TauPartitionedMemory:
    def __init__(self, router: TauRouter, chunk_size: int = 65_536, d_model: int = 0):
        self.router = router
        self.chunk_size = int(chunk_size)
        self.d_model = int(d_model)
        self.partitions: List[BasinPartition] = [
            BasinPartition(i, router.g_of_basin(i), self.chunk_size, self.d_model) for i in range(router.num_basins)
        ]
        self.total_tokens = 0
        self._next_position = 0

    def append_batch(
        self,
        token_ids: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
        positions: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=np.int32)
        x0 = np.asarray(x0, dtype=np.int64)
        x1 = np.asarray(x1, dtype=np.int64)
        n = int(len(token_ids))
        if len(x0) != n or len(x1) != n:
            raise ValueError("token_ids, x0, x1 must have same length")

        if positions is None:
            positions = np.arange(self._next_position, self._next_position + n, dtype=np.int64)
            self._next_position += n
        else:
            positions = np.asarray(positions, dtype=np.int64)
            if len(positions) != n:
                raise ValueError("positions length mismatch")
            if n > 0:
                self._next_position = max(self._next_position, int(positions.max()) + 1)

        if self.d_model > 0:
            if embeddings is None:
                raise ValueError("embeddings required because d_model > 0")
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.shape != (n, self.d_model):
                raise ValueError(f"embeddings must have shape ({n}, {self.d_model})")

        basin_ids, _ = self.router.route_batch(x0, x1)
        if n == 0:
            return basin_ids

        order = np.argsort(basin_ids, kind="stable")
        b_sorted = basin_ids[order]
        tok_sorted = token_ids[order]
        x0_sorted = x0[order]
        x1_sorted = x1[order]
        pos_sorted = positions[order]
        emb_sorted = None if embeddings is None else embeddings[order]

        starts = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1]])
        ends = np.r_[starts[1:], np.array([n])]
        for s, e in zip(starts, ends):
            b = int(b_sorted[s])
            self.partitions[b].append_batch(
                positions=pos_sorted[s:e],
                token_ids=tok_sorted[s:e],
                x0=x0_sorted[s:e],
                x1=x1_sorted[s:e],
                embeddings=None if emb_sorted is None else emb_sorted[s:e],
            )
        self.total_tokens += n
        return basin_ids

    def _empty_retrieval(self, basin_id: int) -> Dict[str, np.ndarray]:
        out = {
            "positions": np.empty(0, dtype=np.int64),
            "token_ids": np.empty(0, dtype=np.int32),
            "x0": np.empty(0, dtype=np.int64),
            "x1": np.empty(0, dtype=np.int64),
            "basin_id": np.asarray([int(basin_id)], dtype=np.int32),
            "g": np.asarray([self.router.g_of_basin(int(basin_id))], dtype=np.int64),
        }
        if self.d_model > 0:
            out["embeddings"] = np.empty((0, self.d_model), dtype=np.float32)
        return out

    def retrieve_same_basin(
        self,
        query_x0: int,
        query_x1: int,
        *,
        max_tokens: int = 1024,
        window: Optional[Tuple[int, int]] = None,
        newest_first: bool = True,
    ) -> Dict[str, np.ndarray]:
        basin_id = self.router.route_scalar(int(query_x0), int(query_x1))
        return self.retrieve_by_basin(basin_id, max_tokens=max_tokens, window=window, newest_first=newest_first)

    def retrieve_by_basin(
        self,
        basin_id: int,
        *,
        max_tokens: int = 1024,
        window: Optional[Tuple[int, int]] = None,
        newest_first: bool = True,
    ) -> Dict[str, np.ndarray]:
        p = self.partitions[int(basin_id)]
        if max_tokens <= 0 or p.total_tokens == 0:
            return self._empty_retrieval(int(basin_id))

        start_w, end_w = (None, None)
        if window is not None:
            start_w, end_w = int(window[0]), int(window[1])

        pos_parts: List[np.ndarray] = []
        tok_parts: List[np.ndarray] = []
        x0_parts: List[np.ndarray] = []
        x1_parts: List[np.ndarray] = []
        emb_parts: List[np.ndarray] = []
        remaining = int(max_tokens)

        chunk_iter = reversed(p.chunks) if newest_first else iter(p.chunks)
        for ch in chunk_iter:
            if remaining <= 0 or ch.size == 0:
                if remaining <= 0:
                    break
                continue

            pos = ch.positions[: ch.size]
            if start_w is not None:
                if pos[-1] < start_w or pos[0] > end_w:
                    continue
                mask = (pos >= start_w) & (pos <= end_w)
                if not np.any(mask):
                    continue
                idx = np.flatnonzero(mask)
                if newest_first:
                    idx = idx[::-1]
                idx = idx[:remaining]
                pos_parts.append(pos[idx])
                tok_parts.append(ch.token_ids[idx])
                x0_parts.append(ch.x0[idx])
                x1_parts.append(ch.x1[idx])
                if ch.embeddings is not None:
                    emb_parts.append(ch.embeddings[idx])
                remaining -= len(idx)
            else:
                take = min(remaining, ch.size)
                if newest_first:
                    sl = slice(ch.size - take, ch.size)
                    pos_parts.append(ch.positions[sl][::-1])
                    tok_parts.append(ch.token_ids[sl][::-1])
                    x0_parts.append(ch.x0[sl][::-1])
                    x1_parts.append(ch.x1[sl][::-1])
                    if ch.embeddings is not None:
                        emb_parts.append(ch.embeddings[sl][::-1])
                else:
                    sl = slice(0, take)
                    pos_parts.append(ch.positions[sl])
                    tok_parts.append(ch.token_ids[sl])
                    x0_parts.append(ch.x0[sl])
                    x1_parts.append(ch.x1[sl])
                    if ch.embeddings is not None:
                        emb_parts.append(ch.embeddings[sl])
                remaining -= take

        if not pos_parts:
            return self._empty_retrieval(int(basin_id))

        out = {
            "positions": np.concatenate(pos_parts),
            "token_ids": np.concatenate(tok_parts),
            "x0": np.concatenate(x0_parts),
            "x1": np.concatenate(x1_parts),
            "basin_id": np.asarray([int(basin_id)], dtype=np.int32),
            "g": np.asarray([p.g], dtype=np.int64),
        }
        if self.d_model > 0:
            out["embeddings"] = np.concatenate(emb_parts, axis=0) if emb_parts else np.empty((0, self.d_model), dtype=np.float32)
        return out

    def iter_basin_blocks(self, basin_id: int) -> Iterable[Dict[str, np.ndarray]]:
        p = self.partitions[int(basin_id)]
        for ch in p.chunks:
            if ch.size:
                yield ch.slice(0, ch.size)

    def memory_stats(self) -> Dict[str, object]:
        counts = np.asarray([p.total_tokens for p in self.partitions], dtype=np.int64)
        bytes_alloc = int(sum(p.bytes_allocated() for p in self.partitions))
        nonempty = counts > 0
        return {
            "k": self.router.k,
            "tau_k": self.router.num_basins,
            "total_tokens": int(self.total_tokens),
            "nonempty_basins": int(nonempty.sum()),
            "max_basin_load": int(counts.max(initial=0)),
            "mean_basin_load": float(counts.mean()) if len(counts) else 0.0,
            "allocated_bytes": bytes_alloc,
            "allocated_gb": bytes_alloc / (1024 ** 3),
            "counts_per_basin": counts,
        }

    def top_loaded_basins(self, topk: int = 10) -> List[Tuple[int, int, int]]:
        topk = max(1, int(topk))
        items = [(p.basin_id, p.g, p.total_tokens) for p in self.partitions]
        items.sort(key=lambda t: t[2], reverse=True)
        return items[:topk]

    def to_flat_arrays(self, sort_by_position: bool = True) -> Dict[str, np.ndarray]:
        """Materialize a flat view for baseline comparisons / exports."""
        if self.total_tokens == 0:
            out = {
                "positions": np.empty(0, dtype=np.int64),
                "token_ids": np.empty(0, dtype=np.int32),
                "x0": np.empty(0, dtype=np.int64),
                "x1": np.empty(0, dtype=np.int64),
                "basin_ids": np.empty(0, dtype=np.int32),
            }
            if self.d_model > 0:
                out["embeddings"] = np.empty((0, self.d_model), dtype=np.float32)
            return out

        positions: List[np.ndarray] = []
        token_ids: List[np.ndarray] = []
        x0s: List[np.ndarray] = []
        x1s: List[np.ndarray] = []
        basin_ids: List[np.ndarray] = []
        embs: List[np.ndarray] = []
        for p in self.partitions:
            if p.total_tokens == 0:
                continue
            arrs = p.concat_arrays()
            n = len(arrs["token_ids"])
            positions.append(arrs["positions"])
            token_ids.append(arrs["token_ids"])
            x0s.append(arrs["x0"])
            x1s.append(arrs["x1"])
            basin_ids.append(np.full(n, p.basin_id, dtype=np.int32))
            if self.d_model > 0:
                embs.append(arrs["embeddings"])

        out = {
            "positions": np.concatenate(positions),
            "token_ids": np.concatenate(token_ids),
            "x0": np.concatenate(x0s),
            "x1": np.concatenate(x1s),
            "basin_ids": np.concatenate(basin_ids),
        }
        if self.d_model > 0:
            out["embeddings"] = np.concatenate(embs, axis=0)

        if sort_by_position and len(out["positions"]) > 1:
            order = np.argsort(out["positions"], kind="stable")
            for k in list(out.keys()):
                out[k] = out[k][order]
        return out

    def save(self, path: Union[str, Path], overwrite: bool = False) -> Path:
        """Persist to disk as .npy files; can be re-opened with memory mapping."""
        pth = Path(path)
        if pth.exists():
            if not overwrite:
                raise FileExistsError(f"{pth} already exists; pass overwrite=True")
        pth.mkdir(parents=True, exist_ok=True)

        meta = {
            "format": "tau_lattice_memmap_v1",
            "k": self.router.k,
            "tau_k": self.router.num_basins,
            "chunk_size": self.chunk_size,
            "d_model": self.d_model,
            "total_tokens": self.total_tokens,
            "basins": [],
        }
        for part in self.partitions:
            arrays = part.concat_arrays()
            basin_meta = {"basin_id": part.basin_id, "g": part.g, "count": int(part.total_tokens)}
            if part.total_tokens:
                stem = f"basin_{part.basin_id:03d}"
                np.save(pth / f"{stem}_positions.npy", arrays["positions"])
                np.save(pth / f"{stem}_token_ids.npy", arrays["token_ids"])
                np.save(pth / f"{stem}_x0.npy", arrays["x0"])
                np.save(pth / f"{stem}_x1.npy", arrays["x1"])
                if self.d_model > 0:
                    np.save(pth / f"{stem}_embeddings.npy", arrays["embeddings"])
                basin_meta["files"] = {
                    "positions": f"{stem}_positions.npy",
                    "token_ids": f"{stem}_token_ids.npy",
                    "x0": f"{stem}_x0.npy",
                    "x1": f"{stem}_x1.npy",
                }
                if self.d_model > 0:
                    basin_meta["files"]["embeddings"] = f"{stem}_embeddings.npy"
            else:
                basin_meta["files"] = {}
            meta["basins"].append(basin_meta)

        with open(pth / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return pth

    @classmethod
    def load_mmap(cls, path: Union[str, Path]) -> "TauMemmapMemory":
        return TauMemmapMemory.load(path)


class _MemmapPartition:
    def __init__(self, basin_id: int, g: int, arrays: Dict[str, np.ndarray], d_model: int, chunk_size: int):
        self.basin_id = int(basin_id)
        self.g = int(g)
        self.arrays = arrays
        self.d_model = int(d_model)
        self.chunk_size = int(chunk_size)
        self.total_tokens = int(len(arrays.get("token_ids", np.empty(0))))

    def bytes_allocated(self) -> int:
        total = 0
        for v in self.arrays.values():
            try:
                total += int(v.nbytes)
            except Exception:
                pass
        return total


class TauMemmapMemory:
    """Read-only memory-mapped view loaded from TauPartitionedMemory.save()."""

    def __init__(self, router: TauRouter, chunk_size: int, d_model: int, partitions: List[_MemmapPartition], total_tokens: int, root: Path):
        self.router = router
        self.chunk_size = int(chunk_size)
        self.d_model = int(d_model)
        self.partitions = partitions
        self.total_tokens = int(total_tokens)
        self.root = Path(root)

    @classmethod
    def load(cls, path: Union[str, Path], mmap_mode: str = "r") -> "TauMemmapMemory":
        root = Path(path)
        with open(root / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") != "tau_lattice_memmap_v1":
            raise ValueError("Unsupported memmap format")
        router = TauRouter(int(meta["k"]))
        d_model = int(meta.get("d_model", 0))
        chunk_size = int(meta.get("chunk_size", 65536))
        parts: List[_MemmapPartition] = []
        for b in meta["basins"]:
            arrays: Dict[str, np.ndarray] = {}
            for key, fname in b.get("files", {}).items():
                arrays[key] = np.load(root / fname, mmap_mode=mmap_mode)
            parts.append(_MemmapPartition(int(b["basin_id"]), int(b["g"]), arrays, d_model, chunk_size))
        return cls(router, chunk_size, d_model, parts, int(meta.get("total_tokens", 0)), root)

    def _empty_retrieval(self, basin_id: int) -> Dict[str, np.ndarray]:
        out = {
            "positions": np.empty(0, dtype=np.int64),
            "token_ids": np.empty(0, dtype=np.int32),
            "x0": np.empty(0, dtype=np.int64),
            "x1": np.empty(0, dtype=np.int64),
            "basin_id": np.asarray([int(basin_id)], dtype=np.int32),
            "g": np.asarray([self.router.g_of_basin(int(basin_id))], dtype=np.int64),
        }
        if self.d_model > 0:
            out["embeddings"] = np.empty((0, self.d_model), dtype=np.float32)
        return out

    def retrieve_same_basin(self, query_x0: int, query_x1: int, **kwargs) -> Dict[str, np.ndarray]:
        basin_id = self.router.route_scalar(int(query_x0), int(query_x1))
        return self.retrieve_by_basin(basin_id, **kwargs)

    def retrieve_by_basin(
        self,
        basin_id: int,
        *,
        max_tokens: int = 1024,
        window: Optional[Tuple[int, int]] = None,
        newest_first: bool = True,
    ) -> Dict[str, np.ndarray]:
        p = self.partitions[int(basin_id)]
        if p.total_tokens == 0 or max_tokens <= 0:
            return self._empty_retrieval(int(basin_id))
        pos = p.arrays["positions"]
        tok = p.arrays["token_ids"]
        x0 = p.arrays["x0"]
        x1 = p.arrays["x1"]
        emb = p.arrays.get("embeddings")

        if window is not None:
            start_w, end_w = int(window[0]), int(window[1])
            mask = (pos >= start_w) & (pos <= end_w)
            idx = np.flatnonzero(mask)
        else:
            idx = np.arange(p.total_tokens)

        if idx.size == 0:
            return self._empty_retrieval(int(basin_id))
        if newest_first:
            idx = idx[::-1]
        idx = idx[: int(max_tokens)]

        out = {
            "positions": np.asarray(pos[idx]),
            "token_ids": np.asarray(tok[idx]),
            "x0": np.asarray(x0[idx]),
            "x1": np.asarray(x1[idx]),
            "basin_id": np.asarray([int(basin_id)], dtype=np.int32),
            "g": np.asarray([p.g], dtype=np.int64),
        }
        if self.d_model > 0:
            out["embeddings"] = np.asarray(emb[idx]) if emb is not None else np.empty((0, self.d_model), dtype=np.float32)
        return out

    def iter_basin_blocks(self, basin_id: int) -> Iterable[Dict[str, np.ndarray]]:
        p = self.partitions[int(basin_id)]
        if p.total_tokens == 0:
            return
        n = p.total_tokens
        for s in range(0, n, self.chunk_size):
            e = min(n, s + self.chunk_size)
            out = {
                "positions": p.arrays["positions"][s:e],
                "token_ids": p.arrays["token_ids"][s:e],
                "x0": p.arrays["x0"][s:e],
                "x1": p.arrays["x1"][s:e],
            }
            if self.d_model > 0 and "embeddings" in p.arrays:
                out["embeddings"] = p.arrays["embeddings"][s:e]
            yield out

    def memory_stats(self) -> Dict[str, object]:
        counts = np.asarray([p.total_tokens for p in self.partitions], dtype=np.int64)
        bytes_alloc = int(sum(p.bytes_allocated() for p in self.partitions))
        return {
            "k": self.router.k,
            "tau_k": self.router.num_basins,
            "total_tokens": int(self.total_tokens),
            "nonempty_basins": int((counts > 0).sum()),
            "max_basin_load": int(counts.max(initial=0)),
            "mean_basin_load": float(counts.mean()) if len(counts) else 0.0,
            "allocated_bytes": bytes_alloc,
            "allocated_gb": bytes_alloc / (1024 ** 3),
            "counts_per_basin": counts,
            "memory_mapped": True,
        }

    def top_loaded_basins(self, topk: int = 10) -> List[Tuple[int, int, int]]:
        items = [(p.basin_id, p.g, p.total_tokens) for p in self.partitions]
        items.sort(key=lambda t: t[2], reverse=True)
        return items[: max(1, int(topk))]

    def to_flat_arrays(self, sort_by_position: bool = True) -> Dict[str, np.ndarray]:
        positions: List[np.ndarray] = []
        token_ids: List[np.ndarray] = []
        x0s: List[np.ndarray] = []
        x1s: List[np.ndarray] = []
        basin_ids: List[np.ndarray] = []
        embs: List[np.ndarray] = []
        for p in self.partitions:
            if p.total_tokens == 0:
                continue
            positions.append(np.asarray(p.arrays["positions"]))
            token_ids.append(np.asarray(p.arrays["token_ids"]))
            x0s.append(np.asarray(p.arrays["x0"]))
            x1s.append(np.asarray(p.arrays["x1"]))
            basin_ids.append(np.full(p.total_tokens, p.basin_id, dtype=np.int32))
            if self.d_model > 0 and "embeddings" in p.arrays:
                embs.append(np.asarray(p.arrays["embeddings"]))
        if not positions:
            out = {
                "positions": np.empty(0, dtype=np.int64),
                "token_ids": np.empty(0, dtype=np.int32),
                "x0": np.empty(0, dtype=np.int64),
                "x1": np.empty(0, dtype=np.int64),
                "basin_ids": np.empty(0, dtype=np.int32),
            }
            if self.d_model > 0:
                out["embeddings"] = np.empty((0, self.d_model), dtype=np.float32)
            return out
        out = {
            "positions": np.concatenate(positions),
            "token_ids": np.concatenate(token_ids),
            "x0": np.concatenate(x0s),
            "x1": np.concatenate(x1s),
            "basin_ids": np.concatenate(basin_ids),
        }
        if self.d_model > 0 and embs:
            out["embeddings"] = np.concatenate(embs, axis=0)
        if sort_by_position and len(out["positions"]) > 1:
            order = np.argsort(out["positions"], kind="stable")
            for k in list(out.keys()):
                out[k] = out[k][order]
        return out


# =========================
# Synthetic helpers
# =========================

def generate_synthetic_states(n: int, k: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    token_ids = rng.integers(0, 50_000, size=n, dtype=np.int32)
    x0 = rng.integers(1, 10_000_000, size=n, dtype=np.int64)
    x1 = rng.integers(1, 10_000_000, size=n, dtype=np.int64)

    divs = np.asarray(divisors(k), dtype=np.int64)
    pick = rng.integers(0, len(divs), size=n)
    g = divs[pick]
    x0 = (x0 // g) * g
    x1 = (x1 // g) * g
    x0[x0 == 0] = g[x0 == 0]
    x1[x1 == 0] = g[x1 == 0]
    return token_ids, x0, x1


def explicit_cycle_check(router: TauRouter, trials: int = 10) -> bool:
    rng = np.random.default_rng(123)
    for _ in range(trials):
        basin = int(rng.integers(0, router.num_basins))
        g = router.g_of_basin(basin)
        if router.cycle_of_basin(basin) != cycle_for_divisor(router.k, g):
            return False
    return True
