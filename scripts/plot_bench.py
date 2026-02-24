from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot query time vs corpus size from parse_bench_logs CSV output")
    ap.add_argument("csv_path")
    ap.add_argument("--out", default="bench_query_times.png")
    args = ap.parse_args()

    rows = []
    with open(args.csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: int(r.get("sample_tokens") or 0))

    x = [int(r["sample_tokens"]) for r in rows]
    tau = [float(r.get("tau_query_ms") or 0.0) for r in rows]
    faiss = [float(r.get("faiss_query_ms") or 0.0) for r in rows]
    hnsw = [float(r.get("hnsw_query_ms") or 0.0) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, tau, marker='o', label='Tau')
    if any(faiss):
        plt.plot(x, faiss, marker='o', label='FAISS exact')
    if any(hnsw):
        plt.plot(x, hnsw, marker='o', label='HNSW')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Corpus size (tokens)')
    plt.ylabel('Batch query time (ms)')
    plt.title('Tau vs ANN baseline query time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
