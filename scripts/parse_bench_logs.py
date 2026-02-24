from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

PATTERNS = {
    "sample_tokens": re.compile(r"Sample size:\s*([\d,]+)\s*tokens"),
    "tau_candidates": re.compile(r"Tau candidate set size .*?:\s*([0-9.]+)\s*/\s*([0-9]+)\s*/\s*([0-9]+)"),
    "candidate_reduction": re.compile(r"Candidate reduction vs global scan:\s*([0-9.]+)x"),
    "tau_query_ms": re.compile(r"^Tau prefilter .*?\n\s+Query time \(batch of [0-9]+\):\s*([0-9.]+) ms", re.M | re.S),
    "tau_avg_ms": re.compile(r"^Tau prefilter .*?\n\s+Avg/query:\s*([0-9.]+) ms", re.M | re.S),
    "tau_speedup_exact": re.compile(r"^Tau prefilter .*?\n\s+Speedup vs exact brute-force:\s*([0-9.]+)x", re.M | re.S),
    "tau_recall": re.compile(r"^Tau prefilter .*?\n\s+Recall@([0-9]+):\s*([0-9.]+) .*? exact-order match rate: ([0-9.]+)", re.M | re.S),
    "exact_batch_ms": re.compile(r"^Exact brute-force .*?\n\s+Query time \(batch GEMM.*?\):\s*([0-9.]+) ms", re.M | re.S),
    "faiss_query_ms": re.compile(r"^FAISS \(IndexFlatIP, exact\):\n(?:.*\n)*?\s+Query time \(batch of [0-9]+\):\s*([0-9.]+) ms", re.M),
    "faiss_build_ms": re.compile(r"^FAISS \(IndexFlatIP, exact\):\n(?:.*\n)*?\s+Build time:\s*([0-9.]+) ms", re.M),
    "faiss_recall": re.compile(r"^FAISS \(IndexFlatIP, exact\):\n(?:.*\n)*?\s+Recall@([0-9]+):\s*([0-9.]+) .*? exact-order match rate: ([0-9.]+)", re.M),
    "hnsw_query_ms": re.compile(r"^HNSW \(hnswlib, ANN cosine\):\n(?:.*\n)*?\s+Query time \(batch of [0-9]+\):\s*([0-9.]+) ms", re.M),
    "hnsw_build_ms": re.compile(r"^HNSW \(hnswlib, ANN cosine\):\n(?:.*\n)*?\s+Build time:\s*([0-9.]+) ms", re.M),
    "hnsw_recall": re.compile(r"^HNSW \(hnswlib, ANN cosine\):\n(?:.*\n)*?\s+Recall@([0-9]+):\s*([0-9.]+) .*? exact-order match rate: ([0-9.]+)", re.M),
    "peak_mem_bytes": re.compile(r"^\s*([0-9]+)\s+(?:peak memory footprint|maximum resident set size)\s*$", re.M),
}



def parse_one(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    row: Dict[str, object] = {"file": str(path)}
    for key, pat in PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        if key in {"tau_candidates", "tau_recall", "faiss_recall", "hnsw_recall"}:
            row[key] = m.groups()
        else:
            row[key] = m.group(1)

    # Normalize fields
    def _f(name: str):
        return float(str(row[name])) if name in row else None

    def _i(name: str):
        return int(str(row[name]).replace(',', '')) if name in row else None

    out = {
        "file": str(path),
        "sample_tokens": _i("sample_tokens"),
        "candidate_reduction_x": _f("candidate_reduction"),
        "tau_query_ms": _f("tau_query_ms"),
        "tau_avg_ms": _f("tau_avg_ms"),
        "tau_speedup_vs_exact_x": _f("tau_speedup_exact"),
        "exact_batch_ms": _f("exact_batch_ms"),
        "faiss_build_ms": _f("faiss_build_ms"),
        "faiss_query_ms": _f("faiss_query_ms"),
        "hnsw_build_ms": _f("hnsw_build_ms"),
        "hnsw_query_ms": _f("hnsw_query_ms"),
        "peak_mem_bytes": _i("peak_mem_bytes"),
    }

    tc = row.get("tau_candidates")
    if tc:
        mean_c, min_c, max_c = tc
        out["tau_candidates_mean"] = float(mean_c)
        out["tau_candidates_min"] = int(min_c)
        out["tau_candidates_max"] = int(max_c)

    for prefix in ["tau", "faiss", "hnsw"]:
        rv = row.get(f"{prefix}_recall")
        if rv:
            k, recall, exact_order = rv
            out["top_k"] = int(k)
            out[f"{prefix}_recall"] = float(recall)
            out[f"{prefix}_exact_order_match"] = float(exact_order)

    # Derived convenience metrics
    if out.get("faiss_query_ms") and out.get("tau_query_ms"):
        out["tau_vs_faiss_query_speedup_x"] = float(out["faiss_query_ms"]) / max(float(out["tau_query_ms"]), 1e-12)
    if out.get("hnsw_query_ms") and out.get("tau_query_ms"):
        out["tau_vs_hnsw_query_speedup_x"] = float(out["hnsw_query_ms"]) / max(float(out["tau_query_ms"]), 1e-12)
    if out.get("peak_mem_bytes"):
        out["peak_mem_gb"] = float(out["peak_mem_bytes"]) / (1024 ** 3)

    return out


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r.get("sample_tokens") or 0))
    headers = [
        "sample_tokens", "tau_query_ms", "faiss_query_ms", "hnsw_query_ms",
        "tau_speedup_vs_exact_x", "candidate_reduction_x", "tau_recall", "faiss_recall", "hnsw_recall",
        "peak_mem_gb"
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                if abs(v) >= 1000:
                    vals.append(f"{v:.1f}")
                else:
                    vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse tau benchmark logs into CSV/Markdown summaries")
    ap.add_argument("logs", nargs="+", help="Text logs produced by tau-infinite")
    ap.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV output path")
    ap.add_argument("--markdown", dest="md_path", default=None, help="Optional markdown table output path")
    args = ap.parse_args()

    rows = [parse_one(Path(p)) for p in args.logs]
    rows.sort(key=lambda r: (r.get("sample_tokens") or 0, r["file"]))

    if args.csv_path:
        write_csv(rows, Path(args.csv_path))
    if args.md_path:
        write_markdown(rows, Path(args.md_path))

    # print a tiny summary to stdout
    for r in rows:
        n = r.get("sample_tokens")
        tq = r.get("tau_query_ms")
        fq = r.get("faiss_query_ms")
        hq = r.get("hnsw_query_ms")
        print(f"N={n}: Tau={tq} ms | FAISS={fq} ms | HNSW={hq} ms")


if __name__ == "__main__":
    main()
