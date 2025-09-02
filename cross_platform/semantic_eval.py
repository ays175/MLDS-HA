#!/usr/bin/env python3
"""
Semantic search evaluation harness (offline):
- Computes Recall@k, MRR, and nDCG for curated query→relevant doc sets
- Accepts ranked results per query (list of doc ids in order)
- Saves summary JSON under metrics/global/

Usage (example):
  from cross_platform.semantic_eval import evaluate_semantic_search
  results = evaluate_semantic_search(queries, ground_truth, ranked_results)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime


def recall_at_k(relevant: List[str], ranked: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(ranked[:k])
    hit = len(set(relevant) & topk)
    return float(hit / len(set(relevant)))


def mrr(relevant: List[str], ranked: List[str]) -> float:
    if not relevant:
        return 0.0
    relset = set(relevant)
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relset:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: List[int], k: int) -> float:
    import math
    r = relevances[:k]
    if not r:
        return 0.0
    return r[0] + sum(rel / math.log2(i + 1) for i, rel in enumerate(r[1:], start=2))


def ndcg_at_k(relevant: List[str], ranked: List[str], k: int) -> float:
    # Binary relevance
    relset = set(relevant)
    relevances = [1 if d in relset else 0 for d in ranked]
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return float(dcg / ideal)


def evaluate_semantic_search(
    queries: List[str],
    ground_truth: Dict[str, List[str]],
    ranked_results: Dict[str, List[str]],
    ks: List[int] | None = None,
) -> Dict[str, Any]:
    ks = ks or [5, 10, 20]
    out: Dict[str, Any] = {"per_query": {}, "summary": {}}
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    ndcgs: Dict[int, List[float]] = {k: [] for k in ks}
    mrrs: List[float] = []

    for q in queries:
        rel = ground_truth.get(q, [])
        ranked = ranked_results.get(q, [])
        q_metrics = {}
        for k in ks:
            q_metrics[f"recall@{k}"] = recall_at_k(rel, ranked, k)
            q_metrics[f"ndcg@{k}"] = ndcg_at_k(rel, ranked, k)
            recalls[k].append(q_metrics[f"recall@{k}"])
            ndcgs[k].append(q_metrics[f"ndcg@{k}"])
        q_mrr = mrr(rel, ranked)
        mrrs.append(q_mrr)
        q_metrics["mrr"] = q_mrr
        out["per_query"][q] = q_metrics

    summary = {f"recall@{k}": float(sum(recalls[k]) / max(1, len(recalls[k]))) for k in ks}
    summary.update({f"ndcg@{k}": float(sum(ndcgs[k]) / max(1, len(ndcgs[k]))) for k in ks})
    summary["mrr"] = float(sum(mrrs) / max(1, len(mrrs)))
    out["summary"] = summary
    out["generated_at"] = datetime.now().isoformat()
    return out


def save_eval_results(results: Dict[str, Any], project_root: Path | None = None) -> Path:
    project_root = project_root or Path(__file__).resolve().parents[1]
    out_dir = project_root / "metrics" / "global"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"semantic_eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return out_path


if __name__ == "__main__":
    # Minimal CLI example scaffold
    import argparse
    parser = argparse.ArgumentParser(description="Semantic eval harness")
    parser.add_argument("--input", help="Path to JSON with queries, ground_truth, ranked_results", required=False)
    args = parser.parse_args()
    # Placeholder if no input is provided
    if not args.input:
        demo_queries = ["demo_query"]
        demo_gt = {"demo_query": ["doc1", "doc3"]}
        demo_ranked = {"demo_query": ["doc3", "doc2", "doc1"]}
        res = evaluate_semantic_search(demo_queries, demo_gt, demo_ranked)
        p = save_eval_results(res)
        print(f"Saved: {p}")
    else:
        with open(args.input, "r") as f:
            payload = json.load(f)
        res = evaluate_semantic_search(payload.get("queries", []), payload.get("ground_truth", {}), payload.get("ranked_results", {}))
        p = save_eval_results(res)
        print(f"Saved: {p}")


