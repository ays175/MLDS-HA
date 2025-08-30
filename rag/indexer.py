#!/usr/bin/env python3
"""
Simple RAG indexer for social analytics artifacts (TikTok now, IG/FB later).

Indexes the latest reports, insights, and metrics for a given platform and dataset_id
into a single Weaviate collection `SocialAnalyticsDoc` for agent retrieval.

Minimal, dependency-free (besides weaviate-client). Keep it simple.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

import weaviate
import weaviate.classes.config as wvcc


COLLECTION_NAME = "SocialAnalyticsDoc"


def ensure_schema(client) -> None:
    """Ensure the shared RAG collection exists."""
    try:
        client.collections.get(COLLECTION_NAME)
        return
    except Exception:
        pass

    client.collections.create(
        name=COLLECTION_NAME,
        description="Unified RAG docs for social analytics (TikTok/Instagram/Facebook)",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Source platform"),
            wvcc.Property(name="dataset_id", data_type=wvcc.DataType.TEXT, description="Dataset identifier"),
            wvcc.Property(name="doc_type", data_type=wvcc.DataType.TEXT, description="metrics|insight|summary|report|dashboard|guide"),
            wvcc.Property(name="title", data_type=wvcc.DataType.TEXT, description="Short document title"),
            wvcc.Property(name="text", data_type=wvcc.DataType.TEXT, description="Vectorized text content") ,
            wvcc.Property(name="file_path", data_type=wvcc.DataType.TEXT, description="Source file path"),
            wvcc.Property(name="tags", data_type=wvcc.DataType.TEXT_ARRAY, description="Labels/tags"),
            wvcc.Property(name="created_at", data_type=wvcc.DataType.TEXT, description="ISO timestamp"),
            wvcc.Property(name="is_latest", data_type=wvcc.DataType.BOOL, description="Marks latest snapshot for this artifact"),
            wvcc.Property(name="json_blob", data_type=wvcc.DataType.TEXT, description="Raw JSON if applicable"),
        ],
    )


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    if not text:
        return [""]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def load_text_from_file(path: Path, json_blob_limit: int) -> Dict[str, str]:
    """Return a dict with 'text' (vectorized) and optional 'json_blob' (raw JSON string)."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return {"text": text}
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            raw = json.dumps(data, ensure_ascii=False)
            # Truncate very large JSON to keep indexing fast
            if len(raw) > json_blob_limit:
                text = raw[:json_blob_limit]
                return {"text": text, "json_blob": text}
            return {"text": raw, "json_blob": raw}
        except Exception:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return {"text": raw, "json_blob": raw}
    # Fallback (rare): treat as plain text
    return {"text": path.read_text(encoding="utf-8", errors="ignore")}


def find_latest_metric_files(metrics_dir: Path, platform: str = "tiktok") -> List[Path]:
    """Pick the latest JSON for each known metric family; include latest summary.

    Adapts filename patterns by platform (tiktok vs facebook).
    """
    if not metrics_dir.exists():
        return []

    # Platform-specific filename prefixes
    if platform == "facebook":
        families = [
            "facebook_brand_performance_*.json",
            "facebook_content_type_performance_*.json",
            "facebook_temporal_analytics_*.json",
            # Facebook exporter currently does not emit duration_performance
            "facebook_top_performers_*.json",
            "facebook_worst_performers_*.json",
        ]
        summary_names = [
            "latest_metrics_summary_facebook.json",
            # fallback if ever produced without suffix
            "latest_metrics_summary.json",
        ]
    elif platform == "instagram":
        families = [
            "instagram_brand_performance_*.json",
            "instagram_content_type_performance_*.json",
            "instagram_temporal_analytics_*.json",
            # No duration_performance for IG currently
            "instagram_top_performers_*.json",
            "instagram_worst_performers_*.json",
        ]
        summary_names = [
            "latest_metrics_summary_instagram.json",
            "latest_metrics_summary.json",
        ]
    else:
        # Default to TikTok patterns
        families = [
            "tiktok_brand_performance_*.json",
            "tiktok_content_type_performance_*.json",
            "tiktok_temporal_analytics_*.json",
            "tiktok_duration_performance_*.json",
            "tiktok_top_performers_*.json",
            "tiktok_worst_performers_*.json",
            # Backward compat
            "brand_performance_*.json",
            "content_type_performance_*.json",
            "temporal_analytics_*.json",
            "duration_performance_*.json",
            "top_performers_*.json",
            "worst_performers_*.json",
        ]
        summary_names = ["latest_metrics_summary_tiktok.json", "latest_metrics_summary.json"]
    files: List[Path] = []

    # Include the latest summary if present (platform-specific first)
    for name in summary_names:
        summary = metrics_dir / name
        if summary.exists():
            files.append(summary)
            break

    for pattern in families:
        matches = sorted(metrics_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            files.append(matches[0])
    return files


def find_insight_files(insights_dir: Path, dataset_id: str) -> List[Path]:
    if not insights_dir.exists():
        return []
    safe_ds = ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(dataset_id or 'dataset'))

    patterns = [
        f"ai_insights_*_{safe_ds}_*.json",
        f"ai_insights_*_{safe_ds}_*.md",
        # Unprefixed executive artifacts
        f"executive_summary_{safe_ds}_*.json",
        f"executive_report_{safe_ds}_*.md",
        # Prefixed executive artifacts (TikTok & Facebook)
        f"tiktok_executive_summary_{safe_ds}_*.json",
        f"tiktok_executive_report_{safe_ds}_*.md",
        f"facebook_executive_summary_{safe_ds}_*.json",
        f"facebook_executive_report_{safe_ds}_*.md",
        # Instagram executive artifacts
        f"instagram_executive_summary_{safe_ds}_*.json",
        f"instagram_executive_report_{safe_ds}_*.md",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(insights_dir.glob(pattern))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def infer_doc_type(path: Path) -> str:
    name = path.name
    if name.startswith("ai_insights_"):
        return "insight"
    if name.startswith("executive_summary_") or name.startswith("tiktok_executive_summary_") or name.startswith("facebook_executive_summary_") or name.startswith("instagram_executive_summary_"):
        return "summary"
    if name.startswith("executive_report_") or name.startswith("tiktok_executive_report_") or name.startswith("facebook_executive_report_") or name.startswith("instagram_executive_report_"):
        return "report"
    if name in ("latest_metrics_summary.json", "latest_metrics_summary_facebook.json", "latest_metrics_summary_tiktok.json", "latest_metrics_summary_instagram.json"):
        return "metrics_summary"
    return "metrics" if name.endswith(".json") else "report"


def infer_title(path: Path) -> str:
    stem = path.stem
    return stem.replace("_", " ").title()


def upsert_docs(
    client,
    platform: str,
    dataset_id: str,
    files: List[Path],
    *,
    max_chars_per_file: int,
    max_chunks_per_file: int,
    chunk_size: int,
    batch_size: int,
) -> int:
    collection = client.collections.get(COLLECTION_NAME)
    added = 0
    now = datetime.now().isoformat()

    with tqdm(total=len(files), desc="Index files", unit="file") as pbar_files:
        for path in files:
            payload = load_text_from_file(path, json_blob_limit=max_chars_per_file)
            text = payload.get("text", "")
            json_blob = payload.get("json_blob")

            chunks = chunk_text(text, max_chars=chunk_size)
            if len(chunks) > max_chunks_per_file:
                chunks = chunks[:max_chunks_per_file]

            with collection.batch.fixed_size(batch_size=batch_size) as batch:
                for chunk in tqdm(chunks, desc=f"Index {path.name}", unit="chunk", leave=False):
                    obj = {
                        "platform": platform,
                        "dataset_id": dataset_id,
                        "doc_type": infer_doc_type(path),
                        "title": infer_title(path),
                        "text": chunk,
                        "file_path": str(path),
                        "tags": [],
                        "created_at": now,
                        "is_latest": True,
                    }
                    if json_blob is not None:
                        obj["json_blob"] = json_blob
                    batch.add_object(obj)
                    added += 1
            pbar_files.update(1)

    return added


def main():
    parser = argparse.ArgumentParser(description="Index social analytics artifacts into Weaviate RAG")
    parser.add_argument("--platform", default="tiktok", choices=["tiktok", "instagram", "facebook"], help="Source platform")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier to index")
    parser.add_argument("--metrics-dir", default="./metrics/tiktok", help="Metrics directory")
    parser.add_argument("--insights-dir", default="./insights/tiktok", help="Insights directory")
    parser.add_argument("--chunk-size", type=int, default=8000, help="Chars per chunk for vector text")
    parser.add_argument("--max-chunks-per-file", type=int, default=50, help="Max chunks per file")
    parser.add_argument("--max-chars-per-file", type=int, default=300_000, help="Max chars to read per JSON blob")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for Weaviate inserts")
    args = parser.parse_args()

    # Connect to local Weaviate (Docker-based)
    client = weaviate.connect_to_local()
    try:
        ensure_schema(client)

        metrics_dir = Path(args.metrics_dir)
        insights_dir = Path(args.insights_dir)

        # Validate dataset match from latest summary if present (platform-aware)
        platform_summary = {
            "tiktok": "latest_metrics_summary_tiktok.json",
            "facebook": "latest_metrics_summary_facebook.json",
            "instagram": "latest_metrics_summary_instagram.json",
        }.get(args.platform, "latest_metrics_summary.json")
        summary_candidates = [
            metrics_dir / platform_summary,
            metrics_dir / "latest_metrics_summary.json",
        ]
        for latest_summary in summary_candidates:
            if latest_summary.exists():
                try:
                    summary = json.loads(latest_summary.read_text(encoding="utf-8", errors="ignore"))
                    if summary.get("dataset_id") and summary.get("dataset_id") != args.dataset_id:
                        print(f"Warning: {latest_summary.name} dataset_id={summary.get('dataset_id')} != {args.dataset_id}")
                except Exception:
                    pass
                break

        files_to_index: List[Path] = []
        files_to_index.extend(find_latest_metric_files(metrics_dir, platform=args.platform))
        files_to_index.extend(find_insight_files(insights_dir, args.dataset_id))

        count = upsert_docs(
            client,
            args.platform,
            args.dataset_id,
            files_to_index,
            max_chars_per_file=args.max_chars_per_file,
            max_chunks_per_file=args.max_chunks_per_file,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
        print(f"Indexed {count} chunks from {len(files_to_index)} files for dataset_id={args.dataset_id} platform={args.platform}")
    finally:
        client.close()


if __name__ == "__main__":
    main()


