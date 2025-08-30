#!/usr/bin/env python3
"""
Create a single dashboard bundle JSON for a given platform/dataset_id by
merging the latest metrics and insights files. The frontend loads this bundle.

Usage:
  uv run python dashboard/bundle.py --platform tiktok --dataset-id <id> \
    --metrics-dir ./metrics/tiktok --insights-dir ./insights/tiktok \
    --out ./dashboard/dashboard_bundle_<id>.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def _latest_by_pattern(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _read_json(path: Optional[Path]) -> Dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _read_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _sanitize(obj):
    """Recursively replace NaN/Inf with None to produce strict JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def build_bundle(platform: str, dataset_id: str, metrics_dir: Path, insights_dir: Path) -> Dict:
    bundle: Dict = {"platform": platform, "dataset_id": dataset_id}

    # Summary + latest family files
    # Backward compat: prefer new tiktok summary, fallback to old name
    summary = _read_json(metrics_dir / "latest_metrics_summary_tiktok.json") or _read_json(metrics_dir / "latest_metrics_summary.json")
    bundle["latest_summary"] = summary
    # Include full dataset overview (contains key_metrics totals)
    bundle["dataset_overview"] = _read_json(_latest_by_pattern(metrics_dir, "tiktok_dataset_overview_*.json")) or _read_json(_latest_by_pattern(metrics_dir, "dataset_overview_*.json"))

    bundle["brand_performance"] = _read_json(_latest_by_pattern(metrics_dir, "tiktok_brand_performance_*.json")) or _read_json(_latest_by_pattern(metrics_dir, "brand_performance_*.json"))
    bundle["content_type_performance"] = _read_json(_latest_by_pattern(metrics_dir, "tiktok_content_type_performance_*.json")) or _read_json(_latest_by_pattern(metrics_dir, "content_type_performance_*.json"))
    bundle["temporal_analytics"] = _read_json(_latest_by_pattern(metrics_dir, "tiktok_temporal_analytics_*.json")) or _read_json(_latest_by_pattern(metrics_dir, "temporal_analytics_*.json"))
    bundle["duration_performance"] = _read_json(_latest_by_pattern(metrics_dir, "tiktok_duration_performance_*.json")) or _read_json(_latest_by_pattern(metrics_dir, "duration_performance_*.json"))
    bundle["top_performers"] = _read_json(_latest_by_pattern(metrics_dir, "tiktok_top_performers_*.json")) or _read_json(_latest_by_pattern(metrics_dir, "top_performers_*.json"))
    bundle["worst_performers"] = _read_json(_latest_by_pattern(metrics_dir, "tiktok_worst_performers_*.json")) or _read_json(_latest_by_pattern(metrics_dir, "worst_performers_*.json"))
    bundle["per_post_sample"] = _read_json(_latest_by_pattern(metrics_dir, "tiktok_per_post_sample_*.json")) or _read_json(_latest_by_pattern(metrics_dir, "per_post_sample_*.json"))

    # Executive dashboard + reports/insights (markdown rendered in UI)
    bundle["executive_dashboard"] = _read_json(insights_dir / "executive_dashboard.json")

    safe_ds = ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(dataset_id))
    # Prefer dataset-specific executive report; fallback to latest any if not found
    exec_md_path = _latest_by_pattern(insights_dir, f"executive_report_{safe_ds}_*.md")
    if not exec_md_path:
        exec_md_path = _latest_by_pattern(insights_dir, "executive_report_*.md")
    bundle["executive_report_md"] = _read_text(exec_md_path)

    # Optional: include latest insights markdown per focus (if present)
    focuses = ["comprehensive", "performance_optimization", "content_strategy", "posting_optimization"]
    insights_md: Dict[str, str] = {}
    for f in focuses:
        md = _read_text(_latest_by_pattern(insights_dir, f"ai_insights_{f}_{safe_ds}_*.md"))
        if md:
            insights_md[f] = md
    bundle["insights_md"] = insights_md

    return _sanitize(bundle)


def main():
    parser = argparse.ArgumentParser(description="Build dashboard bundle JSON for a dataset")
    parser.add_argument("--platform", default="tiktok", help="Platform name")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier")
    parser.add_argument("--metrics-dir", default="./metrics/tiktok", help="Metrics directory")
    parser.add_argument("--insights-dir", default="./insights/tiktok", help="Insights directory")
    parser.add_argument("--out", default=None, help="Output bundle path (defaults to ./dashboard/dashboard_bundle_<dataset>.json)")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    insights_dir = Path(args.insights_dir)
    out_path = Path(args.out) if args.out else Path(f"./dashboard/dashboard_bundle_{args.dataset_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle(args.platform, args.dataset_id, metrics_dir, insights_dir)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Also write/update a stable pointer for the dashboard to auto-load
    latest_path = out_path.parent / "dashboard_tiktok_bundle_latest.json"
    latest_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Wrote {latest_path}")

    # Also write a TikTok-specific alias for convenience
    tiktok_latest_path = out_path.parent / "dashboard_tiktok_bundle_latest.json"
    tiktok_latest_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Wrote {tiktok_latest_path}")


if __name__ == "__main__":
    main()


