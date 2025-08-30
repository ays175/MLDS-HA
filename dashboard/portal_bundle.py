#!/usr/bin/env python3
"""
Create a single portal bundle JSON for cross-platform by merging the latest
metrics and insights files. Modeled after facebook_bundle.py.

Usage:
  uv run python dashboard/portal_bundle.py --dataset-id <id> \
    --metrics-dir ./metrics/cross_platform --insights-dir ./insights/cross_platform \
    --out ./dashboard/portal_bundle_<id>.json
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
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def build_bundle(dataset_id: str, metrics_dir: Path, insights_dir: Path) -> Dict:
    bundle: Dict = {"platform": "cross_platform", "dataset_id": dataset_id}

    # Summary + families (cross-platform naming)
    bundle["latest_summary"] = _read_json(metrics_dir / "latest_metrics_summary_cross_platform.json")
    bundle["dataset_overview"] = _read_json(_latest_by_pattern(metrics_dir, "cross_platform_dataset_overview_*.json"))
    bundle["brand_performance"] = _read_json(_latest_by_pattern(metrics_dir, "cross_platform_brand_performance_*.json"))
    bundle["content_type_performance"] = _read_json(_latest_by_pattern(metrics_dir, "cross_platform_content_type_performance_*.json"))
    bundle["temporal_analytics"] = _read_json(_latest_by_pattern(metrics_dir, "cross_platform_temporal_analytics_*.json"))
    bundle["top_performers"] = _read_json(_latest_by_pattern(metrics_dir, "cross_platform_top_performers_*.json"))
    bundle["worst_performers"] = _read_json(_latest_by_pattern(metrics_dir, "cross_platform_worst_performers_*.json"))
    bundle["per_post_sample"] = _read_json(_latest_by_pattern(metrics_dir, "cross_platform_per_post_sample_*.json"))

    # Executive dashboard + report
    bundle["executive_dashboard"] = _read_json(insights_dir / "cross_platform_executive_dashboard.json")
    safe_ds = ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(dataset_id))
    exec_md = _latest_by_pattern(insights_dir, f"cross_platform_executive_report_{safe_ds}_*.md") or _latest_by_pattern(insights_dir, "cross_platform_executive_report_*.md")
    bundle["executive_report_md"] = _read_text(exec_md)

    # Insights markdown per focus
    focuses = ["comprehensive", "performance_optimization", "content_strategy", "posting_optimization"]
    insights_md: Dict[str, str] = {}
    for f in focuses:
        md = _read_text(_latest_by_pattern(insights_dir, f"ai_insights_cross_platform_{f}_{safe_ds}_*.md")) or _read_text(_latest_by_pattern(insights_dir, f"ai_insights_cross_platform_{f}_*.md"))
        if md:
            insights_md[f] = md
    bundle["insights_md"] = insights_md

    return _sanitize(bundle)


def main():
    parser = argparse.ArgumentParser(description="Build portal dashboard bundle JSON")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--metrics-dir", default="./metrics/cross_platform")
    parser.add_argument("--insights-dir", default="./insights/cross_platform")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    insights_dir = Path(args.insights_dir)
    out_path = Path(args.out) if args.out else Path(f"./dashboard/dashboard_portal_bundle_{args.dataset_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle(args.dataset_id, metrics_dir, insights_dir)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Wrote {out_path}")

    latest_path = out_path.parent / "dashboard_portal_bundle_latest.json"
    latest_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Wrote {latest_path}")


if __name__ == "__main__":
    main()


