#!/usr/bin/env python3
"""
Cross-Platform Analytics Agent

Consumes cross-platform metrics in ./metrics/cross_platform and produces
AI insights for four focuses plus the executive trio. Outputs are saved to
./insights/cross_platform with 'cross_platform_' prefixes.
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm


class CrossPlatformAnalyticsAgent:
    def __init__(self, metrics_dir: str = "./metrics/cross_platform", insights_dir: str = "./insights/cross_platform", ollama_host: str = "http://localhost:11434"):
        self.metrics_dir = Path(metrics_dir)
        self.insights_dir = Path(insights_dir)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_host = ollama_host
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --------------- Public API ---------------
    async def analyze(self, focus: str = "comprehensive", skip_if_exists: bool = False) -> Dict[str, Any]:
        metrics = self._load_all_metrics()
        if not metrics:
            raise FileNotFoundError("No cross-platform metrics found. Run cross_platform_metrics_export.py first.")

        dataset_id = metrics.get("summary", {}).get("dataset_id")
        if skip_if_exists and dataset_id and self._existing_analysis_file(focus, dataset_id):
            return {
                "skipped": True,
                "reason": "analysis_already_exists",
                "focus": focus,
                "dataset_id": dataset_id,
                "generated_at": datetime.now().isoformat(),
            }

        if focus == "comprehensive":
            return await self._comprehensive(metrics)
        if focus == "performance_optimization":
            return await self._performance_optimization(metrics)
        if focus == "content_strategy":
            return await self._content_strategy(metrics)
        if focus == "posting_optimization":
            return await self._posting_optimization(metrics)
        return await self._comprehensive(metrics)

    async def generate_executive(self, skip_if_exists: bool = False) -> Dict[str, Any]:
        metrics = self._load_all_metrics()
        summary = metrics.get("summary", {})
        dataset_id = summary.get("dataset_id")
        if skip_if_exists and dataset_id and self._existing_executive_file(dataset_id):
            return {"skipped": True, "reason": "executive_exists", "dataset_id": dataset_id}

        prompt = f"""
CROSS-PLATFORM EXECUTIVE SUMMARY (Sephora Portfolio)

Key Metrics:
- Total Posts: {summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0):,}
- Avg Engagement Rate: {summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0):.2f}%
- Viral Threshold: {summary.get('quick_access', {}).get('dataset_overview', {}).get('viral_threshold', 0):,.0f}

Provide a 3-paragraph executive assessment:
1) Overall portfolio performance across TikTok, Facebook, Instagram
2) Opportunities/risks, with platform-specific highlights
3) Strategic recommendations with business impact
"""

        ai_response = await self._ollama(prompt)
        result = {
            "executive_summary": ai_response,
            "kpi_snapshot": {
                "total_posts": summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0),
                "avg_engagement_rate": summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0),
                "viral_threshold": summary.get('quick_access', {}).get('dataset_overview', {}).get('viral_threshold', 0),
            },
            "dataset_id": dataset_id,
            "generated_at": datetime.now().isoformat(),
        }
        self._save_executive_outputs(result)
        return result

    # --------------- Metrics Loading ---------------
    def _load_all_metrics(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        # Summary
        summary_file = self.metrics_dir / "latest_metrics_summary_cross_platform.json"
        if summary_file.exists():
            data["summary"] = json.loads(summary_file.read_text(encoding="utf-8", errors="ignore"))

        # Families
        families = [
            ("brand_performance", "cross_platform_brand_performance_*.json"),
            ("content_type_performance", "cross_platform_content_type_performance_*.json"),
            ("temporal_analytics", "cross_platform_temporal_analytics_*.json"),
            ("top_performers", "cross_platform_top_performers_*.json"),
            ("worst_performers", "cross_platform_worst_performers_*.json"),
        ]
        for key, pattern in families:
            matches = sorted(self.metrics_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                try:
                    data[key] = json.loads(matches[0].read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
        return data

    # --------------- Existence checks ---------------
    def _safe_ds(self, dataset_id: Optional[str]) -> str:
        value = dataset_id or "dataset"
        return ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(value))

    def _existing_analysis_file(self, focus: str, dataset_id: str) -> Optional[Path]:
        pattern = f"ai_insights_cross_platform_{focus}_{self._safe_ds(dataset_id)}_*.json"
        matches = list(self.insights_dir.glob(pattern))
        return max(matches, key=lambda p: p.stat().st_mtime) if matches else None

    def _existing_executive_file(self, dataset_id: str) -> Optional[Path]:
        pattern = f"cross_platform_executive_summary_{self._safe_ds(dataset_id)}_*.json"
        matches = list(self.insights_dir.glob(pattern))
        return max(matches, key=lambda p: p.stat().st_mtime) if matches else None

    # --------------- Focused analyses ---------------
    async def _comprehensive(self, m: Dict[str, Any]) -> Dict[str, Any]:
        s = m.get("summary", {})
        prompt = f"""
CROSS-PLATFORM COMPREHENSIVE ANALYSIS

Dataset Overview:
- Total Posts: {s.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0):,}
- Avg Engagement Rate: {s.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0):.2f}%
- Viral Threshold: {s.get('quick_access', {}).get('dataset_overview', {}).get('viral_threshold', 0):,.0f}

Provide portfolio-level insights across TikTok, Facebook, Instagram:
- What brands/content types overperform consistently across platforms?
- Cross-channel posting time synergies or conflicts
- Risks and opportunities by platform
"""
        resp = await self._ollama(prompt)
        result = {
            "analysis_type": "comprehensive",
            "ai_insights": resp,
            "dataset_id": s.get("dataset_id"),
            "generated_at": datetime.now().isoformat(),
        }
        self._save_analysis_outputs(result, "comprehensive")
        return result

    async def _performance_optimization(self, m: Dict[str, Any]) -> Dict[str, Any]:
        s = m.get("summary", {})
        prompt = f"""
CROSS-PLATFORM PERFORMANCE OPTIMIZATION

Focus: underperformers and lift tactics across platforms.
List specific actions per platform (TikTok/Facebook/Instagram) and expected impact.
"""
        resp = await self._ollama(prompt)
        result = {
            "analysis_type": "performance_optimization",
            "ai_insights": resp,
            "dataset_id": s.get("dataset_id"),
            "generated_at": datetime.now().isoformat(),
        }
        self._save_analysis_outputs(result, "performance_optimization")
        return result

    async def _content_strategy(self, m: Dict[str, Any]) -> Dict[str, Any]:
        s = m.get("summary", {})
        prompt = f"""
CROSS-PLATFORM CONTENT STRATEGY

Identify high-performing content patterns and brand collaborations across platforms.
Provide prioritized recommendations for Sephora portfolio content.
"""
        resp = await self._ollama(prompt)
        result = {
            "analysis_type": "content_strategy",
            "ai_insights": resp,
            "dataset_id": s.get("dataset_id"),
            "generated_at": datetime.now().isoformat(),
        }
        self._save_analysis_outputs(result, "content_strategy")
        return result

    async def _posting_optimization(self, m: Dict[str, Any]) -> Dict[str, Any]:
        s = m.get("summary", {})
        prompt = f"""
CROSS-PLATFORM POSTING OPTIMIZATION

Propose an integrated posting schedule (hours/days) balancing all platforms.
"""
        resp = await self._ollama(prompt)
        result = {
            "analysis_type": "posting_optimization",
            "ai_insights": resp,
            "dataset_id": s.get("dataset_id"),
            "generated_at": datetime.now().isoformat(),
        }
        self._save_analysis_outputs(result, "posting_optimization")
        return result

    # --------------- Persistence ---------------
    def _save_analysis_outputs(self, result: Dict, focus: str) -> None:
        ts = self.timestamp
        safe_ds = self._safe_ds(result.get("dataset_id"))
        json_path = self.insights_dir / f"ai_insights_cross_platform_{focus}_{safe_ds}_{ts}.json"
        json_path.write_text(json.dumps(result, indent=2, default=str))

        md_path = self.insights_dir / f"ai_insights_cross_platform_{focus}_{safe_ds}_{ts}.md"
        md = [
            f"# Cross-Platform AI Insights - {focus.replace('_',' ').title()}",
            f"\n**Generated:** {result.get('generated_at', datetime.now().isoformat())}",
            f"\n**Dataset ID:** {result.get('dataset_id', 'dataset')}",
            "\n## Insights\n",
            str(result.get("ai_insights", "No insights")),
        ]
        md_path.write_text("\n".join(md))

    def _save_executive_outputs(self, result: Dict) -> None:
        ts = self.timestamp
        safe_ds = self._safe_ds(result.get("dataset_id"))

        exec_json = self.insights_dir / f"cross_platform_executive_summary_{safe_ds}_{ts}.json"
        exec_json.write_text(json.dumps(result, indent=2, default=str))

        dashboard = self.insights_dir / "cross_platform_executive_dashboard.json"
        dashboard_data = {
            "dashboard_type": "executive",
            "kpis": result.get("kpi_snapshot", {}),
            "executive_summary": result.get("executive_summary", ""),
            "last_updated": datetime.now().isoformat(),
        }
        dashboard.write_text(json.dumps(dashboard_data, indent=2, default=str))

        latest = self.insights_dir / "latest_executive_summary_cross_platform.json"
        latest.write_text(json.dumps(result, indent=2, default=str))

        report_md = self.insights_dir / f"cross_platform_executive_report_{safe_ds}_{ts}.md"
        lines = [
            "# Cross-Platform Executive Report\n",
            f"**Generated:** {result.get('generated_at', datetime.now().isoformat())}\n\n",
            "## KPIs\n",
            f"- Total Posts: {result.get('kpi_snapshot', {}).get('total_posts', 0):,}\n",
            f"- Avg Engagement Rate: {result.get('kpi_snapshot', {}).get('avg_engagement_rate', 0)}%\n",
            f"- Viral Threshold: {result.get('kpi_snapshot', {}).get('viral_threshold', 0)}\n\n",
            "## Executive Summary\n\n",
            str(result.get("executive_summary", "")),
            "\n",
        ]
        report_md.write_text("".join(lines))

    # --------------- LLM I/O ---------------
    async def _ollama(self, prompt: str) -> str:
        import aiohttp
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 2048},
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_host}/api/generate", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "No response generated")
                    return f"Error: HTTP {resp.status}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}"


async def run_all(skip_if_exists: bool = False) -> None:
    agent = CrossPlatformAnalyticsAgent()
    results: List[Dict[str, Any]] = []
    focuses = [
        "comprehensive",
        "performance_optimization",
        "content_strategy",
        "posting_optimization",
    ]
    with tqdm(total=len(focuses) + 1, desc="Cross Analyses", unit="task") as pbar:
        for f in focuses:
            results.append(await agent.analyze(f, skip_if_exists=skip_if_exists))
            pbar.update(1)
        results.append(await agent.generate_executive(skip_if_exists=skip_if_exists))
        pbar.update(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-Platform Analytics Agent")
    parser.add_argument("--focus", choices=["comprehensive", "performance_optimization", "content_strategy", "posting_optimization"], default="comprehensive")
    parser.add_argument("--all", action="store_true", help="Run all analyses plus executive")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--index-rag", action="store_true", help="After generation, index cross-platform artifacts")
    args = parser.parse_args()

    async def _run():
        if args.all:
            await run_all(skip_if_exists=args.skip_if_exists)
        else:
            agent = CrossPlatformAnalyticsAgent()
            await agent.analyze(args.focus, skip_if_exists=args.skip_if_exists)

    asyncio.run(_run())

    if args.index_rag:
        # Trigger RAG indexing for cross-platform artifacts
        import subprocess, sys, json as _json
        summary_path = Path("./metrics/cross_platform/latest_metrics_summary_cross_platform.json")
        dataset_id = None
        if summary_path.exists():
            try:
                dataset_id = (_json.loads(summary_path.read_text(encoding="utf-8", errors="ignore")) or {}).get("dataset_id")
            except Exception:
                pass
        if dataset_id:
            cmd = [
                sys.executable,
                "rag/indexer.py",
                "--platform", "cross_platform",
                "--dataset-id", str(dataset_id),
                "--metrics-dir", "./metrics/cross_platform",
                "--insights-dir", "./insights/cross_platform",
            ]
            try:
                subprocess.run(cmd, check=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()


