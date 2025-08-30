#!/usr/bin/env python3
"""
Instagram Analytics Agent - Simplified approach using metrics files
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class InstagramAnalyticsAgent:
    """Single agent that consumes Instagram metrics and produces insights"""

    def __init__(self, metrics_dir: str = "./metrics/instagram", insights_dir: str = "./insights/instagram", ollama_host: str = "http://localhost:11434"):
        self.metrics_dir = Path(metrics_dir)
        self.insights_dir = Path(insights_dir)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_host = ollama_host
        self.name = "instagram_analytics"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def analyze_performance(self, focus_area: str = "comprehensive", skip_if_exists: bool = False) -> Dict[str, Any]:
        metrics = self._load_all_metrics()
        if not metrics:
            raise FileNotFoundError("No Instagram metrics found. Run ingestion first.")

        dataset_id = metrics.get("summary", {}).get("dataset_id")
        if skip_if_exists and dataset_id:
            existing = self._existing_analysis_file(focus_area, dataset_id)
            if existing:
                return {"skipped": True, "reason": "analysis_already_exists", "focus": focus_area, "dataset_id": dataset_id, "existing_file": str(existing), "generated_at": datetime.now().isoformat()}

        if focus_area == "comprehensive":
            return await self._comprehensive_analysis(metrics)
        elif focus_area == "performance_optimization":
            return await self._performance_optimization_analysis(metrics)
        elif focus_area == "content_strategy":
            return await self._content_strategy_analysis(metrics)
        elif focus_area == "posting_optimization":
            return await self._posting_optimization_analysis(metrics)
        else:
            return await self._comprehensive_analysis(metrics)

    def _load_all_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        summary_file = self.metrics_dir / "latest_metrics_summary_instagram.json"
        if summary_file.exists():
            with open(summary_file) as f:
                metrics["summary"] = json.load(f)

        metric_types = [
            "instagram_brand_performance",
            "instagram_content_type_performance",
            "instagram_temporal_analytics",
            "instagram_top_performers",
            "instagram_worst_performers",
        ]
        for pattern in metric_types:
            latest = self._find_latest_metric_file(pattern)
            if latest:
                with open(latest) as f:
                    metrics[pattern] = json.load(f)
        return metrics

    def _find_latest_metric_file(self, prefix: str) -> Optional[Path]:
        files = list(self.metrics_dir.glob(f"{prefix}_*.json"))
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
        return None

    def _safe_dataset_id(self, dataset_id: Optional[str]) -> str:
        value = dataset_id or "dataset"
        return ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(value))

    def _existing_analysis_file(self, focus: str, dataset_id: str) -> Optional[Path]:
        safe_ds = self._safe_dataset_id(dataset_id)
        pattern = f"ai_insights_instagram_{focus}_{safe_ds}_*.json"
        files = list(self.insights_dir.glob(pattern))
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
        return None

    def _existing_exec_file(self, dataset_id: str) -> Optional[Path]:
        safe_ds = self._safe_dataset_id(dataset_id)
        pattern = f"instagram_executive_summary_{safe_ds}_*.json"
        files = list(self.insights_dir.glob(pattern))
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
        return None

    def _metrics_ready(self) -> bool:
        summary_ok = (self.metrics_dir / "latest_metrics_summary_instagram.json").exists()
        any_detail = any(self.metrics_dir.glob("instagram_*_performance_*.json")) or \
                     any(self.metrics_dir.glob("instagram_temporal_analytics_*.json"))
        return summary_ok and any_detail

    def _insights_ready(self, dataset_id: Optional[str]) -> bool:
        safe_ds = self._safe_dataset_id(dataset_id)
        focuses = [
            'comprehensive', 'performance_optimization', 'content_strategy', 'posting_optimization'
        ]
        focus_ok = all(list(self.insights_dir.glob(f"ai_insights_instagram_{f}_{safe_ds}_*.json")) for f in focuses)
        exec_ok = any(self.insights_dir.glob(f"instagram_executive_summary_{safe_ds}_*.json"))
        return focus_ok and exec_ok

    async def _comprehensive_analysis(self, metrics: Dict) -> Dict[str, Any]:
        summary = metrics.get("summary", {})
        top_performers = metrics.get("instagram_top_performers", {})
        worst_performers = metrics.get("instagram_worst_performers", {})

        analysis_prompt = f"""
INSTAGRAM PERFORMANCE ANALYSIS

DATASET OVERVIEW:
- Total Posts: {summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0):,}
- Average Engagement Rate: {summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0):.2f}%

TOP PERFORMERS:
{json.dumps(top_performers.get('top_posts', [])[:5], indent=2)}

WORST PERFORMERS:
{json.dumps(worst_performers.get('worst_posts', [])[:5], indent=2)}

Provide actionable business insights focused on content performance and posting optimization.
"""
        ai_response = await self._get_ollama_response(analysis_prompt)
        result = {
            "analysis_type": "comprehensive",
            "ai_insights": ai_response,
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        self._save_analysis_outputs(result, "comprehensive")
        self._save_analysis_markdown(result, "comprehensive")
        return result

    async def _performance_optimization_analysis(self, metrics: Dict) -> Dict[str, Any]:
        summary = metrics.get("summary", {})
        worst_performers = metrics.get("instagram_worst_performers", {})
        prompt = "Focus on concrete steps to improve underperforming Instagram content based on worst performers."
        ai_response = await self._get_ollama_response(prompt)
        result = {
            "analysis_type": "performance_optimization",
            "ai_insights": ai_response,
            "underperformer_count": len(worst_performers.get('worst_posts', [])),
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        self._save_analysis_outputs(result, "performance_optimization")
        self._save_analysis_markdown(result, "performance_optimization")
        return result

    async def _content_strategy_analysis(self, metrics: Dict) -> Dict[str, Any]:
        summary = metrics.get("summary", {})
        prompt = "Give content strategy recommendations for Instagram based on brand and content-type performance."
        ai_response = await self._get_ollama_response(prompt)
        result = {"analysis_type": "content_strategy", "ai_insights": ai_response, "dataset_id": summary.get('dataset_id'), "generated_at": datetime.now().isoformat()}
        self._save_analysis_outputs(result, "content_strategy")
        self._save_analysis_markdown(result, "content_strategy")
        return result

    async def _posting_optimization_analysis(self, metrics: Dict) -> Dict[str, Any]:
        summary = metrics.get("summary", {})
        prompt = "Recommend optimal posting times for Instagram based on temporal analytics."
        ai_response = await self._get_ollama_response(prompt)
        result = {"analysis_type": "posting_optimization", "ai_insights": ai_response, "dataset_id": summary.get('dataset_id'), "generated_at": datetime.now().isoformat()}
        self._save_analysis_outputs(result, "posting_optimization")
        self._save_analysis_markdown(result, "posting_optimization")
        return result

    def _save_analysis_outputs(self, result: Dict, focus: str):
        ts = self.timestamp
        ds = result.get('dataset_id') or 'dataset'
        safe_ds = ''.join(c if (c.isalnum() or c in ('-','_')) else '-' for c in str(ds))
        json_file = self.insights_dir / f"ai_insights_instagram_{focus}_{safe_ds}_{ts}.json"
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    async def _get_ollama_response(self, prompt: str) -> str:
        import aiohttp
        payload = {"model": "llama3.2:3b", "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 1024}}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_host}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "No response generated")
                    return f"Error: HTTP {response.status}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

    def _save_analysis_markdown(self, result: Dict, focus: str):
        ts = self.timestamp
        ds = result.get('dataset_id') or 'dataset'
        safe_ds = ''.join(c if (c.isalnum() or c in ('-','_')) else '-' for c in str(ds))
        md_file = self.insights_dir / f"ai_insights_instagram_{focus}_{safe_ds}_{ts}.md"
        with open(md_file, 'w') as f:
            f.write(f"# Instagram AI Insights - {focus.replace('_',' ').title()}\n\n")
            f.write(f"**Generated:** {result.get('generated_at', datetime.now().isoformat())}\n\n")
            if result.get('dataset_id'):
                f.write(f"**Dataset ID:** {result['dataset_id']}\n\n")
            f.write("## Insights\n\n")
            f.write(result.get('ai_insights', 'No insights'))

    async def generate_executive_summary(self, skip_if_exists: bool = False) -> Dict[str, Any]:
        metrics = self._load_all_metrics()
        summary = metrics.get("summary", {})
        dataset_id = summary.get('dataset_id')

        if skip_if_exists and dataset_id:
            existing = self._existing_exec_file(dataset_id)
            if existing:
                return {"skipped": True, "reason": "executive_already_exists", "dataset_id": dataset_id, "existing_file": str(existing), "generated_at": datetime.now().isoformat()}

        prompt = f"""
Generate an executive summary for Instagram marketing performance:

KEY METRICS:
- Total Posts Analyzed: {summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0):,}
- Portfolio Engagement Rate: {summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0):.2f}%

Provide a concise executive summary:
1) Overall performance assessment
2) Key opportunities and risks
3) Strategic recommendations with business impact
"""
        ai_response = await self._get_ollama_response(prompt)

        result = {
            "executive_summary": ai_response,
            "kpi_snapshot": {
                "total_posts": summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0),
                "avg_engagement_rate": summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0),
                "warning_count": 0
            },
            "dataset_id": dataset_id,
            "generated_at": datetime.now().isoformat()
        }

        self._save_executive_outputs(result)
        return result

    def _save_executive_outputs(self, result: Dict):
        ts = self.timestamp
        ds = result.get('dataset_id') or 'dataset'
        safe_ds = ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(ds))

        # 1) Executive JSON
        exec_json = self.insights_dir / f"instagram_executive_summary_{safe_ds}_{ts}.json"
        with open(exec_json, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # 2) Latest dashboard JSON
        dashboard_file = self.insights_dir / "instagram_executive_dashboard.json"
        dashboard_data = {
            "dashboard_type": "executive",
            "kpis": result.get("kpi_snapshot", {}),
            "executive_summary": result.get("executive_summary", ""),
            "last_updated": datetime.now().isoformat(),
            "data_freshness": "Real-time from latest ingestion",
            "quick_actions": [],
            "alert_status": "HEALTHY"
        }
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        # 3) Latest executive summary for quick access
        latest_exec = self.insights_dir / "latest_executive_summary_instagram.json"
        with open(latest_exec, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # 4) Executive markdown
        md_file = self.insights_dir / f"instagram_executive_report_{safe_ds}_{ts}.md"
        with open(md_file, 'w') as f:
            f.write("# Instagram Marketing Performance - Executive Report\n\n")
            f.write(f"**Generated:** {result.get('generated_at')}\n\n")
            if result.get('dataset_id'):
                f.write(f"**Dataset ID:** {result['dataset_id']}\n\n")
            kpis = result.get('kpi_snapshot', {})
            f.write("## Key Performance Indicators\n\n")
            f.write(f"- **Total Posts Analyzed:** {kpis.get('total_posts', 0)}\n")
            f.write(f"- **Average Engagement Rate:** {kpis.get('avg_engagement_rate', 0):.2f}%\n")
            f.write(f"- **Warning Signals:** {kpis.get('warning_count', 0)} issues detected\n\n")
            f.write("## Executive Summary\n\n")
            f.write(result.get('executive_summary', ''))


async def analyze_instagram_performance(focus_area: str = "comprehensive") -> Dict[str, Any]:
    agent = InstagramAnalyticsAgent()
    return await agent.analyze_performance(focus_area)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Instagram Analytics Agent')
    parser.add_argument('--focus', choices=['comprehensive', 'performance_optimization', 'content_strategy', 'posting_optimization'], 
                       default='comprehensive', help='Analysis focus area')
    parser.add_argument('--executive', action='store_true', help='Generate executive summary only')
    parser.add_argument('--skip-if-exists', action='store_true', help='Skip generation if a report for the dataset already exists')
    parser.add_argument('--all', action='store_true', help='Generate all focus insights and executive outputs')
    parser.add_argument('--index-rag', action='store_true', help='After generation, index outputs into the RAG')
    args = parser.parse_args()

    async def main():
        agent = InstagramAnalyticsAgent()

        if args.all:
            focuses = [
                'comprehensive',
                'performance_optimization',
                'content_strategy',
                'posting_optimization',
            ]
            results = []
            with tqdm(total=len(focuses) + 1, desc="Analyses", unit="task") as pbar:
                for f in focuses:
                    results.append(await agent.analyze_performance(f, skip_if_exists=args.skip_if_exists))
                    pbar.update(1)
                results.append(await agent.generate_executive_summary(skip_if_exists=args.skip_if_exists))
                pbar.update(1)

            print("\nGenerated the following outputs:")
            for r in results:
                if r.get('skipped'):
                    print(f"- {r.get('focus','executive')}: skipped (exists) → {r.get('existing_file')}")
                else:
                    print(f"- {r.get('analysis_type','executive')}: saved for dataset {r.get('dataset_id')}")

            if args.index_rag:
                ds = None
                for r in results:
                    if r.get('dataset_id'):
                        ds = r['dataset_id']
                        break
                if agent._metrics_ready() and agent._insights_ready(ds):
                    import sys, subprocess
                    cmd = [
                        sys.executable,
                        'rag/indexer.py',
                        '--platform', 'instagram',
                        '--dataset-id', str(ds or 'dataset'),
                        '--metrics-dir', './metrics/instagram',
                        '--insights-dir', './insights/instagram',
                    ]
                    subprocess.run(cmd, check=False)
            return

        if args.executive:
            result = await agent.generate_executive_summary(skip_if_exists=args.skip_if_exists)
            print("\n" + "="*50)
            print("INSTAGRAM EXECUTIVE SUMMARY")
            print("="*50)
            print(result.get("executive_summary", "No summary"))
        else:
            result = await agent.analyze_performance(args.focus, skip_if_exists=args.skip_if_exists)
            print("\n" + "="*50)
            print(f"INSTAGRAM ANALYSIS - {args.focus.upper()}")
            print("="*50)
            print(result.get("ai_insights", "No insights"))

    asyncio.run(main())


