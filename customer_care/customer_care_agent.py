#!/usr/bin/env python3
"""
Customer Care Analytics Agent - Parity with Facebook agent, adapted to care data
Generates focus insights and executive outputs from metrics in ./metrics/customer_care
"""
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class CustomerCareAnalyticsAgent:
    """Single agent that consumes Customer Care metrics and produces insights"""

    def __init__(self, metrics_dir: str = "./metrics/customer_care", insights_dir: str = "./insights/customer_care", ollama_host: str = "http://localhost:11434"):
        self.metrics_dir = Path(metrics_dir)
        self.insights_dir = Path(insights_dir)
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_host = ollama_host
        self.name = "customer_care_analytics"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def analyze_performance(self, focus_area: str = "comprehensive", skip_if_exists: bool = False) -> Dict[str, Any]:
        metrics = self._load_all_metrics()
        if not metrics:
            raise FileNotFoundError("No Customer Care metrics found. Run ingestion first.")

        dataset_id = metrics.get("summary", {}).get("dataset_id")
        if skip_if_exists and dataset_id:
            existing = self._existing_analysis_file(focus_area, dataset_id)
            if existing:
                return {"skipped": True, "reason": "analysis_already_exists", "focus": focus_area, "dataset_id": dataset_id, "existing_file": str(existing), "generated_at": datetime.now().isoformat()}

        if focus_area == "comprehensive":
            return await self._comprehensive_analysis(metrics)
        elif focus_area == "operations_optimization":
            return await self._operations_optimization_analysis(metrics)
        elif focus_area == "escalation_prevention":
            return await self._escalation_prevention_analysis(metrics)
        elif focus_area == "service_quality":
            return await self._service_quality_analysis(metrics)
        else:
            return await self._comprehensive_analysis(metrics)

    def _load_all_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        summary_file = self.metrics_dir / "latest_metrics_summary_customer_care.json"
        if summary_file.exists():
            with open(summary_file) as f:
                metrics["summary"] = json.load(f)

        metric_patterns = [
            "customer_care_issue_type_performance",
            "customer_care_channel_performance",
            "customer_care_priority_performance",
            "customer_care_temporal_analytics",
            "customer_care_top_negative",
            "customer_care_worst_experience",
            "customer_care_best_experience",
            "customer_care_escalation_analytics",
            "customer_care_keywords",
            "customer_care_product_mentions",
        ]
        for pattern in metric_patterns:
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
        pattern = f"ai_insights_customer_care_{focus}_{safe_ds}_*.json"
        files = list(self.insights_dir.glob(pattern))
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
        return None

    def _existing_exec_file(self, dataset_id: str) -> Optional[Path]:
        safe_ds = self._safe_dataset_id(dataset_id)
        pattern = f"customer_care_executive_summary_{safe_ds}_*.json"
        files = list(self.insights_dir.glob(pattern))
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
        return None

    def _metrics_ready(self) -> bool:
        summary_ok = (self.metrics_dir / "latest_metrics_summary_customer_care.json").exists()
        any_detail = any(self.metrics_dir.glob("customer_care_*_performance_*.json")) or \
                     any(self.metrics_dir.glob("customer_care_temporal_analytics_*.json"))
        return summary_ok and any_detail

    def _insights_ready(self, dataset_id: Optional[str]) -> bool:
        safe_ds = self._safe_dataset_id(dataset_id)
        focuses = [
            'comprehensive', 'operations_optimization', 'escalation_prevention', 'service_quality'
        ]
        focus_ok = all(list(self.insights_dir.glob(f"ai_insights_customer_care_{f}_{safe_ds}_*.json")) for f in focuses)
        exec_ok = any(self.insights_dir.glob(f"customer_care_executive_summary_{safe_ds}_*.json"))
        return focus_ok and exec_ok

    async def _comprehensive_analysis(self, metrics: Dict) -> Dict[str, Any]:
        summary = metrics.get("summary", {})
        issue_perf = metrics.get("customer_care_issue_type_performance", {})
        channel_perf = metrics.get("customer_care_channel_performance", {})
        temporal = metrics.get("customer_care_temporal_analytics", {})
        prompt = f"""
CUSTOMER CARE PERFORMANCE ANALYSIS

DATASET OVERVIEW:
- Total Cases: {summary.get('quick_access', {}).get('dataset_overview', {}).get('total_cases', 0):,}
- Escalation Rate: {summary.get('quick_access', {}).get('dataset_overview', {}).get('escalation_rate', 0):.2f}
- Average Sentiment: {summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_sentiment', 0):.2f}
- Average Urgency: {summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_urgency', 0):.2f}

ISSUE TYPE HIGHLIGHTS:
{json.dumps(issue_perf.get('issue_rankings', {}), indent=2)}

CHANNEL HIGHLIGHTS:
{json.dumps(channel_perf.get('channel_rankings', {}), indent=2)}

TEMPORAL INSIGHTS:
{json.dumps(temporal.get('optimal_times', {}), indent=2)}

Provide actionable operations and customer experience insights.
"""
        ai_response = await self._get_ollama_response(prompt)
        result = {
            "analysis_type": "comprehensive",
            "ai_insights": ai_response,
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        self._save_analysis_outputs(result, "comprehensive")
        self._save_analysis_markdown(result, "comprehensive")
        return result

    async def _operations_optimization_analysis(self, metrics: Dict) -> Dict[str, Any]:
        summary = metrics.get("summary", {})
        temporal = metrics.get("customer_care_temporal_analytics", {})
        escalation = metrics.get("customer_care_escalation_analytics", {})
        prompt = "Focus on queue and staffing optimization, escalation hotspots, and resolution time reductions using temporal and escalation analytics."
        ai_response = await self._get_ollama_response(prompt)
        result = {
            "analysis_type": "operations_optimization",
            "ai_insights": ai_response,
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        self._save_analysis_outputs(result, "operations_optimization")
        self._save_analysis_markdown(result, "operations_optimization")
        return result

    async def _escalation_prevention_analysis(self, metrics: Dict) -> Dict[str, Any]:
        summary = metrics.get("summary", {})
        issue_perf = metrics.get("customer_care_issue_type_performance", {})
        top_negative = metrics.get("customer_care_top_negative", {})
        prompt = "Propose prevention strategies to reduce escalations and negative experiences based on issue rankings and top negative cases."
        ai_response = await self._get_ollama_response(prompt)
        result = {
            "analysis_type": "escalation_prevention",
            "ai_insights": ai_response,
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        self._save_analysis_outputs(result, "escalation_prevention")
        self._save_analysis_markdown(result, "escalation_prevention")
        return result

    async def _service_quality_analysis(self, metrics: Dict) -> Dict[str, Any]:
        summary = metrics.get("summary", {})
        worst_ex = metrics.get("customer_care_worst_experience", {})
        best_ex = metrics.get("customer_care_best_experience", {})
        prompt = "Assess service quality trends and recommend improvements leveraging worst/best experience cases and CSAT patterns."
        ai_response = await self._get_ollama_response(prompt)
        result = {
            "analysis_type": "service_quality",
            "ai_insights": ai_response,
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        self._save_analysis_outputs(result, "service_quality")
        self._save_analysis_markdown(result, "service_quality")
        return result

    def _save_analysis_outputs(self, result: Dict, focus: str):
        ts = self.timestamp
        ds = result.get('dataset_id') or 'dataset'
        safe_ds = ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(ds))
        json_file = self.insights_dir / f"ai_insights_customer_care_{focus}_{safe_ds}_{ts}.json"
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
        safe_ds = ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(ds))
        md_file = self.insights_dir / f"ai_insights_customer_care_{focus}_{safe_ds}_{ts}.md"
        with open(md_file, 'w') as f:
            f.write(f"# Customer Care AI Insights - {focus.replace('_',' ').title()}\n\n")
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
Generate an executive summary for Customer Care performance:

KEY METRICS:
- Total Cases: {summary.get('quick_access', {}).get('dataset_overview', {}).get('total_cases', 0):,}
- Escalation Rate: {summary.get('quick_access', {}).get('dataset_overview', {}).get('escalation_rate', 0):.2f}
- Average Resolution Time (h): {summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_resolution_time_hours', 0):.2f}

Provide a concise executive summary:
1) Overall support performance
2) Top drivers of escalations and negative outcomes
3) Operational recommendations and expected business impact
"""
        ai_response = await self._get_ollama_response(prompt)

        result = {
            "executive_summary": ai_response,
            "kpi_snapshot": {
                "total_cases": summary.get('quick_access', {}).get('dataset_overview', {}).get('total_cases', 0),
                "avg_sentiment": summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_sentiment', 0),
                "avg_urgency": summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_urgency', 0),
                "escalation_rate": summary.get('quick_access', {}).get('dataset_overview', {}).get('escalation_rate', 0),
                "avg_resolution_time_hours": summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_resolution_time_hours', 0),
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
        exec_json = self.insights_dir / f"customer_care_executive_summary_{safe_ds}_{ts}.json"
        with open(exec_json, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # 2) Latest dashboard JSON
        dashboard_file = self.insights_dir / "customer_care_executive_dashboard.json"
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
        latest_exec = self.insights_dir / "latest_executive_summary_customer_care.json"
        with open(latest_exec, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # 4) Executive markdown
        md_file = self.insights_dir / f"customer_care_executive_report_{safe_ds}_{ts}.md"
        with open(md_file, 'w') as f:
            f.write("# Customer Care Performance - Executive Report\n\n")
            f.write(f"**Generated:** {result.get('generated_at')}\n\n")
            if result.get('dataset_id'):
                f.write(f"**Dataset ID:** {result['dataset_id']}\n\n")
            kpis = result.get('kpi_snapshot', {})
            f.write("## Key Performance Indicators\n\n")
            f.write(f"- **Total Cases:** {kpis.get('total_cases', 0)}\n")
            f.write(f"- **Average Sentiment:** {kpis.get('avg_sentiment', 0):.2f}\n")
            f.write(f"- **Average Urgency:** {kpis.get('avg_urgency', 0):.2f}\n")
            f.write(f"- **Escalation Rate:** {kpis.get('escalation_rate', 0):.2f}\n")
            f.write(f"- **Average Resolution Time (h):** {kpis.get('avg_resolution_time_hours', 0):.2f}\n\n")
            f.write("## Executive Summary\n\n")
            f.write(result.get('executive_summary', ''))


async def analyze_customer_care_performance(focus_area: str = "comprehensive") -> Dict[str, Any]:
    agent = CustomerCareAnalyticsAgent()
    return await agent.analyze_performance(focus_area)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import sys, subprocess

    parser = argparse.ArgumentParser(description='Customer Care Analytics Agent')
    parser.add_argument('--focus', choices=['comprehensive', 'operations_optimization', 'escalation_prevention', 'service_quality'], 
                       default='comprehensive', help='Analysis focus area')
    parser.add_argument('--executive', action='store_true', help='Generate executive summary only')
    parser.add_argument('--skip-if-exists', action='store_true', help='Skip generation if a report for the dataset already exists')
    parser.add_argument('--all', action='store_true', help='Generate all focus insights and executive outputs')
    args = parser.parse_args()

    async def main():
        agent = CustomerCareAnalyticsAgent()

        if args.all:
            focuses = [
                'comprehensive',
                'operations_optimization',
                'escalation_prevention',
                'service_quality',
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
            # After ALL outputs, trigger RAG indexing if metrics + insights are present
            ds = (results[0] or {}).get('dataset_id') if results else None
            if agent._metrics_ready() and agent._insights_ready(ds):
                cmd = [
                    sys.executable,
                    'rag/indexer.py',
                    '--platform', 'customer_care',
                    '--dataset-id', str(ds or 'dataset'),
                    '--metrics-dir', './metrics/customer_care',
                    '--insights-dir', './insights/customer_care',
                ]
                subprocess.run(cmd, check=False)
            return

        if args.executive:
            result = await agent.generate_executive_summary(skip_if_exists=args.skip_if_exists)
            print("\n" + "="*50)
            print("CUSTOMER CARE EXECUTIVE SUMMARY")
            print("="*50)
            print(result.get("executive_summary", "No summary"))
        else:
            result = await agent.analyze_performance(args.focus, skip_if_exists=args.skip_if_exists)
            print("\n" + "="*50)
            print(f"CUSTOMER CARE ANALYSIS - {args.focus.upper()}")
            print("="*50)
            print(result.get("ai_insights", "No insights"))

    asyncio.run(main())
