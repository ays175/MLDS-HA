#!/usr/bin/env python3
"""
TikTok Analytics Agent - Simplified approach using metrics files
"""
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

class TikTokAnalyticsAgent:
    """Single agent that consumes TikTok metrics and produces insights"""
    
    def __init__(self, metrics_dir="./metrics/tiktok", insights_dir="./insights/tiktok", ollama_host="http://localhost:11434"):
        self.metrics_dir = Path(metrics_dir)
        self.insights_dir = Path(insights_dir)
        self.insights_dir.mkdir(parents=True, exist_ok=True)  # Create insights directory
        self.ollama_host = ollama_host
        self.name = "tiktok_analytics"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def analyze_performance(self, focus_area: str = "comprehensive", skip_if_exists: bool = False) -> Dict[str, Any]:
        """Main analysis method - loads metrics and produces insights"""
        
        print(f"Loading TikTok metrics for {focus_area} analysis...")
        
        # Load all available metrics
        metrics = self._load_all_metrics()
        
        if not metrics:
            raise FileNotFoundError("No TikTok metrics found. Run ingestion first.")
        
        # Optional skip if an analysis for this dataset already exists
        dataset_id = metrics.get("summary", {}).get("dataset_id")
        if skip_if_exists and dataset_id:
            existing = self._existing_analysis_file(focus_area, dataset_id)
            if existing:
                return {
                    "skipped": True,
                    "reason": "analysis_already_exists",
                    "focus": focus_area,
                    "dataset_id": dataset_id,
                    "existing_file": str(existing),
                    "generated_at": datetime.now().isoformat()
                }
        
        # Generate insights based on focus area
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
        """Load all TikTok metrics from files"""
        
        metrics = {}
        
        # Always load latest summary first (prefer tiktok-prefixed)
        for name in ("latest_metrics_summary_tiktok.json", "latest_metrics_summary.json"):
            summary_file = self.metrics_dir / name
            if summary_file.exists():
                with open(summary_file) as f:
                    metrics["summary"] = json.load(f)
                break
        
        # Load detailed metrics (find most recent files)
        metric_types = [
            "brand_performance",
            "content_type_performance", 
            "temporal_analytics",
            "duration_performance",
            "top_performers",
            "worst_performers"
        ]
        
        for metric_type in metric_types:
            latest_file = self._find_latest_metric_file(metric_type)
            if latest_file:
                with open(latest_file) as f:
                    metrics[metric_type] = json.load(f)
        
        return metrics
    
    def _find_latest_metric_file(self, metric_type: str) -> Optional[Path]:
        """Find the most recent metric file of given type"""
        
        # Prefer tiktok-prefixed filenames, fall back to legacy
        files = []
        for pattern in (f"tiktok_{metric_type}_*.json", f"{metric_type}_*.json"):
            matches = list(self.metrics_dir.glob(pattern))
            if matches:
                files = matches
                break
        
        if files:
            # Sort by modification time, return newest
            return max(files, key=lambda f: f.stat().st_mtime)
        
        return None

    # ---------- Existence checks ----------
    def _safe_dataset_id(self, dataset_id: Optional[str]) -> str:
        value = dataset_id or "dataset"
        return ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(value))

    def _existing_analysis_file(self, focus: str, dataset_id: str) -> Optional[Path]:
        safe_ds = self._safe_dataset_id(dataset_id)
        pattern = f"ai_insights_{focus}_{safe_ds}_*.json"
        files = list(self.insights_dir.glob(pattern))
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
        return None

    def _existing_executive_file(self, dataset_id: str) -> Optional[Path]:
        safe_ds = self._safe_dataset_id(dataset_id)
        pattern = f"executive_summary_{safe_ds}_*.json"
        files = list(self.insights_dir.glob(pattern))
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
        return None
    
    async def _comprehensive_analysis(self, metrics: Dict) -> Dict[str, Any]:
        """Generate comprehensive TikTok performance analysis"""
        
        summary = metrics.get("summary", {})
        brand_data = metrics.get("brand_performance", {})
        content_data = metrics.get("content_type_performance", {})
        temporal_data = metrics.get("temporal_analytics", {})
        top_performers = metrics.get("top_performers", {})
        worst_performers = metrics.get("worst_performers", {})
        
        # Create structured prompt using actual data
        analysis_prompt = f"""
TIKTOK PERFORMANCE ANALYSIS

DATASET OVERVIEW:
- Total Posts: {summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0):,}
- Average Engagement Rate: {summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0):.2f}%
- Viral Threshold: {summary.get('quick_access', {}).get('dataset_overview', {}).get('viral_threshold', 0):,.0f} views

TOP PERFORMERS:
{self._format_top_performers(summary.get('quick_access', {}).get('top_performing_posts', []))}

WORST PERFORMERS:
{self._format_worst_performers(summary.get('quick_access', {}).get('worst_performing_posts', []))}

BRAND PERFORMANCE:
{self._format_brand_performance(summary.get('quick_access', {}).get('top_brands', []))}

CONTENT STRATEGY:
{self._format_content_recommendations(summary.get('ai_recommendations', {}).get('content_strategy', []))}

TIMING OPTIMIZATION:
- Best Hour: {summary.get('quick_access', {}).get('optimal_posting', {}).get('best_hour', 'N/A')}:00
- Best Day: {summary.get('quick_access', {}).get('optimal_posting', {}).get('best_day', 'N/A')}
- Avoid Hours: {summary.get('quick_access', {}).get('avoid_posting', {}).get('problematic_hours', [])}

WARNING SIGNALS:
{self._format_warning_signals(summary.get('warning_signals', []))}

Provide actionable business insights focusing on:
1. Performance optimization opportunities
2. Content strategy recommendations  
3. Timing and posting optimization
4. Brand collaboration priorities
5. Risk mitigation for underperformers

Format as structured business recommendations with specific metrics and actions.
"""
        
        # Get AI analysis
        ai_response = await self._get_ollama_response(analysis_prompt)
        
        # Save comprehensive analysis
        result = {
            "analysis_type": "comprehensive",
            "ai_insights": ai_response,
            "data_summary": {
                "total_posts": summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0),
                "avg_engagement_rate": summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0),
                "top_brand": summary.get('quick_access', {}).get('top_brands', [{}])[0].get('brand', 'N/A') if summary.get('quick_access', {}).get('top_brands') else 'N/A',
                "optimal_hour": summary.get('quick_access', {}).get('optimal_posting', {}).get('best_hour', 0),
                "warning_count": len(summary.get('warning_signals', []))
            },
            "recommendations": summary.get('ai_recommendations', {}),
            "raw_metrics": {
                "top_performers": top_performers,
                "worst_performers": worst_performers,
                "brand_performance": brand_data,
                "temporal_analytics": temporal_data
            },
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        
        # Save all outputs
        self._save_analysis_outputs(result, "comprehensive")
        
        return result
    
    async def _performance_optimization_analysis(self, metrics: Dict) -> Dict[str, Any]:
        """Focus specifically on performance optimization"""
        
        summary = metrics.get("summary", {})
        worst_performers = metrics.get("worst_performers", {})
        
        prompt = f"""
TIKTOK PERFORMANCE OPTIMIZATION ANALYSIS

UNDERPERFORMERS TO FIX:
{self._format_worst_performers_detailed(worst_performers)}

IMPROVEMENT OPPORTUNITIES:
{self._format_improvement_actions(summary.get('ai_recommendations', {}).get('improvement_actions', []))}

CONTENT TO AVOID:
{self._format_avoidance_insights(summary.get('ai_recommendations', {}).get('content_to_avoid', []))}

Focus on specific, actionable steps to improve underperforming content.
Provide concrete optimization tactics with expected impact.
"""
        
        ai_response = await self._get_ollama_response(prompt)
        
        result = {
            "analysis_type": "performance_optimization",
            "ai_insights": ai_response,
            "underperformer_count": len(worst_performers.get('worst_posts', [])),
            "improvement_actions": summary.get('ai_recommendations', {}).get('improvement_actions', []),
            "raw_data": {
                "worst_performers": worst_performers,
                "avoidance_insights": summary.get('ai_recommendations', {}).get('content_to_avoid', [])
            },
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        
        # Save outputs
        self._save_analysis_outputs(result, "performance_optimization")
        
        return result
    
    async def _content_strategy_analysis(self, metrics: Dict) -> Dict[str, Any]:
        """Focus on content strategy and brand optimization"""
        
        brand_data = metrics.get("brand_performance", {})
        content_data = metrics.get("content_type_performance", {})
        duration_data = metrics.get("duration_performance", {})
        
        prompt = f"""
TIKTOK CONTENT STRATEGY ANALYSIS

BRAND PERFORMANCE:
{self._format_detailed_brand_performance(brand_data)}

CONTENT TYPE EFFECTIVENESS:
{self._format_detailed_content_performance(content_data)}

DURATION OPTIMIZATION:
{self._format_duration_insights(duration_data)}

Focus on content strategy recommendations for brand partnerships and content creation.
Provide specific guidance on what content types and formats work best.
"""
        
        ai_response = await self._get_ollama_response(prompt)
        
        result = {
            "analysis_type": "content_strategy", 
            "ai_insights": ai_response,
            "top_brands": brand_data.get('brand_rankings', {}).get('top_by_engagement_rate', [])[:3],
            "optimal_duration": duration_data.get('optimal_duration', {}),
            "dataset_id": metrics.get('summary', {}).get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        self._save_analysis_outputs(result, "content_strategy")
        return result
    
    async def _posting_optimization_analysis(self, metrics: Dict) -> Dict[str, Any]:
        """Focus on posting timing and frequency optimization"""
        
        temporal_data = metrics.get("temporal_analytics", {})
        summary = metrics.get("summary", {})
        
        prompt = f"""
TIKTOK POSTING OPTIMIZATION ANALYSIS

TIMING INSIGHTS:
{self._format_temporal_insights(temporal_data)}

OPTIMAL SCHEDULE:
{self._format_scheduling_recommendations(summary.get('ai_recommendations', {}).get('posting_schedule', []))}

TIMES TO AVOID:
{self._format_avoid_posting_times(summary.get('quick_access', {}).get('avoid_posting', {}))}

Focus on specific posting schedule recommendations with expected engagement improvements.
Provide concrete timing strategy with hourly and daily optimization.
"""
        
        ai_response = await self._get_ollama_response(prompt)
        
        result = {
            "analysis_type": "posting_optimization",
            "ai_insights": ai_response,
            "optimal_posting": summary.get('quick_access', {}).get('optimal_posting', {}),
            "avoid_posting": summary.get('quick_access', {}).get('avoid_posting', {}),
            "dataset_id": summary.get('dataset_id'),
            "generated_at": datetime.now().isoformat()
        }
        self._save_analysis_outputs(result, "posting_optimization")
        return result
    
    # Helper formatting methods
    def _format_top_performers(self, top_posts: List) -> str:
        if not top_posts:
            return "No top performer data available"
        
        formatted = ""
        for i, post in enumerate(top_posts[:3], 1):
            formatted += f"{i}. Post {post.get('tiktok_id', 'N/A')}: {post.get('engagement_rate', 0):.1f}% engagement, {post.get('tiktok_insights_completion_rate', 0):.1f}% completion\n"
        
        return formatted
    
    def _format_worst_performers(self, worst_posts: List) -> str:
        if not worst_posts:
            return "No worst performer data available"
        
        formatted = ""
        for i, post in enumerate(worst_posts[:3], 1):
            formatted += f"{i}. Post {post.get('tiktok_id', 'N/A')}: {post.get('engagement_rate', 0):.1f}% engagement, {post.get('tiktok_insights_completion_rate', 0):.1f}% completion\n"
        
        return formatted
    
    def _format_brand_performance(self, top_brands: List) -> str:
        if not top_brands:
            return "No brand performance data available"
        
        formatted = ""
        for i, brand in enumerate(top_brands[:3], 1):
            formatted += f"{i}. {brand.get('brand', 'N/A')}: {brand.get('avg_engagement_rate', 0):.1f}% engagement\n"
        
        return formatted
    
    def _format_content_recommendations(self, recommendations: List) -> str:
        if not recommendations:
            return "No content recommendations available"
        
        return "\n".join([f"- {rec}" for rec in recommendations[:5]])
    
    def _format_warning_signals(self, warnings: List) -> str:
        if not warnings:
            return "No warning signals detected"
        
        formatted = ""
        for warning in warnings:
            formatted += f"- {warning.get('signal', 'Unknown')}: {warning.get('action', 'No action specified')}\n"
        
        return formatted
    
    def _format_worst_performers_detailed(self, worst_data: Dict) -> str:
        if not worst_data:
            return "No underperformer analysis available"
        
        anti_patterns = worst_data.get('anti_patterns', {})
        return f"""
Problematic Brands: {anti_patterns.get('problematic_brands', {})}
Problematic Content Types: {anti_patterns.get('problematic_content_types', {})}
Problematic Durations: {anti_patterns.get('problematic_durations', {})}
Common Issues: {anti_patterns.get('common_characteristics', [])}
"""
    
    def _format_improvement_actions(self, actions: List) -> str:
        return "\n".join([f"- {action}" for action in actions]) if actions else "No improvement actions available"
    
    def _format_avoidance_insights(self, insights: List) -> str:
        return "\n".join([f"- {insight}" for insight in insights]) if insights else "No avoidance insights available"
    
    def _format_detailed_brand_performance(self, brand_data: Dict) -> str:
        rankings = brand_data.get('brand_rankings', {})
        return f"""
Top Engagement: {rankings.get('top_by_engagement_rate', [])}
Top Completion: {rankings.get('top_by_completion_rate', [])}
Most Active: {rankings.get('most_active_brands', [])}
"""
    
    def _format_detailed_content_performance(self, content_data: Dict) -> str:
        rankings = content_data.get('content_rankings', {})
        return f"""
Highest Engagement: {rankings.get('highest_engagement_content', [])}
Best Completion: {rankings.get('highest_completion_content', [])}
Most Popular: {rankings.get('most_popular_content', [])}
"""
    
    def _format_duration_insights(self, duration_data: Dict) -> str:
        optimal = duration_data.get('optimal_duration', {})
        return f"""
Best for Engagement: {optimal.get('best_for_engagement', 'N/A')}
Best for Completion: {optimal.get('best_for_completion', 'N/A')}  
Best for Views: {optimal.get('best_for_views', 'N/A')}
"""
    
    def _format_temporal_insights(self, temporal_data: Dict) -> str:
        optimal = temporal_data.get('optimal_times', {})
        return f"""
Peak Engagement Hour: {optimal.get('peak_engagement_hour', 'N/A')}:00
Peak Engagement Day: {optimal.get('peak_engagement_day', 'N/A')}
Best Hours: {optimal.get('best_hours', [])}
Best Days: {optimal.get('best_days', [])}
"""
    
    def _format_scheduling_recommendations(self, recommendations: List) -> str:
        return "\n".join([f"- {rec}" for rec in recommendations]) if recommendations else "No scheduling recommendations"
    
    def _format_avoid_posting_times(self, avoid_data: Dict) -> str:
        return f"""
Worst Hour: {avoid_data.get('worst_hour', 'N/A')}:00
Problematic Hours: {avoid_data.get('problematic_hours', [])}
"""

    def _extract_quick_actions(self, exec_result: Dict) -> List[str]:
        actions: List[str] = []
        kpis = exec_result.get("kpi_snapshot", {})
        warning_count = kpis.get("warning_count", 0) or 0
        avg_er = kpis.get("avg_engagement_rate", 0) or 0
        top_brand = kpis.get("top_brand")
        if warning_count > 0:
            actions.append("Review warning signals and address top issues")
        if isinstance(avg_er, (int, float)) and avg_er < 3:
            actions.append("Optimize content hooks and posting windows to lift engagement rate")
        if top_brand and top_brand != 'N/A':
            actions.append(f"Scale content with top brand partner: {top_brand}")
        return actions

    def _get_alert_status(self, kpis: Dict) -> str:
        wc = kpis.get("warning_count", 0) or 0
        if wc >= 10:
            return "CRITICAL"
        if wc >= 1:
            return "WARNING"
        return "HEALTHY"

    def _save_analysis_outputs(self, result: Dict, focus: str):
        """Save analysis outputs (JSON + Markdown) to insights_dir."""
        ts = self.timestamp
        ds = result.get('dataset_id') or 'dataset'
        safe_ds = ''.join(c if (c.isalnum() or c in ('-','_')) else '-' for c in str(ds))
        json_file = self.insights_dir / f"ai_insights_{focus}_{safe_ds}_{ts}.json"
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        md_file = self.insights_dir / f"ai_insights_{focus}_{safe_ds}_{ts}.md"
        with open(md_file, 'w') as f:
            f.write(f"# TikTok AI Insights - {focus.replace('_',' ').title()}\n\n")
            f.write(f"**Generated:** {result.get('generated_at', datetime.now().isoformat())}\n\n")
            if result.get('dataset_id'):
                f.write(f"**Dataset ID:** {result['dataset_id']}\n\n")
            f.write("## Insights\n\n")
            f.write(result.get('ai_insights', 'No insights'))
            f.write("\n\n")
            if 'data_summary' in result:
                ds = result['data_summary']
                f.write("## Data Summary\n\n")
                f.write(f"- Total Posts: {ds.get('total_posts','N/A')}\n")
                f.write(f"- Avg Engagement Rate: {ds.get('avg_engagement_rate','N/A')}\n")
                f.write(f"- Top Brand: {ds.get('top_brand','N/A')}\n")
                f.write(f"- Optimal Hour: {ds.get('optimal_hour','N/A')}\n")
                f.write(f"- Warning Count: {ds.get('warning_count','N/A')}\n")
    
    async def _get_ollama_response(self, prompt: str) -> str:
        """Get response from Ollama"""
        
        import aiohttp
        
        payload = {
            "model": "llama3.2:3b",  # Fast model for metrics analysis
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_host}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "No response generated")
                    else:
                        return f"Error: HTTP {response.status}"
        
        except Exception as e:
            return f"Error connecting to Ollama: {e}"
    
    async def generate_executive_summary(self, skip_if_exists: bool = False) -> Dict[str, Any]:
        """Generate executive-level summary for business stakeholders"""
        
        metrics = self._load_all_metrics()
        summary = metrics.get("summary", {})
        
        prompt = f"""
Generate an executive summary for TikTok marketing performance:

KEY METRICS:
- Total Posts Analyzed: {summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0):,}
- Portfolio Engagement Rate: {summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0):.2f}%
- Top Performing Brand: {summary.get('quick_access', {}).get('top_brands', [{}])[0].get('brand', 'N/A') if summary.get('quick_access', {}).get('top_brands') else 'N/A'}
- Optimal Posting Time: {summary.get('quick_access', {}).get('optimal_posting', {}).get('best_hour', 'N/A')}:00 on {summary.get('quick_access', {}).get('optimal_posting', {}).get('best_day', 'N/A')}

WARNING SIGNALS: {len(summary.get('warning_signals', []))} issues detected

Provide a 3-paragraph executive summary covering:
1. Overall performance assessment
2. Key opportunities and risks
3. Strategic recommendations with business impact
"""
        
        ai_response = await self._get_ollama_response(prompt)
        
        result = {
            "executive_summary": ai_response,
            "kpi_snapshot": {
                "total_posts": summary.get('quick_access', {}).get('dataset_overview', {}).get('total_posts', 0),
                "avg_engagement_rate": summary.get('quick_access', {}).get('dataset_overview', {}).get('avg_engagement_rate', 0),
                "warning_count": len(summary.get('warning_signals', [])),
                "top_brand": summary.get('quick_access', {}).get('top_brands', [{}])[0].get('brand', 'N/A') if summary.get('quick_access', {}).get('top_brands') else 'N/A'
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Save executive outputs
        self._save_executive_outputs(result)
        
        return result
    
    def _save_executive_outputs(self, result: Dict):
        """Save executive summary in multiple formats"""
        
        timestamp = self.timestamp
        ds = result.get('dataset_id') or 'dataset'
        safe_ds = ''.join(c if (str(c).isalnum() or c in ('-','_')) else '-' for c in str(ds))
        
        # 1. Executive JSON for AI agents
        exec_json = self.insights_dir / f"tiktok_executive_summary_{safe_ds}_{timestamp}.json"
        with open(exec_json, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # 2. Executive dashboard JSON (always latest)
        dashboard_file = self.insights_dir / "tiktok_executive_dashboard.json"
        dashboard_data = {
            "dashboard_type": "executive",
            "kpis": result["kpi_snapshot"],
            "executive_summary": result["executive_summary"],
            "last_updated": datetime.now().isoformat(),
            "data_freshness": "Real-time from latest ingestion",
            "quick_actions": self._extract_quick_actions(result),
            "alert_status": self._get_alert_status(result["kpi_snapshot"])
        }
        
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        # 3. Latest executive summary for AI quick access
        latest_exec = self.insights_dir / "latest_executive_summary_tiktok.json"
        with open(latest_exec, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Saved executive outputs: {exec_json}, {dashboard_file}, {latest_exec}")
        
        # 4. Business report markdown
        self._save_executive_markdown(result, timestamp, safe_ds)
    
    def _save_executive_markdown(self, result: Dict, timestamp: str, safe_ds: str):
        """Save executive summary as business-friendly markdown"""
        
        md_file = self.insights_dir / f"tiktok_executive_report_{safe_ds}_{timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write("# TikTok Marketing Performance - Executive Report\n\n")
            f.write(f"**Generated:** {result['generated_at']}\n\n")
            if result.get('dataset_id'):
                f.write(f"**Dataset ID:** {result['dataset_id']}\n\n")
            
            f.write("## Key Performance Indicators\n\n")
            kpis = result['kpi_snapshot']
            f.write(f"- **Total Posts Analyzed:** {kpis['total_posts']:,}\n")
            f.write(f"- **Average Engagement Rate:** {kpis['avg_engagement_rate']:.2f}%\n")
            f.write(f"- **Top Performing Brand:** {kpis['top_brand']}\n")
            f.write(f"- **Warning Signals:** {kpis['warning_count']} issues detected\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(result['executive_summary'])
            f.write("\n\n")
            
            f.write("## Alert Status\n\n")
            alert_status = self._get_alert_status(kpis)
            f.write(f"**Status:** {alert_status}\n\n")
            
            if alert_status != "HEALTHY":
                f.write("### Recommended Actions\n")
                actions = self._extract_quick_actions(result)
                for action in actions:
                    f.write(f"- {action}\n")
        
        print(f"Saved executive markdown: {md_file}")

# Simple usage function
async def analyze_tiktok_performance(focus_area: str = "comprehensive") -> Dict[str, Any]:
    """Simple function to analyze TikTok performance"""
    
    agent = TikTokAnalyticsAgent()
    return await agent.analyze_performance(focus_area)

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TikTok Analytics Agent')
    parser.add_argument('--focus', choices=['comprehensive', 'performance_optimization', 'content_strategy', 'posting_optimization'], 
                       default='comprehensive', help='Analysis focus area')
    parser.add_argument('--executive', action='store_true', help='Generate executive summary only')
    parser.add_argument('--skip-if-exists', action='store_true', help='Skip generation if a report for the dataset already exists')
    parser.add_argument('--all', action='store_true', help='Run all analyses including executive')
    parser.add_argument('--index-rag', action='store_true', help='After generation, index outputs into the RAG')
    
    args = parser.parse_args()
    
    async def main():
        agent = TikTokAnalyticsAgent()
        
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
                # Always generate executive summary when --all is used
                results.append(await agent.generate_executive_summary(skip_if_exists=args.skip_if_exists))
                pbar.update(1)
            # Print a short summary of what ran/skipped
            print("\nGenerated the following outputs:")
            for r in results:
                if r.get('skipped'):
                    print(f"- {r.get('focus','executive')}: skipped (exists) → {r.get('existing_file')}")
                else:
                    print(f"- {r.get('analysis_type','executive')}: saved for dataset {r.get('dataset_id')}")

            # Optional: Index into RAG immediately after generation
            if args.index_rag:
                dataset_id = None
                for r in results:
                    if r.get('dataset_id'):
                        dataset_id = r['dataset_id']
                        break
                if dataset_id:
                    import subprocess, sys
                    cmd = [
                        sys.executable,
                        'rag/indexer.py',
                        '--platform', 'tiktok',
                        '--dataset-id', str(dataset_id),
                        '--metrics-dir', './metrics/tiktok',
                        '--insights-dir', './insights/tiktok',
                    ]
                    print(f"\nIndexing into RAG for dataset_id={dataset_id}...")
                    try:
                        with tqdm(total=1, desc="RAG Index", unit="step") as pbar:
                            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
                            pbar.update(1)
                        if res.stdout:
                            print(res.stdout.strip())
                        if res.stderr:
                            print(res.stderr.strip())
                    except Exception as e:
                        print(f"RAG indexing failed: {e}")
            return

        if args.executive:
            result = await agent.generate_executive_summary(skip_if_exists=args.skip_if_exists)
            print("\n" + "="*50)
            print("EXECUTIVE SUMMARY")
            print("="*50)
            print(result["executive_summary"])
            print(f"\nKPIs: {result['kpi_snapshot']}")
        else:
            result = await agent.analyze_performance(args.focus, skip_if_exists=args.skip_if_exists)
            print("\n" + "="*50)
            print(f"TIKTOK ANALYSIS - {args.focus.upper()}")
            print("="*50)
            print(result["ai_insights"])
            
            if "recommendations" in result:
                print("\nRECOMMENDATIONS:")
                for category, recs in result["recommendations"].items():
                    if recs:
                        print(f"\n{category.upper()}:")
                        for rec in recs:
                            print(f"  - {rec}")
    
    asyncio.run(main())