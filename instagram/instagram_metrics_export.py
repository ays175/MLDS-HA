#!/usr/bin/env python3
"""
Instagram metrics export system for agentic AI consumption
Saves to ./metrics/instagram and prefixes filenames with 'instagram_'
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class InstagramMetricsExporter:
    """Export Instagram metrics in AI-agent-friendly formats"""

    def __init__(self, output_dir: str = "./metrics/instagram"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_all_metrics(self, df: pd.DataFrame, entities: Dict, dataset_id: str):
        """Export comprehensive metrics suite for AI agents"""

        print("📊 Generating Instagram AI-agent metrics...")

        # Normalize infinities to NaN once up front
        df = df.replace([np.inf, -np.inf], np.nan)

        # 1. Dataset Overview Metrics
        overview = self.generate_dataset_overview(df, dataset_id)
        self.save_json(overview, f"instagram_dataset_overview_{self.timestamp}.json")
        flat_overview = {
            'dataset_id': overview.get('dataset_id'),
            'generated_at': overview.get('generated_at'),
            'total_posts': overview.get('total_posts'),
            'start_date': (overview.get('date_range') or {}).get('start_date'),
            'end_date': (overview.get('date_range') or {}).get('end_date'),
            'total_days': (overview.get('date_range') or {}).get('total_days'),
            'total_impressions': (overview.get('key_metrics') or {}).get('total_impressions'),
            'total_engagements': (overview.get('key_metrics') or {}).get('total_engagements'),
            'total_video_views': (overview.get('key_metrics') or {}).get('total_video_views'),
            'average_engagement_rate': (overview.get('key_metrics') or {}).get('average_engagement_rate'),
            'average_story_completion_rate': (overview.get('key_metrics') or {}).get('average_story_completion_rate'),
        }
        self.save_csv([flat_overview], f"instagram_dataset_overview_{self.timestamp}.csv")

        # 1b. Per-post sampled metrics for distributions (max 5000 rows)
        try:
            sample_cols = ['instagram_insights_engagement','instagram_insights_video_views','instagram_insights_impressions','instagram_insights_story_completion_rate']
            sample_df = df[sample_cols].dropna(how='all')
            n = min(5000, len(sample_df))
            if n > 0:
                sample_out = sample_df.sample(n=n, random_state=42).to_dict('records')
                self.save_json({'per_post_sample': sample_out}, f"instagram_per_post_sample_{self.timestamp}.json")
        except Exception:
            pass

        # 2. Performance Metrics by Brand
        brand_metrics = self.generate_brand_performance_metrics(df)
        self.save_json(brand_metrics, f"instagram_brand_performance_{self.timestamp}.json")
        self.save_csv(brand_metrics.get('brand_summary', []), f"instagram_brand_performance_{self.timestamp}.csv")

        # 3. Content Type Performance
        content_metrics = self.generate_content_type_metrics(df)
        self.save_json(content_metrics, f"instagram_content_type_performance_{self.timestamp}.json")
        self.save_csv(content_metrics.get('content_summary', []), f"instagram_content_type_performance_{self.timestamp}.csv")

        # 4. Temporal Performance Analytics
        temporal_metrics = self.generate_temporal_metrics(df)
        self.save_json(temporal_metrics, f"instagram_temporal_analytics_{self.timestamp}.json")
        self.save_csv(temporal_metrics.get('hourly_performance', []), f"instagram_hourly_performance_{self.timestamp}.csv")
        self.save_csv(temporal_metrics.get('daily_performance', []), f"instagram_daily_performance_{self.timestamp}.csv")
        self.save_json({'hourly_performance': temporal_metrics.get('hourly_performance', [])}, f"instagram_hourly_performance_{self.timestamp}.json")
        self.save_json({'daily_performance': temporal_metrics.get('daily_performance', [])}, f"instagram_daily_performance_{self.timestamp}.json")

        # 5. Top Performers Analysis
        top_performers = self.generate_top_performers_analysis(df)
        self.save_json(top_performers, f"instagram_top_performers_{self.timestamp}.json")
        self.save_csv(top_performers.get('top_posts', []), f"instagram_top_performers_{self.timestamp}.csv")

        # 6. Worst Performers Analysis
        worst_performers = self.generate_worst_performers_analysis(df)
        self.save_json(worst_performers, f"instagram_worst_performers_{self.timestamp}.json")
        self.save_csv(worst_performers.get('worst_posts', []), f"instagram_worst_performers_{self.timestamp}.csv")

        # 7. AI Agent Instructions and Query Examples
        ai_guide = self.generate_ai_agent_guide(overview, brand_metrics, content_metrics, temporal_metrics)
        self.save_json(ai_guide, f"instagram_ai_agent_guide_{self.timestamp}.json")

        # 8. Consolidated Metrics Summary for Quick AI Access
        consolidated = self.generate_consolidated_summary(overview, brand_metrics, content_metrics, temporal_metrics, top_performers, worst_performers)
        self.save_json(consolidated, "latest_metrics_summary_instagram.json")

        print(f"✅ All Instagram metrics exported to {self.output_dir}")
        return consolidated

    def generate_dataset_overview(self, df: pd.DataFrame, dataset_id: str) -> Dict:
        df = df.copy()
        numeric_cols = ['instagram_insights_impressions', 'instagram_insights_video_views', 'instagram_insights_engagement', 'instagram_insights_reach']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

        with pd.option_context('mode.use_inf_as_na', True):
            er_series = (df['instagram_insights_engagement'] / df['instagram_insights_impressions'] * 100).replace([pd.NA, pd.NaT], 0).fillna(0)
        scr_series = pd.to_numeric(df.get('instagram_insights_story_completion_rate', 0), errors='coerce').fillna(0)

        overview = {
            "dataset_id": dataset_id,
            "generated_at": datetime.now().isoformat(),
            "total_posts": len(df),
            "date_range": self._get_date_range(df),
            "key_metrics": {
                "total_impressions": int(df['instagram_insights_impressions'].sum()),
                "total_engagements": int(df['instagram_insights_engagement'].sum()),
                "total_video_views": int(df['instagram_insights_video_views'].sum()),
                "average_engagement_rate": round((df['instagram_insights_engagement'].sum() / df['instagram_insights_impressions'].sum() * 100), 2) if df['instagram_insights_impressions'].sum() > 0 else 0,
                "average_story_completion_rate": round(float(df['instagram_insights_story_completion_rate'].mean() or 0), 2),
                "max_engagements_post": int(df['instagram_insights_engagement'].max() or 0),
                "max_views_post": int(df['instagram_insights_video_views'].max() or 0),
                "max_impressions_post": int(df['instagram_insights_impressions'].max() or 0)
            },
            "performance_benchmarks": {
                "high_engagement_threshold": float(df['instagram_insights_engagement'].quantile(0.8)) if len(df) else 0.0,
                "high_story_completion_threshold": float(df['instagram_insights_story_completion_rate'].quantile(0.8)) if len(df) else 0.0,
                "viral_threshold": float(df['instagram_insights_video_views'].quantile(0.95)) if len(df) else 0.0
            },
            "quality_metrics": {
                "engagement_rate_median": round(float(er_series.quantile(0.5) if len(er_series) else 0.0), 2),
                "engagement_rate_p75": round(float(er_series.quantile(0.75) if len(er_series) else 0.0), 2),
                "engagement_rate_p95": round(float(er_series.quantile(0.95) if len(er_series) else 0.0), 2),
                "story_completion_rate_median": round(float(scr_series.quantile(0.5) if len(scr_series) else 0.0), 2),
                "story_completion_rate_p75": round(float(scr_series.quantile(0.75) if len(scr_series) else 0.0), 2),
                "story_completion_rate_p95": round(float(scr_series.quantile(0.95) if len(scr_series) else 0.0), 2)
            }
        }

        return overview

    def generate_brand_performance_metrics(self, df: pd.DataFrame) -> Dict:
        def _to_int(x: Any) -> int:
            try:
                if x is None:
                    return 0
                v = float(x)
                if pd.isna(v):
                    return 0
                return int(v)
            except Exception:
                return 0

        def _to_float(x: Any) -> float:
            try:
                if x is None:
                    return 0.0
                v = float(x)
                if pd.isna(v):
                    return 0.0
                return float(v)
            except Exception:
                return 0.0

        brand_data = []
        for _, row in df.iterrows():
            labels = str(row.get('instagram_post_labels_names', ''))
            brands = self._extract_brands_from_labels(labels)
            for brand in brands:
                brand_data.append({
                    'post_id': row.get('instagram_id'),
                    'brand': brand,
                    'impressions': _to_int(row.get('instagram_insights_impressions', 0)),
                    'engagements': _to_int(row.get('instagram_insights_engagement', 0)),
                    'video_views': _to_int(row.get('instagram_insights_video_views', 0)),
                    'story_completion_rate': _to_float(row.get('instagram_insights_story_completion_rate', 0)),
                    'posted_date': row.get('created_time', '')
                })

        brand_df = pd.DataFrame(brand_data)
        if len(brand_df) == 0:
            return {"brand_summary": [], "brand_rankings": [], "ai_insights": []}

        brand_summary = brand_df.groupby('brand').agg({
            'post_id': 'count',
            'impressions': ['sum', 'mean'],
            'engagements': ['sum', 'mean'],
            'video_views': ['sum', 'mean'],
            'story_completion_rate': 'mean'
        }).round(2)

        brand_summary.columns = ['total_posts', 'total_impressions', 'avg_impressions',
                                 'total_engagements', 'avg_engagements', 'total_views', 'avg_views', 'avg_story_completion_rate']
        brand_summary['avg_engagement_rate'] = (brand_summary['avg_engagements'] / brand_summary['avg_impressions'] * 100).replace([pd.NA], 0).fillna(0).round(2)
        brand_summary = brand_summary.reset_index()

        brand_rankings = {
            "top_by_engagement_rate": brand_summary.nlargest(5, 'avg_engagement_rate')[['brand', 'avg_engagement_rate']].to_dict('records'),
            "top_by_story_completion": brand_summary.nlargest(5, 'avg_story_completion_rate')[['brand', 'avg_story_completion_rate']].to_dict('records'),
            "top_by_total_views": brand_summary.nlargest(5, 'total_views')[['brand', 'total_views']].to_dict('records'),
            "most_active_brands": brand_summary.nlargest(5, 'total_posts')[['brand', 'total_posts']].to_dict('records')
        }

        return {
            "brand_summary": brand_summary.to_dict('records'),
            "brand_rankings": brand_rankings,
            "ai_insights": self._generate_brand_ai_insights(brand_summary),
            "generated_at": datetime.now().isoformat()
        }

    def generate_content_type_metrics(self, df: pd.DataFrame) -> Dict:
        def _to_int(x: Any) -> int:
            try:
                if x is None:
                    return 0
                v = float(x)
                if pd.isna(v):
                    return 0
                return int(v)
            except Exception:
                return 0

        def _to_float(x: Any) -> float:
            try:
                if x is None:
                    return 0.0
                v = float(x)
                if pd.isna(v):
                    return 0.0
                return float(v)
            except Exception:
                return 0.0

        content_data = []
        for _, row in df.iterrows():
            labels = str(row.get('instagram_post_labels_names', ''))
            content_types = self._extract_content_types_from_labels(labels)
            for ct in content_types:
                content_data.append({
                    'post_id': row.get('instagram_id'),
                    'content_type': ct['name'],
                    'category': ct['category'],
                    'impressions': _to_int(row.get('instagram_insights_impressions', 0)),
                    'engagements': _to_int(row.get('instagram_insights_engagement', 0)),
                    'video_views': _to_int(row.get('instagram_insights_video_views', 0)),
                    'story_completion_rate': _to_float(row.get('instagram_insights_story_completion_rate', 0))
                })

        content_df = pd.DataFrame(content_data)
        if len(content_df) == 0:
            return {"content_summary": [], "content_rankings": [], "ai_recommendations": []}

        content_summary = content_df.groupby(['content_type', 'category']).agg({
            'post_id': 'count',
            'impressions': ['sum', 'mean'],
            'engagements': ['sum', 'mean'],
            'video_views': ['sum', 'mean'],
            'story_completion_rate': 'mean'
        }).round(2)

        content_summary.columns = ['total_posts', 'total_impressions', 'avg_impressions',
                                   'total_engagements', 'avg_engagements', 'total_views',
                                   'avg_views', 'avg_story_completion_rate']
        content_summary['avg_engagement_rate'] = (content_summary['avg_engagements'] / content_summary['avg_impressions'] * 100).replace([pd.NA], 0).fillna(0).round(2)
        content_summary = content_summary.reset_index()

        content_rankings = {
            "highest_engagement_content": content_summary.nlargest(5, 'avg_engagement_rate')[['content_type', 'category', 'avg_engagement_rate']].to_dict('records'),
            "highest_story_completion": content_summary.nlargest(5, 'avg_story_completion_rate')[['content_type', 'category', 'avg_story_completion_rate']].to_dict('records'),
            "most_popular_content": content_summary.nlargest(5, 'total_posts')[['content_type', 'category', 'total_posts']].to_dict('records')
        }

        return {
            "content_summary": content_summary.to_dict('records'),
            "content_rankings": content_rankings,
            "ai_recommendations": self._generate_content_ai_recommendations(content_summary),
            "generated_at": datetime.now().isoformat()
        }

    def generate_temporal_metrics(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce')
        df['hour'] = df['created_time'].dt.hour
        df['day_of_week'] = df['created_time'].dt.day_name()

        hourly_perf = df.groupby('hour').agg({
            'instagram_insights_engagement': 'mean',
            'instagram_insights_video_views': 'mean',
            'instagram_insights_story_completion_rate': 'mean',
            'instagram_id': 'count'
        }).round(2)
        hourly_perf.columns = ['avg_engagements', 'avg_views', 'avg_story_completion_rate', 'post_count']
        hourly_perf['hour'] = hourly_perf.index
        hourly_perf = hourly_perf.reset_index(drop=True)

        daily_perf = df.groupby('day_of_week').agg({
            'instagram_insights_engagement': 'mean',
            'instagram_insights_video_views': 'mean',
            'instagram_insights_story_completion_rate': 'mean',
            'instagram_id': 'count'
        }).round(2)
        daily_perf.columns = ['avg_engagements', 'avg_views', 'avg_story_completion_rate', 'post_count']
        daily_perf['day_of_week'] = daily_perf.index
        daily_perf = daily_perf.reset_index(drop=True)

        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=ordered_days, ordered=True)
        hour_day = df.groupby(['day_of_week', 'hour']).agg({
            'instagram_insights_engagement': 'mean',
            'instagram_insights_video_views': 'mean'
        }).round(2).reset_index()
        hour_by_day = [
            {
                'day_of_week': str(row['day_of_week']),
                'hour': int(row['hour']),
                'avg_engagements': float(row['instagram_insights_engagement']) if pd.notna(row['instagram_insights_engagement']) else 0.0,
                'avg_views': float(row['instagram_insights_video_views']) if pd.notna(row['instagram_insights_video_views']) else 0.0,
            }
            for _, row in hour_day.iterrows()
        ]

        optimal_times = {
            "best_hours": hourly_perf.nlargest(3, 'avg_engagements')[['hour', 'avg_engagements']].to_dict('records') if len(hourly_perf) else [],
            "best_days": daily_perf.nlargest(3, 'avg_engagements')[['day_of_week', 'avg_engagements']].to_dict('records') if len(daily_perf) else [],
        }

        return {
            "hourly_performance": hourly_perf.to_dict('records'),
            "daily_performance": daily_perf.to_dict('records'),
            "hour_by_day": hour_by_day,
            "optimal_times": optimal_times,
            "generated_at": datetime.now().isoformat()
        }

    def generate_top_performers_analysis(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df['engagement_rate'] = (pd.to_numeric(df['instagram_insights_engagement'], errors='coerce') /
                                  pd.to_numeric(df['instagram_insights_impressions'], errors='coerce') * 100).round(2)
        top_threshold = df['engagement_rate'].quantile(0.9) if len(df) else 0
        top_posts = df[df['engagement_rate'] >= top_threshold].copy()
        return {
            "top_posts": top_posts[['instagram_id', 'instagram_post_labels_names', 'engagement_rate', 'instagram_insights_story_completion_rate', 'instagram_insights_engagement']].to_dict('records'),
        }

    def generate_worst_performers_analysis(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df['engagement_rate'] = (pd.to_numeric(df['instagram_insights_engagement'], errors='coerce') /
                                  pd.to_numeric(df['instagram_insights_impressions'], errors='coerce') * 100).round(2)
        bottom_threshold = df['engagement_rate'].quantile(0.1) if len(df) else 0
        worst_posts = df[df['engagement_rate'] <= bottom_threshold].copy()
        return {
            "worst_posts": worst_posts[['instagram_id', 'instagram_post_labels_names', 'engagement_rate', 'instagram_insights_story_completion_rate', 'created_time', 'instagram_insights_engagement']].to_dict('records'),
        }

    def generate_ai_agent_guide(self, overview: Dict, brand_metrics: Dict, content_metrics: Dict, temporal_metrics: Dict) -> Dict:
        return {
            "ai_agent_instructions": {
                "overview": "This metrics suite provides comprehensive Instagram performance analytics for data-driven content strategy",
                "key_files": {
                    "latest_metrics_summary_instagram.json": "Always up-to-date consolidated metrics - START HERE",
                    "instagram_brand_performance_*.json": "Detailed brand analytics and rankings",
                    "instagram_temporal_analytics_*.json": "Posting time optimization data",
                    "instagram_top_performers_*.json": "High-performing content patterns"
                }
            }
        }

    def generate_consolidated_summary(self, overview: Dict, brand_metrics: Dict, content_metrics: Dict, temporal_metrics: Dict, top_performers: Dict, worst_performers: Dict) -> Dict:
        return {
            "dataset_id": overview.get("dataset_id"),
            "quick_access": {
                "dataset_overview": {
                    "total_posts": overview.get("total_posts"),
                    "avg_engagement_rate": (overview.get("key_metrics") or {}).get("average_engagement_rate"),
                    "viral_threshold": (overview.get("performance_benchmarks") or {}).get("viral_threshold")
                },
                "top_performing_posts": (top_performers or {}).get("top_posts", [])[:5],
                "worst_performing_posts": (worst_performers or {}).get("worst_posts", [])[:5]
            },
            "last_updated": datetime.now().isoformat(),
            "files_generated": {
                "brand_performance": f"instagram_brand_performance_{self.timestamp}.json",
                "content_performance": f"instagram_content_type_performance_{self.timestamp}.json",
                "temporal_analytics": f"instagram_temporal_analytics_{self.timestamp}.json",
                "top_performers": f"instagram_top_performers_{self.timestamp}.json",
                "worst_performers": f"instagram_worst_performers_{self.timestamp}.json",
                "ai_guide": f"instagram_ai_agent_guide_{self.timestamp}.json"
            }
        }

    def _extract_brands_from_labels(self, labels_str: str) -> List[str]:
        brands: List[str] = []
        if labels_str and pd.notna(labels_str):
            parts = [part.strip() for part in labels_str.split(',')]
            for part in parts:
                if '[Brand]' in part:
                    brand = part.replace('[Brand]', '').strip()
                    if brand:
                        brands.append(brand)
        return brands

    def _extract_content_types_from_labels(self, labels_str: str) -> List[Dict]:
        content_types: List[Dict] = []
        if labels_str and pd.notna(labels_str):
            parts = [part.strip() for part in labels_str.split(',')]
            for part in parts:
                if '[Axis]' in part:
                    name = part.replace('[Axis]', '').strip()
                    if name:
                        content_types.append({'name': name, 'category': 'axis'})
                elif '[Asset]' in part:
                    name = part.replace('[Asset]', '').strip()
                    if name:
                        content_types.append({'name': name, 'category': 'asset'})
                elif '[Axis & Offer]' in part:
                    name = part.replace('[Axis & Offer]', '').strip()
                    if name:
                        content_types.append({'name': name, 'category': 'axis_offer'})
        return content_types

    def _get_date_range(self, df: pd.DataFrame) -> Dict:
        try:
            dates = pd.to_datetime(df['created_time'], errors='coerce').dropna()
            if len(dates) > 0:
                return {
                    "start_date": dates.min().isoformat(),
                    "end_date": dates.max().isoformat(),
                    "total_days": (dates.max() - dates.min()).days
                }
        except Exception:
            pass
        return {"start_date": None, "end_date": None, "total_days": 0}

    def _generate_brand_ai_insights(self, brand_summary: pd.DataFrame) -> List[str]:
        insights: List[str] = []
        if len(brand_summary) > 0:
            top_brand = brand_summary.loc[brand_summary['avg_engagement_rate'].idxmax()]
            insights.append(f"'{top_brand['brand']}' achieves highest engagement rate at {top_brand['avg_engagement_rate']}%")
        return insights

    def _generate_content_ai_recommendations(self, content_summary: pd.DataFrame) -> List[str]:
        recommendations: List[str] = []
        if len(content_summary) > 0:
            top_content = content_summary.loc[content_summary['avg_engagement_rate'].idxmax()]
            recommendations.append(f"Focus on '{top_content['content_type']}' content type - achieves {top_content['avg_engagement_rate']}% engagement rate")
        return recommendations

    def save_json(self, data: Dict, filename: str):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Saved: {filepath}")

    def save_csv(self, data: List[Dict], filename: str):
        if data:
            filepath = self.output_dir / filename
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"💾 Saved: {filepath}")


