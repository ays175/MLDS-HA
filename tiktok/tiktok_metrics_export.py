#!/usr/bin/env python3
"""
TikTok metrics export system for agentic AI consumption
Generates structured metrics files optimized for AI agent access
"""
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

class TikTokMetricsExporter:
    """Export TikTok metrics in AI-agent-friendly formats"""
    
    def __init__(self, output_dir="./metrics/tiktok"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_all_metrics(self, df: pd.DataFrame, entities: Dict, dataset_id: str):
        """Export comprehensive metrics suite for AI agents"""
        
        print("📊 Generating AI-agent metrics...")
        
        # 1. Dataset Overview Metrics
        overview = self.generate_dataset_overview(df, dataset_id)
        self.save_json(overview, f"tiktok_dataset_overview_{self.timestamp}.json")
        # Also save a flattened 1-row CSV for convenience
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
            'average_completion_rate': (overview.get('key_metrics') or {}).get('average_completion_rate'),
            'average_duration': (overview.get('key_metrics') or {}).get('average_duration'),
            'viral_threshold': (overview.get('performance_benchmarks') or {}).get('viral_threshold'),
            'high_engagement_threshold': (overview.get('performance_benchmarks') or {}).get('high_engagement_threshold'),
            'high_completion_threshold': (overview.get('performance_benchmarks') or {}).get('high_completion_threshold'),
        }
        self.save_csv([flat_overview], f"tiktok_dataset_overview_{self.timestamp}.csv")

        # 1b. Per-post sampled metrics for distributions (max 5000 rows)
        try:
            sample_cols = ['tiktok_insights_engagements','tiktok_insights_video_views','tiktok_insights_impressions','tiktok_insights_completion_rate','tiktok_duration']
            sample_df = df[sample_cols].dropna(how='all')
            n = min(5000, len(sample_df))
            if n > 0:
                sample_out = sample_df.sample(n=n, random_state=42).to_dict('records')
                self.save_json({ 'per_post_sample': sample_out }, f"tiktok_per_post_sample_{self.timestamp}.json")
        except Exception:
            pass
        
        # 2. Performance Metrics by Brand
        brand_metrics = self.generate_brand_performance_metrics(df)
        self.save_json(brand_metrics, f"tiktok_brand_performance_{self.timestamp}.json")
        self.save_csv(brand_metrics['brand_summary'], f"tiktok_brand_performance_{self.timestamp}.csv")
        
        # 3. Content Type Performance 
        content_metrics = self.generate_content_type_metrics(df)
        self.save_json(content_metrics, f"tiktok_content_type_performance_{self.timestamp}.json")
        self.save_csv(content_metrics['content_summary'], f"tiktok_content_type_performance_{self.timestamp}.csv")
        
        # 4. Temporal Performance Analytics
        temporal_metrics = self.generate_temporal_metrics(df)
        self.save_json(temporal_metrics, f"tiktok_temporal_analytics_{self.timestamp}.json")
        self.save_csv(temporal_metrics['hourly_performance'], f"tiktok_hourly_performance_{self.timestamp}.csv")
        self.save_csv(temporal_metrics['daily_performance'], f"tiktok_daily_performance_{self.timestamp}.csv")
        # Also save hourly/daily as JSON files
        self.save_json({'hourly_performance': temporal_metrics['hourly_performance']}, f"tiktok_hourly_performance_{self.timestamp}.json")
        self.save_json({'daily_performance': temporal_metrics['daily_performance']}, f"tiktok_daily_performance_{self.timestamp}.json")
        
        # 5. Video Duration Performance
        duration_metrics = self.generate_duration_performance_metrics(df)
        self.save_json(duration_metrics, f"tiktok_duration_performance_{self.timestamp}.json")
        self.save_csv(duration_metrics['duration_summary'], f"tiktok_duration_performance_{self.timestamp}.csv")
        
        # 6. Top Performers Analysis
        top_performers = self.generate_top_performers_analysis(df)
        self.save_json(top_performers, f"tiktok_top_performers_{self.timestamp}.json")
        self.save_csv(top_performers['top_posts'], f"tiktok_top_performers_{self.timestamp}.csv")
        
        # 7. Worst Performers Analysis - NEW
        worst_performers = self.generate_worst_performers_analysis(df)
        self.save_json(worst_performers, f"tiktok_worst_performers_{self.timestamp}.json")
        self.save_csv(worst_performers['worst_posts'], f"tiktok_worst_performers_{self.timestamp}.csv")
        
        # 8. AI Agent Instructions and Query Examples
        ai_guide = self.generate_ai_agent_guide(overview, brand_metrics, content_metrics, temporal_metrics)
        self.save_json(ai_guide, f"tiktok_ai_agent_guide_{self.timestamp}.json")
        
        # 9. Consolidated Metrics Summary for Quick AI Access
        consolidated = self.generate_consolidated_summary(overview, brand_metrics, content_metrics, temporal_metrics, duration_metrics, top_performers, worst_performers)
        self.save_json(consolidated, "latest_metrics_summary_tiktok.json")  # Always overwrite for latest access
        
        print(f"✅ All metrics exported to {self.output_dir}")
        return consolidated
    
    def generate_dataset_overview(self, df: pd.DataFrame, dataset_id: str) -> Dict:
        """Generate high-level dataset overview metrics"""
        
        # Convert numeric columns safely
        numeric_cols = ['tiktok_insights_impressions', 'tiktok_insights_video_views', 'tiktok_insights_engagements', 
                       'tiktok_insights_likes', 'tiktok_insights_shares', 'tiktok_insights_reach', 'tiktok_duration']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Per-post engagement rate (%) and completion rate
        with pd.option_context('mode.use_inf_as_na', True):
            er_series = (df['tiktok_insights_engagements'] / df['tiktok_insights_impressions'] * 100).replace([pd.NA, pd.NaT], 0).fillna(0)
        cr_series = pd.to_numeric(df.get('tiktok_insights_completion_rate', 0), errors='coerce').fillna(0)
        
        # Quality quantiles
        er_median = float(er_series.quantile(0.5)) if len(er_series) else 0.0
        er_p75 = float(er_series.quantile(0.75)) if len(er_series) else 0.0
        er_p95 = float(er_series.quantile(0.95)) if len(er_series) else 0.0
        cr_median = float(cr_series.quantile(0.5)) if len(cr_series) else 0.0
        cr_p75 = float(cr_series.quantile(0.75)) if len(cr_series) else 0.0
        cr_p95 = float(cr_series.quantile(0.95)) if len(cr_series) else 0.0
        
        overview = {
            "dataset_id": dataset_id,
            "generated_at": datetime.now().isoformat(),
            "total_posts": len(df),
            "date_range": self._get_date_range(df),
            "key_metrics": {
                "total_impressions": int(df['tiktok_insights_impressions'].sum()),
                "total_engagements": int(df['tiktok_insights_engagements'].sum()),
                "total_video_views": int(df['tiktok_insights_video_views'].sum()),
                "average_engagement_rate": round((df['tiktok_insights_engagements'].sum() / df['tiktok_insights_impressions'].sum() * 100), 2) if df['tiktok_insights_impressions'].sum() > 0 else 0,
                "average_completion_rate": round(df['tiktok_insights_completion_rate'].mean(), 2),
                "average_duration": round(df['tiktok_duration'].mean(), 1),
                "max_engagements_post": int(df['tiktok_insights_engagements'].max()),
                "max_views_post": int(df['tiktok_insights_video_views'].max()),
                "max_impressions_post": int(df['tiktok_insights_impressions'].max())
            },
            "performance_benchmarks": {
                "high_engagement_threshold": df['tiktok_insights_engagements'].quantile(0.8),
                "high_completion_threshold": df['tiktok_insights_completion_rate'].quantile(0.8),
                "viral_threshold": df['tiktok_insights_video_views'].quantile(0.95)
            },
            "quality_metrics": {
                "engagement_rate_median": round(er_median, 2),
                "engagement_rate_p75": round(er_p75, 2),
                "engagement_rate_p95": round(er_p95, 2),
                "completion_rate_median": round(float(cr_median), 4),
                "completion_rate_p75": round(float(cr_p75), 4),
                "completion_rate_p95": round(float(cr_p95), 4)
            }
        }

        # Benchmarks attainment
        try:
            he_thr = overview["performance_benchmarks"]["high_engagement_threshold"] or 0
            hc_thr = overview["performance_benchmarks"]["high_completion_threshold"] or 0
            pct_he = float((df['tiktok_insights_engagements'] > he_thr).mean() * 100) if len(df) else 0.0
            pct_hc = float((df['tiktok_insights_completion_rate'] > hc_thr).mean() * 100) if len(df) else 0.0
            overview["benchmark_attainment"] = {
                "pct_posts_above_high_engagement": round(pct_he, 2),
                "pct_posts_above_high_completion": round(pct_hc, 2)
            }
        except Exception:
            pass
        
        return overview
    
    def generate_brand_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Generate detailed brand performance analytics"""
        
        brand_data = []
        
        for _, row in df.iterrows():
            labels = str(row.get('tiktok_post_labels_names', ''))
            brands = self._extract_brands_from_labels(labels)
            
            for brand in brands:
                brand_data.append({
                    'post_id': row.get('tiktok_id'),
                    'brand': brand,
                    'impressions': int(row.get('tiktok_insights_impressions', 0) or 0),
                    'engagements': int(row.get('tiktok_insights_engagements', 0) or 0),
                    'video_views': int(row.get('tiktok_insights_video_views', 0) or 0),
                    'completion_rate': float(row.get('tiktok_insights_completion_rate', 0) or 0),
                    'duration': float(row.get('tiktok_duration', 0) or 0),
                    'posted_date': row.get('created_time', '')
                })
        
        brand_df = pd.DataFrame(brand_data)
        
        if len(brand_df) == 0:
            return {"brand_summary": [], "brand_rankings": [], "ai_insights": []}
        
        # Calculate brand performance summary
        brand_summary = brand_df.groupby('brand').agg({
            'post_id': 'count',
            'impressions': ['sum', 'mean'],
            'engagements': ['sum', 'mean'],
            'video_views': ['sum', 'mean'],
            'completion_rate': 'mean',
            'duration': 'mean'
        }).round(2)
        
        brand_summary.columns = ['total_posts', 'total_impressions', 'avg_impressions', 
                                'total_engagements', 'avg_engagements', 'total_views', 
                                'avg_views', 'avg_completion_rate', 'avg_duration']
        
        # Calculate engagement rate
        brand_summary['avg_engagement_rate'] = (brand_summary['avg_engagements'] / brand_summary['avg_impressions'] * 100).round(2)
        
        brand_summary = brand_summary.reset_index()
        
        # Brand rankings for AI decision making
        brand_rankings = {
            "top_by_engagement_rate": brand_summary.nlargest(5, 'avg_engagement_rate')[['brand', 'avg_engagement_rate']].to_dict('records'),
            "top_by_completion_rate": brand_summary.nlargest(5, 'avg_completion_rate')[['brand', 'avg_completion_rate']].to_dict('records'),
            "top_by_total_views": brand_summary.nlargest(5, 'total_views')[['brand', 'total_views']].to_dict('records'),
            "most_active_brands": brand_summary.nlargest(5, 'total_posts')[['brand', 'total_posts']].to_dict('records')
        }
        
        # AI insights
        ai_insights = self._generate_brand_ai_insights(brand_summary)
        
        return {
            "brand_summary": brand_summary.to_dict('records'),
            "brand_rankings": brand_rankings,
            "ai_insights": ai_insights,
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_content_type_metrics(self, df: pd.DataFrame) -> Dict:
        """Generate content type performance analytics"""
        
        content_data = []
        
        for _, row in df.iterrows():
            labels = str(row.get('tiktok_post_labels_names', ''))
            content_types = self._extract_content_types_from_labels(labels)
            
            for content_type in content_types:
                content_data.append({
                    'post_id': row.get('tiktok_id'),
                    'content_type': content_type['name'],
                    'category': content_type['category'],  # axis, asset, package
                    'impressions': int(row.get('tiktok_insights_impressions', 0) or 0),
                    'engagements': int(row.get('tiktok_insights_engagements', 0) or 0),
                    'video_views': int(row.get('tiktok_insights_video_views', 0) or 0),
                    'completion_rate': float(row.get('tiktok_insights_completion_rate', 0) or 0),
                    'duration': float(row.get('tiktok_duration', 0) or 0)
                })
        
        content_df = pd.DataFrame(content_data)
        
        if len(content_df) == 0:
            return {"content_summary": [], "content_rankings": [], "ai_recommendations": []}
        
        # Content type performance summary
        content_summary = content_df.groupby(['content_type', 'category']).agg({
            'post_id': 'count',
            'impressions': ['sum', 'mean'],
            'engagements': ['sum', 'mean'], 
            'video_views': ['sum', 'mean'],
            'completion_rate': 'mean',
            'duration': 'mean'
        }).round(2)
        
        content_summary.columns = ['total_posts', 'total_impressions', 'avg_impressions',
                                  'total_engagements', 'avg_engagements', 'total_views',
                                  'avg_views', 'avg_completion_rate', 'avg_duration']
        
        content_summary['avg_engagement_rate'] = (content_summary['avg_engagements'] / content_summary['avg_impressions'] * 100).round(2)
        content_summary = content_summary.reset_index()
        
        # Content rankings
        content_rankings = {
            "highest_engagement_content": content_summary.nlargest(5, 'avg_engagement_rate')[['content_type', 'category', 'avg_engagement_rate']].to_dict('records'),
            "highest_completion_content": content_summary.nlargest(5, 'avg_completion_rate')[['content_type', 'category', 'avg_completion_rate']].to_dict('records'),
            "most_popular_content": content_summary.nlargest(5, 'total_posts')[['content_type', 'category', 'total_posts']].to_dict('records')
        }
        
        # AI recommendations
        ai_recommendations = self._generate_content_ai_recommendations(content_summary)
        
        return {
            "content_summary": content_summary.to_dict('records'),
            "content_rankings": content_rankings,
            "ai_recommendations": ai_recommendations,
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_temporal_metrics(self, df: pd.DataFrame) -> Dict:
        """Generate posting time performance analytics"""
        
        # Parse dates and extract temporal features
        df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce')
        df['hour'] = df['created_time'].dt.hour
        df['day_of_week'] = df['created_time'].dt.day_name()
        df['date'] = df['created_time'].dt.date
        
        # Hourly performance
        hourly_perf = df.groupby('hour').agg({
            'tiktok_insights_engagements': 'mean',
            'tiktok_insights_video_views': 'mean',
            'tiktok_insights_completion_rate': 'mean',
            'tiktok_id': 'count'
        }).round(2)
        
        hourly_perf.columns = ['avg_engagements', 'avg_views', 'avg_completion_rate', 'post_count']
        hourly_perf['hour'] = hourly_perf.index
        hourly_perf = hourly_perf.reset_index(drop=True)
        
        # Daily performance
        daily_perf = df.groupby('day_of_week').agg({
            'tiktok_insights_engagements': 'mean',
            'tiktok_insights_video_views': 'mean', 
            'tiktok_insights_completion_rate': 'mean',
            'tiktok_id': 'count'
        }).round(2)
        
        daily_perf.columns = ['avg_engagements', 'avg_views', 'avg_completion_rate', 'post_count']
        daily_perf['day_of_week'] = daily_perf.index
        daily_perf = daily_perf.reset_index(drop=True)

        # Hour-by-day heatmap source (7x24 grid rows)
        # Ensure ordered days
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=ordered_days, ordered=True)
        hour_day = df.groupby(['day_of_week', 'hour']).agg({
            'tiktok_insights_engagements': 'mean',
            'tiktok_insights_video_views': 'mean'
        }).round(2).reset_index()
        # Flatten into list of dicts for JSON
        hour_by_day = [
            {
                'day_of_week': str(row['day_of_week']),
                'hour': int(row['hour']),
                'avg_engagements': float(row['tiktok_insights_engagements']) if pd.notna(row['tiktok_insights_engagements']) else 0.0,
                'avg_views': float(row['tiktok_insights_video_views']) if pd.notna(row['tiktok_insights_video_views']) else 0.0,
            }
            for _, row in hour_day.iterrows()
        ]
        
        # AI-optimized insights
        optimal_times = {
            "best_hours": hourly_perf.nlargest(3, 'avg_engagements')[['hour', 'avg_engagements']].to_dict('records'),
            "best_days": daily_perf.nlargest(3, 'avg_engagements')[['day_of_week', 'avg_engagements']].to_dict('records'),
            "peak_engagement_hour": int(hourly_perf.loc[hourly_perf['avg_engagements'].idxmax(), 'hour']),
            "peak_engagement_day": daily_perf.loc[daily_perf['avg_engagements'].idxmax(), 'day_of_week']
        }
        
        return {
            "hourly_performance": hourly_perf.to_dict('records'),
            "daily_performance": daily_perf.to_dict('records'),
            "hour_by_day": hour_by_day,
            "optimal_times": optimal_times,
            "ai_scheduling_recommendations": self._generate_scheduling_recommendations(hourly_perf, daily_perf),
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_duration_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Generate video duration performance analytics"""
        
        # Create duration categories
        df['duration_category'] = pd.cut(
            pd.to_numeric(df['tiktok_duration'], errors='coerce'),
            bins=[0, 15, 30, 60, float('inf')],
            labels=['Short (0-15s)', 'Medium (16-30s)', 'Long (31-60s)', 'Extended (60s+)'],
            include_lowest=True
        )
        
        # Duration performance summary
        duration_summary = df.groupby('duration_category').agg({
            'tiktok_insights_engagements': 'mean',
            'tiktok_insights_video_views': 'mean',
            'tiktok_insights_completion_rate': 'mean',
            'tiktok_id': 'count'
        }).round(2)
        
        duration_summary.columns = ['avg_engagements', 'avg_views', 'avg_completion_rate', 'post_count']
        duration_summary['duration_category'] = duration_summary.index
        duration_summary = duration_summary.reset_index(drop=True)
        
        # Calculate engagement rate by duration
        for idx, row in duration_summary.iterrows():
            category = row['duration_category']
            category_df = df[df['duration_category'] == category]
            total_impressions = category_df['tiktok_insights_impressions'].sum()
            total_engagements = category_df['tiktok_insights_engagements'].sum()
            duration_summary.loc[idx, 'avg_engagement_rate'] = round((total_engagements / total_impressions * 100), 2) if total_impressions > 0 else 0
        
        # AI optimization insights
        optimal_duration = {
            "best_for_engagement": duration_summary.loc[duration_summary['avg_engagement_rate'].idxmax(), 'duration_category'],
            "best_for_completion": duration_summary.loc[duration_summary['avg_completion_rate'].idxmax(), 'duration_category'],
            "best_for_views": duration_summary.loc[duration_summary['avg_views'].idxmax(), 'duration_category']
        }
        
        return {
            "duration_summary": duration_summary.to_dict('records'),
            "optimal_duration": optimal_duration,
            "ai_duration_recommendations": self._generate_duration_recommendations(duration_summary),
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_top_performers_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze top performing posts for pattern recognition"""
        
        # Calculate engagement rate for each post
        df['engagement_rate'] = (pd.to_numeric(df['tiktok_insights_engagements'], errors='coerce') / 
                                pd.to_numeric(df['tiktok_insights_impressions'], errors='coerce') * 100).round(2)
        
        # Get top performers (top 10% by engagement rate)
        top_threshold = df['engagement_rate'].quantile(0.9)
        top_posts = df[df['engagement_rate'] >= top_threshold].copy()
        
        # Analyze patterns in top performers
        top_analysis = {
            "top_posts": top_posts[['tiktok_id', 'tiktok_post_labels_names', 'engagement_rate', 
                                   'tiktok_insights_completion_rate', 'tiktok_duration', 'tiktok_insights_engagements']].to_dict('records'),
            "common_patterns": self._analyze_top_performer_patterns(top_posts),
            "performance_insights": self._generate_performance_insights(df, top_posts)
        }
        
        return top_analysis
    
    def generate_ai_agent_guide(self, overview: Dict, brand_metrics: Dict, content_metrics: Dict, temporal_metrics: Dict) -> Dict:
        """Generate comprehensive guide for AI agents to use these metrics"""
        
        guide = {
            "ai_agent_instructions": {
                "overview": "This metrics suite provides comprehensive TikTok performance analytics for data-driven content strategy",
                "use_cases": [
                    "Content strategy optimization",
                    "Brand performance comparison", 
                    "Optimal posting time recommendations",
                    "Video duration optimization",
                    "Performance benchmarking"
                ],
                "key_files": {
                    "latest_metrics_summary.json": "Always up-to-date consolidated metrics - START HERE",
                    "brand_performance_*.json": "Detailed brand analytics and rankings",
                    "temporal_analytics_*.json": "Posting time optimization data",
                    "duration_performance_*.json": "Video length performance insights",
                    "top_performers_*.json": "High-performing content patterns"
                }
            },
            "query_examples": {
                "find_best_brand": "Use brand_rankings.top_by_engagement_rate from brand_performance file",
                "optimal_posting_time": "Use optimal_times.peak_engagement_hour from temporal_analytics file",
                "content_strategy": "Use ai_recommendations from content_type_performance file",
                "performance_benchmark": "Use performance_benchmarks from dataset_overview file"
            },
            "metric_definitions": {
                "engagement_rate": "Engagements divided by impressions, percentage",
                "completion_rate": "Percentage of video watched to completion",
                "view_rate": "Video views divided by impressions, percentage",
                "viral_threshold": "95th percentile of video views in dataset"
            },
            "ai_action_items": self._generate_ai_action_items(overview, brand_metrics, temporal_metrics)
        }
        
        return guide
    
    def generate_consolidated_summary(self, overview: Dict, brand_metrics: Dict, content_metrics: Dict, 
                                    temporal_metrics: Dict, duration_metrics: Dict, top_performers: Dict, worst_performers: Dict) -> Dict:
        """Generate consolidated summary for quick AI agent access"""
        
        return {
            "dataset_id": overview.get("dataset_id"),
            "quick_access": {
                "dataset_overview": {
                    "total_posts": overview["total_posts"],
                    "avg_engagement_rate": overview["key_metrics"]["average_engagement_rate"],
                    "viral_threshold": overview["performance_benchmarks"]["viral_threshold"]
                },
                "top_brands": brand_metrics["brand_rankings"]["top_by_engagement_rate"][:3],
                "worst_brands": self._extract_worst_brands(brand_metrics),
                "optimal_posting": {
                    "best_hour": temporal_metrics["optimal_times"]["peak_engagement_hour"],
                    "best_day": temporal_metrics["optimal_times"]["peak_engagement_day"]
                },
                "avoid_posting": self._extract_worst_posting_times(worst_performers),
                "optimal_duration": duration_metrics["optimal_duration"]["best_for_engagement"],
                "avoid_duration": self._extract_problematic_durations(worst_performers),
                "top_performing_posts": top_performers["top_posts"][:5],
                "worst_performing_posts": worst_performers["worst_posts"][:5]
            },
            "ai_recommendations": {
                "content_strategy": content_metrics.get("ai_recommendations", []),
                "posting_schedule": temporal_metrics.get("ai_scheduling_recommendations", []),
                "video_duration": duration_metrics.get("ai_duration_recommendations", []),
                "brand_focus": self._extract_brand_recommendations(brand_metrics),
                "content_to_avoid": worst_performers.get("avoidance_insights", []),
                "improvement_actions": worst_performers.get("improvement_recommendations", [])
            },
            "warning_signals": worst_performers.get("warning_signals", []),
            "performance_benchmarks": overview["performance_benchmarks"],
            "last_updated": datetime.now().isoformat(),
            "files_generated": {
                "brand_performance": f"tiktok_brand_performance_{self.timestamp}.json",
                "content_performance": f"tiktok_content_type_performance_{self.timestamp}.json",
                "temporal_analytics": f"tiktok_temporal_analytics_{self.timestamp}.json",
                "duration_analytics": f"tiktok_duration_performance_{self.timestamp}.json",
                "top_performers": f"tiktok_top_performers_{self.timestamp}.json",
                "worst_performers": f"tiktok_worst_performers_{self.timestamp}.json",
                "ai_guide": f"tiktok_ai_agent_guide_{self.timestamp}.json"
            }
        }
    
    def _extract_brands_from_labels(self, labels_str: str) -> List[str]:
        """Extract brand names from label string"""
        brands = []
        if labels_str and pd.notna(labels_str):
            parts = [part.strip() for part in labels_str.split(',')]
            for part in parts:
                if '[Brand]' in part:
                    brand = part.replace('[Brand]', '').strip()
                    if brand:
                        brands.append(brand)
        return brands
    
    def _extract_content_types_from_labels(self, labels_str: str) -> List[Dict]:
        """Extract content types with categories from label string"""
        content_types = []
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
                elif '[Package M]' in part:
                    name = part.replace('[Package M]', '').strip()
                    if name:
                        content_types.append({'name': name, 'category': 'package'})
        return content_types
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict:
        """Get date range from dataframe"""
        try:
            dates = pd.to_datetime(df['created_time'], errors='coerce').dropna()
            if len(dates) > 0:
                return {
                    "start_date": dates.min().isoformat(),
                    "end_date": dates.max().isoformat(),
                    "total_days": (dates.max() - dates.min()).days
                }
        except:
            pass
        return {"start_date": None, "end_date": None, "total_days": 0}
    
    def _generate_brand_ai_insights(self, brand_summary: pd.DataFrame) -> List[str]:
        """Generate AI-actionable insights about brand performance"""
        insights = []
        
        if len(brand_summary) > 0:
            top_brand = brand_summary.loc[brand_summary['avg_engagement_rate'].idxmax()]
            insights.append(f"'{top_brand['brand']}' achieves highest engagement rate at {top_brand['avg_engagement_rate']}%")
            
            if len(brand_summary) > 1:
                most_active = brand_summary.loc[brand_summary['total_posts'].idxmax()]
                insights.append(f"'{most_active['brand']}' is most active with {most_active['total_posts']} posts")
                
                high_completion = brand_summary.loc[brand_summary['avg_completion_rate'].idxmax()]
                insights.append(f"'{high_completion['brand']}' achieves best completion rate at {high_completion['avg_completion_rate']}%")
        
        return insights
    
    def _generate_content_ai_recommendations(self, content_summary: pd.DataFrame) -> List[str]:
        """Generate AI recommendations for content strategy"""
        recommendations = []
        
        if len(content_summary) > 0:
            top_content = content_summary.loc[content_summary['avg_engagement_rate'].idxmax()]
            recommendations.append(f"Focus on '{top_content['content_type']}' content type - achieves {top_content['avg_engagement_rate']}% engagement rate")
            
            if len(content_summary) > 1:
                best_completion = content_summary.loc[content_summary['avg_completion_rate'].idxmax()]
                recommendations.append(f"'{best_completion['content_type']}' content drives highest completion rates at {best_completion['avg_completion_rate']}%")
        
    def _extract_worst_brands(self, brand_metrics: Dict) -> List[Dict]:
        """Extract worst performing brands for AI avoidance"""
        
        if not brand_metrics.get("brand_summary"):
            return []
        
        brand_df = pd.DataFrame(brand_metrics["brand_summary"])
        if len(brand_df) == 0:
            return []
        
        # Get bottom 3 brands by engagement rate
        worst_brands = brand_df.nsmallest(3, 'avg_engagement_rate')[['brand', 'avg_engagement_rate']].to_dict('records')
        return worst_brands
    
    def _extract_worst_posting_times(self, worst_performers: Dict) -> Dict:
        """Extract worst posting times for AI scheduling avoidance"""
        
        anti_patterns = worst_performers.get("anti_patterns", {})
        problematic_times = anti_patterns.get("problematic_posting_times", {})
        
        if problematic_times:
            worst_hour = max(problematic_times.items(), key=lambda x: x[1])[0] if problematic_times else None
            return {
                "worst_hour": worst_hour,
                "problematic_hours": list(problematic_times.keys())
            }
        
        return {"worst_hour": None, "problematic_hours": []}
    
    def _extract_problematic_durations(self, worst_performers: Dict) -> Dict:
        """Extract problematic video durations"""
        
        anti_patterns = worst_performers.get("anti_patterns", {})
        duration_info = anti_patterns.get("problematic_durations", {})
        
        return {
            "avoid_duration_range": duration_info.get("most_common_range", ""),
            "problematic_avg_duration": duration_info.get("average_duration", 0)
        }
    
    def _generate_scheduling_recommendations(self, hourly_perf: pd.DataFrame, daily_perf: pd.DataFrame) -> List[str]:
        """Generate AI scheduling recommendations"""
        recommendations = []
        
        if len(hourly_perf) > 0:
            best_hour = hourly_perf.loc[hourly_perf['avg_engagements'].idxmax()]
            recommendations.append(f"Post at {best_hour['hour']}:00 for optimal engagement ({best_hour['avg_engagements']} avg engagements)")
        
        if len(daily_perf) > 0:
            best_day = daily_perf.loc[daily_perf['avg_engagements'].idxmax()]
            recommendations.append(f"Post on {best_day['day_of_week']} for best performance ({best_day['avg_engagements']} avg engagements)")
        
        return recommendations
    
    def _generate_duration_recommendations(self, duration_summary: pd.DataFrame) -> List[str]:
        """Generate AI duration optimization recommendations"""
        recommendations = []
        
        if len(duration_summary) > 0:
            best_engagement = duration_summary.loc[duration_summary['avg_engagement_rate'].idxmax()]
            recommendations.append(f"Use {best_engagement['duration_category']} videos for highest engagement ({best_engagement['avg_engagement_rate']}% rate)")
            
            best_completion = duration_summary.loc[duration_summary['avg_completion_rate'].idxmax()]
            recommendations.append(f"Use {best_completion['duration_category']} videos for best completion ({best_completion['avg_completion_rate']}% completion)")
        
        return recommendations
    
    def _analyze_top_performer_patterns(self, top_posts: pd.DataFrame) -> Dict:
        """Analyze patterns in top performing posts"""
        patterns = {
            "common_brands": [],
            "common_content_types": [],
            "average_duration": 0,
            "common_posting_hours": []
        }
        
        if len(top_posts) > 0:
            # Most common brands in top posts
            all_brands = []
            for labels in top_posts['tiktok_post_labels_names'].dropna():
                brands = self._extract_brands_from_labels(str(labels))
                all_brands.extend(brands)
            
            if all_brands:
                brand_counts = pd.Series(all_brands).value_counts()
                patterns["common_brands"] = brand_counts.head(5).to_dict()
            
            # Average duration of top posts
            patterns["average_duration"] = round(top_posts['tiktok_duration'].mean(), 1)
            
            # Common posting times
            top_posts['hour'] = pd.to_datetime(top_posts['created_time']).dt.hour
            hour_counts = top_posts['hour'].value_counts()
            patterns["common_posting_hours"] = hour_counts.head(3).to_dict()
        
    def generate_worst_performers_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze worst performing posts for pattern avoidance"""
        
        # Calculate engagement rate for each post
        df['engagement_rate'] = (pd.to_numeric(df['tiktok_insights_engagements'], errors='coerce') / 
                                pd.to_numeric(df['tiktok_insights_impressions'], errors='coerce') * 100).round(2)
        
        # Get worst performers (bottom 10% by engagement rate)
        bottom_threshold = df['engagement_rate'].quantile(0.1)
        worst_posts = df[df['engagement_rate'] <= bottom_threshold].copy()
        
        # Also include posts with very low completion rates
        low_completion = df[pd.to_numeric(df['tiktok_insights_completion_rate'], errors='coerce') < 0.2]
        
        # Combine worst performers
        worst_posts = pd.concat([worst_posts, low_completion]).drop_duplicates(subset=['tiktok_id'])
        
        # Analyze anti-patterns in worst performers
        worst_analysis = {
            "worst_posts": worst_posts[['tiktok_id', 'tiktok_post_labels_names', 'engagement_rate', 
                                      'tiktok_insights_completion_rate', 'tiktok_duration', 'created_time', 'tiktok_insights_engagements']].to_dict('records'),
            "anti_patterns": self._analyze_worst_performer_patterns(worst_posts),
            "avoidance_insights": self._generate_avoidance_insights(df, worst_posts),
            "improvement_recommendations": self._generate_improvement_recommendations(worst_posts),
            "warning_signals": self._identify_warning_signals(worst_posts)
        }
        
        return worst_analysis
    
    def _analyze_worst_performer_patterns(self, worst_posts: pd.DataFrame) -> Dict:
        """Identify anti-patterns in worst performing content"""
        
        anti_patterns = {
            "problematic_brands": [],
            "problematic_content_types": [],
            "problematic_durations": {},
            "problematic_posting_times": [],
            "common_characteristics": []
        }
        
        if len(worst_posts) == 0:
            return anti_patterns
        
        # Analyze problematic brands
        all_brands = []
        for labels in worst_posts['tiktok_post_labels_names'].dropna():
            brands = self._extract_brands_from_labels(str(labels))
            all_brands.extend(brands)
        
        if all_brands:
            brand_counts = pd.Series(all_brands).value_counts()
            # Brands that appear frequently in worst performers
            anti_patterns["problematic_brands"] = brand_counts.head(5).to_dict()
        
        # Analyze problematic content types
        all_content_types = []
        for labels in worst_posts['tiktok_post_labels_names'].dropna():
            content_types = self._extract_content_types_from_labels(str(labels))
            all_content_types.extend([ct['name'] for ct in content_types])
        
        if all_content_types:
            content_counts = pd.Series(all_content_types).value_counts()
            anti_patterns["problematic_content_types"] = content_counts.head(5).to_dict()
        
        # Analyze duration patterns
        durations = pd.to_numeric(worst_posts['tiktok_duration'], errors='coerce').dropna()
        if len(durations) > 0:
            anti_patterns["problematic_durations"] = {
                "average_duration": round(durations.mean(), 1),
                "most_common_range": self._get_duration_range(durations.median()),
                "duration_distribution": durations.describe().to_dict()
            }
        
        # Analyze posting times
        worst_posts['hour'] = pd.to_datetime(worst_posts['created_time'], errors='coerce').dt.hour
        hour_counts = worst_posts['hour'].value_counts()
        if len(hour_counts) > 0:
            anti_patterns["problematic_posting_times"] = hour_counts.head(3).to_dict()
        
        # Common characteristics
        avg_engagement = worst_posts['engagement_rate'].mean()
        avg_completion = worst_posts['tiktok_insights_completion_rate'].mean()
        
        anti_patterns["common_characteristics"] = [
            f"Average engagement rate: {avg_engagement:.2f}% (very low)",
            f"Average completion rate: {avg_completion:.2f}% (poor retention)",
            f"Total worst performers: {len(worst_posts)} posts"
        ]
        
        return anti_patterns
    
    def _generate_avoidance_insights(self, df: pd.DataFrame, worst_posts: pd.DataFrame) -> List[str]:
        """Generate specific insights about what to avoid"""
        
        insights = []
        
        if len(worst_posts) > 0 and len(df) > 0:
            # Compare worst vs average
            worst_avg_engagement = worst_posts['engagement_rate'].mean()
            overall_avg_engagement = df['engagement_rate'].mean()
            
            insights.append(f"AVOID: Worst performers average {worst_avg_engagement:.1f}% engagement vs {overall_avg_engagement:.1f}% overall")
            
            # Duration analysis
            worst_avg_duration = worst_posts['tiktok_duration'].mean()
            overall_avg_duration = df['tiktok_duration'].mean()
            
            if worst_avg_duration > overall_avg_duration * 1.3:
                insights.append("AVOID: Very long videos tend to underperform in your content")
            elif worst_avg_duration < overall_avg_duration * 0.7:
                insights.append("AVOID: Very short videos may not provide enough value")
            
            # Completion rate analysis
            worst_completion = worst_posts['tiktok_insights_completion_rate'].mean()
            if worst_completion < 0.3:
                insights.append("AVOID: Content with <30% completion rates - viewers lose interest quickly")
            
            # Time analysis
            worst_posts['hour'] = pd.to_datetime(worst_posts['created_time'], errors='coerce').dt.hour
            worst_hours = worst_posts['hour'].value_counts().head(2)
            if len(worst_hours) > 0:
                bad_hours = list(worst_hours.index)
                insights.append(f"AVOID: Posting at {bad_hours} hours shows poor performance")
        
        return insights
    
    def _generate_improvement_recommendations(self, worst_posts: pd.DataFrame) -> List[str]:
        """Generate specific recommendations to improve poor performance"""
        
        recommendations = []
        
        if len(worst_posts) == 0:
            return recommendations
        
        # Analyze what could be improved
        avg_completion = worst_posts['tiktok_insights_completion_rate'].mean()
        if avg_completion < 0.4:
            recommendations.append("IMPROVE: Focus on stronger video hooks in first 3 seconds")
            recommendations.append("IMPROVE: Test shorter, more engaging intro sequences")
        
        # Duration recommendations
        durations = pd.to_numeric(worst_posts['tiktok_duration'], errors='coerce').dropna()
        if len(durations) > 0:
            if durations.mean() > 45:
                recommendations.append("IMPROVE: Consider shorter video formats for better retention")
            elif durations.mean() < 10:
                recommendations.append("IMPROVE: Provide more value - videos may be too short")
        
        # Content recommendations
        recommendations.extend([
            "TEST: A/B test different content angles for underperforming topics",
            "REVIEW: Analyze top performers and adapt successful elements",
            "OPTIMIZE: Review posting times - may be reaching wrong audience timing"
        ])
        
        return recommendations
    
    def _identify_warning_signals(self, worst_posts: pd.DataFrame) -> List[Dict]:
        """Identify warning signals for AI agents to watch for"""
        
        warning_signals = []
        
        if len(worst_posts) == 0:
            return warning_signals
        
        # Engagement rate warnings
        very_low_engagement = worst_posts[worst_posts['engagement_rate'] < 1.0]
        if len(very_low_engagement) > 0:
            warning_signals.append({
                "signal": "very_low_engagement",
                "threshold": "< 1.0% engagement rate",
                "count": len(very_low_engagement),
                "action": "Immediate content strategy review required"
            })
        
        # Completion rate warnings
        very_low_completion = worst_posts[pd.to_numeric(worst_posts['tiktok_insights_completion_rate'], errors='coerce') < 0.15]
        if len(very_low_completion) > 0:
            warning_signals.append({
                "signal": "very_low_completion",
                "threshold": "< 15% completion rate", 
                "count": len(very_low_completion),
                "action": "Review video hooks and pacing"
            })
        
        # Zero engagement warnings
        zero_engagement = worst_posts[pd.to_numeric(worst_posts['tiktok_insights_engagements'], errors='coerce') == 0]
        if len(zero_engagement) > 0:
            warning_signals.append({
                "signal": "zero_engagement",
                "threshold": "0 engagements",
                "count": len(zero_engagement),
                "action": "Content may be off-brand or poorly timed"
            })
        
        return warning_signals
    
    def _get_duration_range(self, duration: float) -> str:
        """Get duration range category"""
        if duration <= 15:
            return "Short (0-15s)"
        elif duration <= 30:
            return "Medium (16-30s)"
        elif duration <= 60:
            return "Long (31-60s)"
        else:
            return "Extended (60s+)"
    
    def _generate_performance_insights(self, df: pd.DataFrame, top_posts: pd.DataFrame) -> List[str]:
        """Generate performance insights by comparing top posts to overall"""
        insights = []
        
        if len(top_posts) > 0 and len(df) > 0:
            avg_engagement = df['engagement_rate'].mean()
            top_avg_engagement = top_posts['engagement_rate'].mean()
            
            insights.append(f"Top performers achieve {top_avg_engagement:.1f}% engagement vs {avg_engagement:.1f}% average")
            
            avg_duration = df['tiktok_duration'].mean()
            top_avg_duration = top_posts['tiktok_duration'].mean()
            
            if top_avg_duration > avg_duration * 1.2:
                insights.append("Top performers tend to use longer video formats")
            elif top_avg_duration < avg_duration * 0.8:
                insights.append("Top performers tend to use shorter video formats")
            else:
                insights.append("Video duration doesn't strongly correlate with top performance")
        
        return insights
    
    def _generate_ai_action_items(self, overview: Dict, brand_metrics: Dict, temporal_metrics: Dict) -> List[str]:
        """Generate specific AI action items based on data"""
        actions = []
        
        # Brand recommendations
        if brand_metrics.get("brand_rankings", {}).get("top_by_engagement_rate"):
            top_brand = brand_metrics["brand_rankings"]["top_by_engagement_rate"][0]
            actions.append(f"PRIORITY: Create more content featuring '{top_brand['brand']}' (current top performer: {top_brand['avg_engagement_rate']}% engagement)")
        
        # Timing recommendations
        if temporal_metrics.get("optimal_times", {}).get("peak_engagement_hour"):
            peak_hour = temporal_metrics["optimal_times"]["peak_engagement_hour"]
            actions.append(f"SCHEDULE: Post at {peak_hour}:00 for optimal engagement")
        
        # Performance optimization
        avg_engagement = overview["key_metrics"]["average_engagement_rate"]
        if avg_engagement < 3.0:
            actions.append("OPTIMIZE: Engagement rate below 3% - review content strategy and posting times")
        elif avg_engagement > 8.0:
            actions.append("SCALE: High engagement rate detected - increase posting frequency")
        
        return actions
    
    def _extract_brand_recommendations(self, brand_metrics: Dict) -> List[str]:
        """Extract brand-focused recommendations for AI agents"""
        recommendations = []
        
        if brand_metrics.get("brand_rankings", {}).get("top_by_engagement_rate"):
            top_brands = brand_metrics["brand_rankings"]["top_by_engagement_rate"][:3]
            recommendations.extend([f"Prioritize {brand['brand']} content" for brand in top_brands])
        
        return recommendations
    
    def save_json(self, data: Dict, filename: str):
        """Save data as JSON file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Saved: {filepath}")
    
    def save_csv(self, data: List[Dict], filename: str):
        """Save data as CSV file"""
        if data:
            filepath = self.output_dir / filename
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"💾 Saved: {filepath}")