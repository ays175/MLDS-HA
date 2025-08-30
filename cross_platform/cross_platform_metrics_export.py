#!/usr/bin/env python3
"""
Cross-platform metrics exporter for agentic AI consumption (Sephora portfolio view)
Outputs a comprehensive suite of metrics to ./metrics/cross_platform with 'cross_platform_' prefix.
Grounded in platform data from Weaviate: TikTok, Facebook, Instagram.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import weaviate


@dataclass
class ExportContext:
    output_dir: Path
    timestamp: str
    include_tiktok: bool = True
    include_facebook: bool = True
    include_instagram: bool = True


class CrossPlatformMetricsExporter:
    """Exports cross-platform metrics across TikTok, Facebook, Instagram."""

    def __init__(self, output_dir: str = "./metrics/cross_platform"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------- Public API ----------------------
    def export_all_metrics(
        self,
        include_tiktok: bool = True,
        include_facebook: bool = True,
        include_instagram: bool = True,
    ) -> Dict[str, Any]:
        ctx = ExportContext(self.output_dir, self.timestamp, include_tiktok, include_facebook, include_instagram)

        print("📊 Generating CROSS-PLATFORM AI-agent metrics...")

        # 1. Load latest per-platform dataset ids (for tracing/provenance)
        platform_ids = self._load_platform_dataset_ids()

        # 2. Read source data from Weaviate and normalize into a single dataframe
        df = self._load_all_posts_from_weaviate(ctx)

        # Standardize numerics and derived fields
        df = self._coerce_and_derive(df)

        # 3. Compute full metrics suite
        overview = self._generate_dataset_overview(df, platform_ids)
        self._save_json(overview, f"cross_platform_dataset_overview_{ctx.timestamp}.json")
        self._save_csv([self._flatten_overview_for_csv(overview)], f"cross_platform_dataset_overview_{ctx.timestamp}.csv")

        # Sample for distributions
        sample = self._generate_per_post_sample(df)
        if sample:
            self._save_json({"per_post_sample": sample}, f"cross_platform_per_post_sample_{ctx.timestamp}.json")

        # Brand metrics
        brand_metrics = self._generate_brand_performance(df)
        self._save_json(brand_metrics, f"cross_platform_brand_performance_{ctx.timestamp}.json")
        self._save_csv(brand_metrics.get("brand_summary", []), f"cross_platform_brand_performance_{ctx.timestamp}.csv")

        # Content type metrics
        content_metrics = self._generate_content_type_performance(df)
        self._save_json(content_metrics, f"cross_platform_content_type_performance_{ctx.timestamp}.json")
        self._save_csv(content_metrics.get("content_summary", []), f"cross_platform_content_type_performance_{ctx.timestamp}.csv")

        # Temporal analytics
        temporal_metrics, hourly_csv, daily_csv = self._generate_temporal_analytics(df)
        self._save_json(temporal_metrics, f"cross_platform_temporal_analytics_{ctx.timestamp}.json")
        self._save_csv(hourly_csv, f"cross_platform_hourly_performance_{ctx.timestamp}.csv")
        self._save_csv(daily_csv, f"cross_platform_daily_performance_{ctx.timestamp}.csv")
        # Also JSON mirrors
        self._save_json({"hourly_performance": hourly_csv}, f"cross_platform_hourly_performance_{ctx.timestamp}.json")
        self._save_json({"daily_performance": daily_csv}, f"cross_platform_daily_performance_{ctx.timestamp}.json")

        # Top / worst performers
        top = self._generate_top_performers(df)
        worst = self._generate_worst_performers(df)
        self._save_json(top, f"cross_platform_top_performers_{ctx.timestamp}.json")
        self._save_csv(top.get("top_posts", []), f"cross_platform_top_performers_{ctx.timestamp}.csv")
        self._save_json(worst, f"cross_platform_worst_performers_{ctx.timestamp}.json")
        self._save_csv(worst.get("worst_posts", []), f"cross_platform_worst_performers_{ctx.timestamp}.csv")

        # Agent guide
        ai_guide = self._generate_ai_agent_guide()
        self._save_json(ai_guide, f"cross_platform_ai_agent_guide_{ctx.timestamp}.json")

        # Consolidated latest summary
        consolidated = self._generate_consolidated_summary(overview, brand_metrics, content_metrics, temporal_metrics, top, worst)
        self._save_json(consolidated, "latest_metrics_summary_cross_platform.json")

        print(f"✅ All CROSS-PLATFORM metrics exported to {self.output_dir}")
        return consolidated

    # ------------------ Data Loading & Normalization ------------------
    def _load_platform_dataset_ids(self) -> Dict[str, Optional[str]]:
        ids: Dict[str, Optional[str]] = {"tiktok": None, "facebook": None, "instagram": None}
        mapping = {
            "tiktok": Path("./metrics/tiktok/latest_metrics_summary_tiktok.json"),
            "facebook": Path("./metrics/facebook/latest_metrics_summary_facebook.json"),
            "instagram": Path("./metrics/instagram/latest_metrics_summary_instagram.json"),
        }
        for k, p in mapping.items():
            try:
                if p.exists():
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    ids[k] = (data.get("dataset_id") or data.get("summary", {}).get("dataset_id"))
            except Exception:
                pass
        return ids

    def _load_all_posts_from_weaviate(self, ctx: ExportContext) -> pd.DataFrame:
        client = weaviate.connect_to_local()
        try:
            frames: List[pd.DataFrame] = []
            if ctx.include_tiktok:
                frames.append(self._fetch_platform_posts(client, platform="tiktok"))
            if ctx.include_facebook:
                frames.append(self._fetch_platform_posts(client, platform="facebook"))
            if ctx.include_instagram:
                frames.append(self._fetch_platform_posts(client, platform="instagram"))
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        finally:
            client.close()

    def _fetch_platform_posts(self, client, platform: str) -> pd.DataFrame:
        """Pull minimal fields from each platform's Post collection and normalize columns."""
        if platform == "tiktok":
            try:
                coll = client.collections.get("TikTokPost")
            except Exception:
                print("⚠️ TikTokPost collection not found; skipping TikTok")
                return pd.DataFrame()
            rows = []
            for obj in coll.iterator():
                p = obj.properties
                rows.append({
                    "platform": "tiktok",
                    "post_id": p.get("tiktok_id"),
                    "created_time": p.get("created_time"),
                    "labels": p.get("tiktok_post_labels_names", ""),
                    "impressions": p.get("tiktok_insights_impressions", 0),
                    "views": p.get("tiktok_insights_video_views", 0),
                    "engagements": p.get("tiktok_insights_engagements", 0),
                    "completion_rate": p.get("tiktok_insights_completion_rate", 0),
                })
            return pd.DataFrame(rows)

        if platform == "facebook":
            try:
                coll = client.collections.get("FacebookPost")
            except Exception:
                print("⚠️ FacebookPost collection not found; skipping Facebook")
                return pd.DataFrame()
            rows = []
            for obj in coll.iterator():
                p = obj.properties
                rows.append({
                    "platform": "facebook",
                    "post_id": p.get("facebook_id"),
                    "created_time": p.get("created_time"),
                    "labels": p.get("facebook_post_labels_names", ""),
                    "impressions": p.get("facebook_insights_impressions", 0),
                    "views": p.get("facebook_insights_video_views", 0),
                    "engagements": p.get("facebook_insights_engagements", 0),
                    "completion_rate": p.get("facebook_insights_video_views_average_completion", 0),
                    "clicks": p.get("facebook_insights_post_clicks", 0),
                    "shares": p.get("facebook_shares", 0),
                    "reactions": p.get("facebook_reactions", 0),
                })
            return pd.DataFrame(rows)

        if platform == "instagram":
            try:
                coll = client.collections.get("InstagramPost")
            except Exception:
                print("⚠️ InstagramPost collection not found; skipping Instagram")
                return pd.DataFrame()
            rows = []
            for obj in coll.iterator():
                p = obj.properties
                rows.append({
                    "platform": "instagram",
                    "post_id": p.get("instagram_id"),
                    "created_time": p.get("created_time"),
                    "labels": p.get("instagram_post_labels_names", ""),
                    "impressions": p.get("instagram_insights_impressions", 0),
                    "views": p.get("instagram_insights_video_views", 0),
                    "engagements": p.get("instagram_insights_engagement", 0),
                    "story_completion_rate": p.get("instagram_insights_story_completion_rate", 0),
                    "clicks": p.get("instagram_insights_post_clicks", 0),
                    "shares": p.get("instagram_shares", 0),
                    "reactions": p.get("instagram_reactions", 0),
                    "saves": p.get("instagram_insights_saves", 0),
                })
            return pd.DataFrame(rows)

        return pd.DataFrame()

    # ------------------ Metrics Generators ------------------
    def _coerce_and_derive(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Replace infs for safety
        df = df.replace([np.inf, -np.inf], np.nan)
        # Coerce numeric columns
        numeric_cols = [
            "impressions", "views", "engagements", "completion_rate", "story_completion_rate",
            "clicks", "shares", "reactions", "saves"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Timestamps
        df["created_time"] = pd.to_datetime(df.get("created_time"), errors='coerce')
        df["hour"] = df["created_time"].dt.hour
        df["day_of_week"] = df["created_time"].dt.day_name()
        # Derived engagement_rate (prefer views; fallback to impressions)
        den = df["views"].where(df["views"] > 0, df["impressions"])  # use impressions when views==0
        df["engagement_rate"] = (df["engagements"] / den.replace(0, np.nan) * 100).fillna(0).round(2)
        # Derived view_rate
        df["view_rate"] = (df["views"] / df["impressions"].replace(0, np.nan) * 100).fillna(0).round(2)
        # Additional rates
        for num_col, rate_col in [
            ("clicks", "click_rate"), ("shares", "share_rate"), ("reactions", "reaction_rate"), ("saves", "save_rate")
        ]:
            if num_col in df.columns:
                df[rate_col] = (df[num_col] / df["impressions"].replace(0, np.nan) * 100).fillna(0).round(2)
        # Brand / content extraction from labels
        brands, ctypes = [], []
        labels_series = df.get("labels")
        if labels_series is None:
            labels_iterable = []
        else:
            # Ensure iterable of strings without NaNs
            labels_iterable = labels_series.fillna("").astype(str).tolist()
        for lbl in labels_iterable:
            b, c = self._extract_from_labels(lbl)
            brands.append(b)
            ctypes.append(c)
        df["brand"] = brands
        df["content_type"] = ctypes
        return df

    def _extract_from_labels(self, labels_str: str) -> Tuple[Optional[str], Optional[str]]:
        if not labels_str:
            return None, None
        brand, content_type = None, None
        for part in [p.strip() for p in labels_str.split(',') if p.strip()]:
            if part.startswith('[Brand]'):
                val = part.replace('[Brand]', '').strip()
                brand = val or brand
            elif part.startswith('[Axis]'):
                val = part.replace('[Axis]', '').strip()
                content_type = val or content_type
            elif part.startswith('[Asset]'):
                val = part.replace('[Asset]', '').strip()
                content_type = content_type or val
            elif part.startswith('[Axis & Offer]'):
                val = part.replace('[Axis & Offer]', '').strip()
                content_type = content_type or val
        return brand, content_type

    def _generate_dataset_overview(self, df: pd.DataFrame, platform_ids: Dict[str, Optional[str]]) -> Dict:
        totals = {
            "total_posts": int(len(df)),
            "total_impressions": int(df.get("impressions", pd.Series(dtype=float)).sum()),
            "total_video_views": int(df.get("views", pd.Series(dtype=float)).sum()),
            "total_engagements": int(df.get("engagements", pd.Series(dtype=float)).sum()),
        }
        avg_eng_rate = float((df["engagements"].sum() / df["views"].sum() * 100) if df["views"].sum() > 0 else 0)
        # Harmonize completion metrics (completion_rate or story_completion_rate)
        comp_series = pd.concat([
            df.get("completion_rate", pd.Series(dtype=float)).fillna(0),
            df.get("story_completion_rate", pd.Series(dtype=float)).fillna(0)
        ], ignore_index=True)
        avg_comp = float(comp_series.mean() if len(comp_series) else 0)
        # Benchmarks
        views_series = df.get("views", pd.Series(dtype=float)).fillna(0)
        viral_thr = float(views_series.quantile(0.95) if len(views_series) else 0)
        er_series = df.get("engagement_rate", pd.Series(dtype=float)).fillna(0)

        overview = {
            "dataset_id_cross": self._compose_cross_id(platform_ids),
            "platform_dataset_ids": platform_ids,
            "generated_at": datetime.now().isoformat(),
            "total_posts": totals["total_posts"],
            "date_range": self._get_date_range(df),
            "key_metrics": {
                **totals,
                "average_engagement_rate": round(avg_eng_rate, 2),
                "average_completion_or_story_rate": round(avg_comp, 2),
                "max_engagements_post": int(df.get("engagements", pd.Series(dtype=float)).max() or 0),
                "max_views_post": int(df.get("views", pd.Series(dtype=float)).max() or 0),
                "max_impressions_post": int(df.get("impressions", pd.Series(dtype=float)).max() or 0),
            },
            "performance_benchmarks": {
                "viral_threshold": round(viral_thr, 2),
                "engagement_rate_median": round(float(er_series.quantile(0.5) if len(er_series) else 0.0), 2),
                "engagement_rate_p75": round(float(er_series.quantile(0.75) if len(er_series) else 0.0), 2),
                "engagement_rate_p95": round(float(er_series.quantile(0.95) if len(er_series) else 0.0), 2),
            },
        }
        return overview

    def _flatten_overview_for_csv(self, ov: Dict) -> Dict:
        km = ov.get("key_metrics", {})
        dr = ov.get("date_range", {})
        return {
            'dataset_id_cross': ov.get('dataset_id_cross'),
            'generated_at': ov.get('generated_at'),
            'total_posts': km.get('total_posts'),
            'start_date': dr.get('start_date'),
            'end_date': dr.get('end_date'),
            'total_impressions': km.get('total_impressions'),
            'total_engagements': km.get('total_engagements'),
            'total_video_views': km.get('total_video_views'),
            'average_engagement_rate': km.get('average_engagement_rate'),
            'average_completion_or_story_rate': km.get('average_completion_or_story_rate'),
        }

    def _generate_per_post_sample(self, df: pd.DataFrame, max_rows: int = 5000) -> List[Dict]:
        cols = [c for c in ["impressions", "views", "engagements", "completion_rate", "story_completion_rate", "platform"] if c in df.columns]
        if not cols or df.empty:
            return []
        sampled = df[cols].dropna(how='all')
        n = min(max_rows, len(sampled))
        if n <= 0:
            return []
        return sampled.sample(n=n, random_state=42).to_dict('records')

    def _generate_brand_performance(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {"brand_summary": [], "brand_rankings": {}, "generated_at": datetime.now().isoformat()}
        use = df.copy()
        use["brand"] = use["brand"].fillna("Unknown")
        grp = use.groupby("brand").agg({
            "post_id": 'count',
            "impressions": ['sum', 'mean'],
            "engagements": ['sum', 'mean'],
            "views": ['sum', 'mean'],
            "completion_rate": 'mean',
            "story_completion_rate": 'mean',
        }).round(2)
        grp.columns = [
            'total_posts', 'total_impressions', 'avg_impressions',
            'total_engagements', 'avg_engagements', 'total_views', 'avg_views',
            'avg_completion_rate', 'avg_story_completion_rate'
        ]
        grp["avg_engagement_rate"] = (grp["avg_engagements"] / grp["avg_impressions"] * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        grp = grp.reset_index()
        brand_summary = grp.to_dict('records')
        brand_rankings = {
            "top_by_engagement_rate": grp.nlargest(5, 'avg_engagement_rate')[['brand', 'avg_engagement_rate']].to_dict('records'),
            "top_by_completion_or_story": grp.assign(_comp=grp[['avg_completion_rate','avg_story_completion_rate']].max(axis=1)).nlargest(5, '_comp')[['brand','_comp']].rename(columns={'_comp':'avg_completion_or_story'}).to_dict('records'),
            "top_by_total_views": grp.nlargest(5, 'total_views')[['brand', 'total_views']].to_dict('records'),
            "most_active_brands": grp.nlargest(5, 'total_posts')[['brand', 'total_posts']].to_dict('records')
        }
        return {"brand_summary": brand_summary, "brand_rankings": brand_rankings, "generated_at": datetime.now().isoformat()}

    def _generate_content_type_performance(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {"content_summary": [], "content_rankings": {}, "generated_at": datetime.now().isoformat()}
        use = df.copy()
        use["content_type"] = use["content_type"].fillna("Unknown")
        grp = use.groupby("content_type").agg({
            "post_id": 'count',
            "impressions": ['sum', 'mean'],
            "engagements": ['sum', 'mean'],
            "views": ['sum', 'mean'],
            "completion_rate": 'mean',
            "story_completion_rate": 'mean',
        }).round(2)
        grp.columns = [
            'total_posts', 'total_impressions', 'avg_impressions',
            'total_engagements', 'avg_engagements', 'total_views', 'avg_views',
            'avg_completion_rate', 'avg_story_completion_rate'
        ]
        grp["avg_engagement_rate"] = (grp["avg_engagements"] / grp["avg_impressions"] * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        grp = grp.reset_index()
        content_summary = grp.to_dict('records')
        content_rankings = {
            "highest_engagement_content": grp.nlargest(5, 'avg_engagement_rate')[['content_type', 'avg_engagement_rate']].to_dict('records'),
            "highest_completion_or_story": grp.assign(_comp=grp[['avg_completion_rate','avg_story_completion_rate']].max(axis=1)).nlargest(5, '_comp')[['content_type','_comp']].rename(columns={'_comp':'avg_completion_or_story'}).to_dict('records'),
            "most_popular_content": grp.nlargest(5, 'total_posts')[['content_type', 'total_posts']].to_dict('records')
        }
        return {"content_summary": content_summary, "content_rankings": content_rankings, "generated_at": datetime.now().isoformat()}

    def _generate_temporal_analytics(self, df: pd.DataFrame) -> Tuple[Dict, List[Dict], List[Dict]]:
        if df.empty:
            return {"hourly_performance": [], "daily_performance": [], "hour_by_day": [], "optimal_times": {}, "generated_at": datetime.now().isoformat()}, [], []
        # Hourly
        h = df.groupby('hour').agg({
            'engagements': 'mean', 'views': 'mean', 'completion_rate': 'mean', 'story_completion_rate': 'mean', 'post_id': 'count'
        }).round(2)
        hourly = h.reset_index().rename(columns={'engagements':'avg_engagements','views':'avg_views','completion_rate':'avg_completion_rate','story_completion_rate':'avg_story_completion_rate','post_id':'post_count'})
        # Daily
        d = df.groupby('day_of_week').agg({
            'engagements': 'mean', 'views': 'mean', 'completion_rate': 'mean', 'story_completion_rate': 'mean', 'post_id': 'count'
        }).round(2)
        daily = d.reset_index().rename(columns={'engagements':'avg_engagements','views':'avg_views','completion_rate':'avg_completion_rate','story_completion_rate':'avg_story_completion_rate','post_id':'post_count'})
        # Hour-by-day grid
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=ordered_days, ordered=True)
        grid = df.groupby(['day_of_week','hour']).agg({'engagements':'mean','views':'mean'}).round(2).reset_index()
        hour_by_day = [
            {
                'day_of_week': str(row['day_of_week']),
                'hour': int(row['hour']),
                'avg_engagements': float(row['engagements']) if pd.notna(row['engagements']) else 0.0,
                'avg_views': float(row['views']) if pd.notna(row['views']) else 0.0,
            } for _, row in grid.iterrows()
        ]
        optimal = {
            "best_hours": hourly.nlargest(3, 'avg_engagements')[['hour','avg_engagements']].to_dict('records') if len(hourly) else [],
            "best_days": daily.nlargest(3, 'avg_engagements')[['day_of_week','avg_engagements']].to_dict('records') if len(daily) else [],
        }
        return (
            {"hourly_performance": hourly.to_dict('records'), "daily_performance": daily.to_dict('records'), "hour_by_day": hour_by_day, "optimal_times": optimal, "generated_at": datetime.now().isoformat()},
            hourly.to_dict('records'),
            daily.to_dict('records')
        )

    def _generate_top_performers(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {"top_posts": []}
        use = df.copy()
        thr = use['engagement_rate'].quantile(0.9) if len(use) else 0
        top = use[use['engagement_rate'] >= thr][['platform','post_id','labels','engagement_rate','completion_rate','story_completion_rate','engagements','views']].copy()
        return {"top_posts": top.sort_values('engagement_rate', ascending=False).to_dict('records')}

    def _generate_worst_performers(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {"worst_posts": []}
        use = df.copy()
        thr = use['engagement_rate'].quantile(0.1) if len(use) else 0
        worst = use[use['engagement_rate'] <= thr][['platform','post_id','labels','engagement_rate','completion_rate','story_completion_rate','engagements','views','created_time']].copy()
        return {"worst_posts": worst.sort_values('engagement_rate', ascending=True).to_dict('records')}

    def _generate_ai_agent_guide(self) -> Dict:
        return {
            "ai_agent_instructions": {
                "overview": "This suite provides cross-platform performance analytics for Sephora's portfolio.",
                "key_files": {
                    "latest_metrics_summary_cross_platform.json": "Always up-to-date consolidated metrics - START HERE",
                    "cross_platform_brand_performance_*.json": "Brand analytics and rankings across platforms",
                    "cross_platform_temporal_analytics_*.json": "Posting time optimization across platforms",
                    "cross_platform_top_performers_*.json": "High-performing content patterns"
                }
            }
        }

    def _generate_consolidated_summary(self, overview: Dict, brand_metrics: Dict, content_metrics: Dict, temporal_metrics: Dict, top: Dict, worst: Dict) -> Dict:
        return {
            "dataset_id": overview.get("dataset_id_cross"),
            "quick_access": {
                "dataset_overview": {
                    "total_posts": overview.get("key_metrics", {}).get("total_posts"),
                    "avg_engagement_rate": overview.get("key_metrics", {}).get("average_engagement_rate"),
                    "viral_threshold": overview.get("performance_benchmarks", {}).get("viral_threshold"),
                },
                "top_performing_posts": (top or {}).get("top_posts", [])[:5],
                "worst_performing_posts": (worst or {}).get("worst_posts", [])[:5],
            },
            "last_updated": datetime.now().isoformat(),
            "files_generated": {
                "brand_performance": f"cross_platform_brand_performance_{self.timestamp}.json",
                "content_performance": f"cross_platform_content_type_performance_{self.timestamp}.json",
                "temporal_analytics": f"cross_platform_temporal_analytics_{self.timestamp}.json",
                "top_performers": f"cross_platform_top_performers_{self.timestamp}.json",
                "worst_performers": f"cross_platform_worst_performers_{self.timestamp}.json",
                "ai_guide": f"cross_platform_ai_agent_guide_{self.timestamp}.json",
            }
        }

    # ------------------ Utils ------------------
    def _compose_cross_id(self, ids: Dict[str, Optional[str]]) -> str:
        parts = [ids.get('tiktok') or 'na', ids.get('facebook') or 'na', ids.get('instagram') or 'na']
        return "cross_platform_" + "_".join([p.replace(' ', '-') for p in parts])

    def _get_date_range(self, df: pd.DataFrame) -> Dict:
        try:
            dates = pd.to_datetime(df['created_time'], errors='coerce').dropna()
            if len(dates) > 0:
                return {"start_date": dates.min().isoformat(), "end_date": dates.max().isoformat(), "total_days": (dates.max() - dates.min()).days}
        except Exception:
            pass
        return {"start_date": None, "end_date": None, "total_days": 0}

    def _save_json(self, data: Dict, filename: str):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Saved: {filepath}")

    def _save_csv(self, data: List[Dict], filename: str):
        if data:
            filepath = self.output_dir / filename
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"💾 Saved: {filepath}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export cross-platform metrics for agent consumption')
    parser.add_argument('--include-tiktok', action='store_true', default=True)
    parser.add_argument('--include-facebook', action='store_true', default=True)
    parser.add_argument('--include-instagram', action='store_true', default=True)
    parser.add_argument('--output-dir', default='./metrics/cross_platform')
    args = parser.parse_args()

    exporter = CrossPlatformMetricsExporter(output_dir=args.output_dir)
    exporter.export_all_metrics(
        include_tiktok=args.include_tiktok,
        include_facebook=args.include_facebook,
        include_instagram=args.include_instagram,
    )


if __name__ == '__main__':
    main()


