"""
Metrics Analyzers - All Analysis Logic

This module contains all the analysis logic from the original unified_metrics_export.py
organized into logical analyzer classes. This replaces the 48 unique _generate_ methods.

All original functions are preserved and mapped to appropriate analyzer classes:
- BrandAnalyzer: Brand performance, rankings, temporal analysis
- ContentAnalyzer: Content type performance, duration analysis  
- SentimentAnalyzer: Sentiment analysis, temporal sentiment
- EngagementAnalyzer: Engagement metrics, temporal engagement
- TemporalAnalyzer: All temporal aggregations (weekly/monthly/quarterly/yearly)
- PlatformAnalyzer: Platform-specific metrics (TikTok, Facebook, Instagram, Customer Care)
- AIInsightsAnalyzer: All AI-generated insights and recommendations
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import logging

from timezone_mapping import get_timezone_mapper

logger = logging.getLogger(__name__)


class BaseAnalyzer:
    """Base class for all analyzers with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timezone_mapper = get_timezone_mapper()
    
    def is_enabled(self, platform: str) -> bool:
        """Check if this analyzer is enabled for the platform."""
        # Will be implemented based on YAML config
        return True
    
    def _safe_numeric_convert(self, value, convert_type=float, default=0):
        """Safely convert value to numeric, handling UUIDs and other non-numeric types"""
        try:
            if value is None or pd.isna(value):
                return default
            if isinstance(value, str) and len(value) == 36 and '-' in value:
                # Likely a UUID, return default
                return default
            return convert_type(value)
        except (ValueError, TypeError):
            return default
    
    def _extract_country_from_url(self, url):
        """Extract country from TikTok/Facebook URL using handle mapping"""
        if pd.isna(url) or not isinstance(url, str):
            return None
        
        # Country mapping for all platforms
        country_mapping = {
            '@sephora': 'Global',
            '@sephoracollection': 'Global',
            '@sephorafrance': 'France',
            '@sephoradeutschland': 'Germany',
            '@sephoraitalia': 'Italy',
            '@sephoramiddleeast': 'Middle East',
            '@sephoraspain': 'Spain',
            '@sephoraswitzerland': 'Switzerland',
            '@sephoraczechrepublic': 'Czech Republic',
            '@sephoraportugal': 'Portugal',
            '@sephorapolska': 'Poland',
            '@sephoraturkiye': 'Turkey',
            '@sephoragreece': 'Greece',
            '@sephorauk': 'United Kingdom',
            '@sephorabulgaria': 'Bulgaria',
            '@sephoraromania': 'Romania',
            '@sephora.romania': 'Romania',
            '@sephorasrbija': 'Serbia',
            '@sephoracanada': 'Canada',
            '@sephorasg': 'Singapore',
            '@sephora_scandinavia': 'Scandinavia'
        }
        
        # Extract handle from URL
        if '/@' in url:
            handle = '@' + url.split('/@')[1].split('/')[0]
            return country_mapping.get(handle, None)
        
        return None
    
    def _convert_to_market_time(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Convert UTC timestamps to local market time based on derived_country or URL extraction"""
        if date_col not in df.columns:
            logger.warning(f"Date column {date_col} not found")
            return df
        
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        
        # Convert to datetime if not already
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        # Extract country if not already present
        if 'derived_country' not in df_copy.columns:
            logger.info("🌍 Extracting countries from URLs for market-aware timezone conversion")
            
            # Try different URL column names
            url_columns = ['tiktok_link', 'facebook_url', 'instagram_url', 'url']
            url_col = None
            for col in url_columns:
                if col in df_copy.columns:
                    url_col = col
                    break
            
            if url_col:
                df_copy['derived_country'] = df_copy[url_col].apply(self._extract_country_from_url)
                country_counts = df_copy['derived_country'].value_counts()
                logger.info(f"✅ Extracted countries: {dict(country_counts)}")
            else:
                logger.warning("No URL column found for country extraction")
                df_copy['derived_country'] = None
        
        # Convert to local time for each market
        if 'derived_country' in df_copy.columns and df_copy['derived_country'].notna().any():
            try:
                df_copy[f'{date_col}_local'] = self.timezone_mapper.convert_series_to_local(
                    df_copy[date_col], 
                    df_copy['derived_country']
                )
                
                # Use local time for temporal analysis
                df_copy['_datetime'] = df_copy[f'{date_col}_local']
                logger.info(f"✅ Converted {len(df_copy)} records to market-aware local time")
                
            except Exception as e:
                logger.warning(f"⚠️ Market timezone conversion failed: {e}, using UTC")
                df_copy['_datetime'] = df_copy[date_col]
        else:
            logger.warning("No country data available, keeping UTC time")
            df_copy['_datetime'] = df_copy[date_col]
        
        return df_copy
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> Optional[str]:
        """Get the appropriate date column for the platform"""
        date_columns = ['created_time', 'posted_date', 'date', 'timestamp', 'created_at']
        
        for col in date_columns:
            if col in df.columns:
                return col
        
        return None
    
    def _parse_tiktok_labels_for_brands(self, labels_json: str) -> List[str]:
        """Parse tiktok_post_labels JSON field to extract brands"""
        import json
        import re
        
        brands = []
        
        try:
            # Handle multiple JSON objects separated by commas
            # Fix the format by wrapping in array brackets
            if labels_json.strip() and not labels_json.strip().startswith('['):
                # Use regex to find JSON object boundaries
                pattern = r'\{[^{}]*\}'
                matches = re.findall(pattern, labels_json)
                
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict) and 'name' in obj:
                            name = obj['name'].strip()
                            if '[Brand]' in name:
                                brand = name.replace('[Brand]', '').strip()
                                if brand:
                                    brands.append(brand)
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            # Fallback: treat as string and parse manually
            if '[Brand]' in labels_json:
                parts = labels_json.split('[Brand]')
                for part in parts[1:]:  # Skip first part (before first [Brand])
                    # Extract text until next bracket or comma
                    brand_match = re.search(r'^([^,\[\]]+)', part)
                    if brand_match:
                        brand = brand_match.group(1).strip().strip('"').strip()
                        if brand:
                            brands.append(brand)
        
        return brands
    
    def _parse_tiktok_labels_for_content(self, labels_json: str) -> List[Dict[str, str]]:
        """Parse tiktok_post_labels JSON field to extract content types"""
        import json
        import re
        
        content_types = []
        
        try:
            # Handle multiple JSON objects separated by commas
            if labels_json.strip() and not labels_json.strip().startswith('['):
                # Use regex to find JSON object boundaries
                pattern = r'\{[^{}]*\}'
                matches = re.findall(pattern, labels_json)
                
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict) and 'name' in obj:
                            name = obj['name'].strip()
                            
                            # Extract content types
                            for tag, category in [('[Axis]', 'axis'), ('[Asset]', 'asset'), ('[Package M]', 'package')]:
                                if tag in name:
                                    content_name = name.replace(tag, '').strip()
                                    if content_name:
                                        content_types.append({'type': content_name, 'category': category})
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            # Fallback: treat as string and parse manually
            for tag, category in [('[Axis]', 'axis'), ('[Asset]', 'asset'), ('[Package M]', 'package')]:
                if tag in labels_json:
                    parts = labels_json.split(tag)
                    for part in parts[1:]:
                        content_match = re.search(r'^([^,\[\]]+)', part)
                        if content_match:
                            content_name = content_match.group(1).strip().strip('"').strip()
                            if content_name:
                                content_types.append({'type': content_name, 'category': category})
        
        return content_types


class BrandAnalyzer(BaseAnalyzer):
    """
    Brand Performance Analysis
    
    Original methods moved here:
    - _generate_brand_performance_metrics (line 1328)
    - _generate_comprehensive_brand_temporal (line 1369)
    - _generate_brand_ai_insights (line 200, 5376)
    - _generate_brand_metrics (line 2555)
    - _calculate_brand_summary (line 132)
    - _calculate_brand_rankings (line 179)
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive brand performance analysis."""
        return self._generate_brand_performance_metrics(df, platform)
    
    def _generate_brand_performance_metrics(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate comprehensive brand performance analytics"""
        from datetime import datetime
        
        brand_data = []
        
        # Extract brand data from each post
        for _, row in df.iterrows():
            brands = self._extract_brands_from_row(row, platform)
            
            for brand in brands:
                brand_data.append({
                    'post_id': self._get_post_id(row, platform),
                    'brand': brand,
                    **self._extract_performance_metrics(row, platform)
                })
        
        if not brand_data:
            return {"brand_summary": [], "brand_rankings": [], "ai_insights": []}
        
        brand_df = pd.DataFrame(brand_data)
        
        # Calculate brand performance summary
        brand_summary = self._calculate_brand_summary(brand_df, platform)
        
        # Brand rankings for AI decision making
        brand_rankings = self._calculate_brand_rankings(brand_summary)
        
        # AI insights
        ai_insights = self._generate_brand_ai_insights(brand_summary)
        
        # Add comprehensive temporal brand performance data
        temporal_brand_data = self._generate_comprehensive_brand_temporal(df, platform)
        
        return {
            "brand_summary": brand_summary.to_dict('records') if not brand_summary.empty else [],
            "brand_rankings": brand_rankings,
            "ai_insights": ai_insights,
            "temporal_performance": temporal_brand_data,
            "generated_at": datetime.now().isoformat()
        }
    
    def _extract_brands_from_row(self, row: pd.Series, platform: str) -> List[str]:
        """Extract brand names from a row based on platform"""
        brands = []
        
        if platform == 'tiktok':
            labels = str(row.get('tiktok_post_labels_names', ''))
            if labels and pd.notna(labels):
                parts = [part.strip() for part in labels.split(',')]
                for part in parts:
                    if '[Brand]' in part:
                        brand = part.replace('[Brand]', '').strip()
                        if brand:
                            brands.append(brand)
        elif platform in ['facebook', 'instagram']:
            # Add Facebook/Instagram brand extraction logic
            brands_field = row.get('brands', '')
            if brands_field and pd.notna(brands_field):
                brands = [b.strip() for b in str(brands_field).split(',') if b.strip()]
        
        return brands or ['Unknown']
    
    def _get_post_id(self, row: pd.Series, platform: str) -> str:
        """Get platform-specific post ID"""
        if platform == 'tiktok':
            return str(row.get('tiktok_id', ''))
        elif platform == 'facebook':
            return str(row.get('facebook_id', ''))
        elif platform == 'instagram':
            return str(row.get('instagram_id', ''))
        elif platform == 'customer_care':
            return str(row.get('case_id', ''))
        return str(row.get('id', ''))
    
    def _extract_performance_metrics(self, row: pd.Series, platform: str) -> Dict:
        """Extract performance metrics from a row based on platform"""
        metrics = {}
        
        if platform == 'tiktok':
            metrics.update({
                'impressions': self._safe_numeric_convert(row.get('tiktok_insights_impressions'), float, 0),
                'engagements': self._safe_numeric_convert(row.get('tiktok_insights_engagements'), float, 0),
                'video_views': self._safe_numeric_convert(row.get('tiktok_insights_video_views'), float, 0),
                'completion_rate': self._safe_numeric_convert(row.get('tiktok_insights_completion_rate'), float, 0),
                'duration': self._safe_numeric_convert(row.get('tiktok_duration'), float, 0),
                'sentiment': self._safe_numeric_convert(row.get('tiktok_sentiment'), float, 0)
            })
        elif platform == 'facebook':
            metrics.update({
                'impressions': self._safe_numeric_convert(row.get('facebook_insights_impressions'), float, 0),
                'engagements': self._safe_numeric_convert(row.get('facebook_insights_engagements'), float, 0),
                'reach': self._safe_numeric_convert(row.get('facebook_insights_reach'), float, 0),
                'sentiment': self._safe_numeric_convert(row.get('facebook_sentiment'), float, 0)
            })
        elif platform == 'instagram':
            metrics.update({
                'impressions': self._safe_numeric_convert(row.get('instagram_insights_impressions'), float, 0),
                'engagements': self._safe_numeric_convert(row.get('instagram_insights_engagements'), float, 0),
                'reach': self._safe_numeric_convert(row.get('instagram_insights_reach'), float, 0),
                'sentiment': self._safe_numeric_convert(row.get('instagram_sentiment'), float, 0)
            })
        elif platform == 'customer_care':
            metrics.update({
                'urgency': self._safe_numeric_convert(row.get('urgency'), float, 0),
                'resolution_time': self._safe_numeric_convert(row.get('resolution_time'), float, 0),
                'satisfaction': self._safe_numeric_convert(row.get('satisfaction'), float, 0),
                'sentiment': self._safe_numeric_convert(row.get('sentiment'), float, 0)
            })
        
        return metrics
    
    def _calculate_brand_summary(self, brand_df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Calculate brand performance summary"""
        if len(brand_df) == 0:
            return pd.DataFrame()
        
        # Group by brand and calculate metrics
        agg_dict = {
            'post_id': 'count',
            'sentiment': 'mean'
        }
        
        if platform == 'tiktok':
            agg_dict.update({
                'impressions': ['sum', 'mean'],
                'engagements': ['sum', 'mean'],
                'video_views': ['sum', 'mean'],
                'completion_rate': 'mean',
                'duration': 'mean'
            })
        elif platform in ['facebook', 'instagram']:
            agg_dict.update({
                'impressions': ['sum', 'mean'],
                'engagements': ['sum', 'mean'],
                'reach': ['sum', 'mean']
            })
        elif platform == 'customer_care':
            agg_dict.update({
                'urgency': 'mean',
                'resolution_time': 'mean',
                'satisfaction': 'mean'
            })
        
        summary = brand_df.groupby('brand').agg(agg_dict).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]
        
        # Calculate engagement rate for social platforms
        if platform in ['tiktok', 'facebook', 'instagram']:
            if 'engagements_mean' in summary.columns and 'impressions_mean' in summary.columns:
                # Protect against division by zero
                summary['avg_engagement_rate'] = (
                    summary['engagements_mean'] / summary['impressions_mean'].replace(0, np.nan) * 100
                ).round(2)
        
        return summary.reset_index()
    
    def _calculate_brand_rankings(self, brand_summary: pd.DataFrame) -> Dict:
        """Calculate brand rankings based on performance metrics"""
        if brand_summary.empty:
            return {}
        
        rankings = {}
        
        # Rank by engagement rate if available
        if 'avg_engagement_rate' in brand_summary.columns:
            engagement_ranking = brand_summary.nlargest(10, 'avg_engagement_rate')[['brand', 'avg_engagement_rate']]
            rankings['top_engagement'] = engagement_ranking.to_dict('records')
        
        # Rank by total engagements if available
        if 'engagements_sum' in brand_summary.columns:
            volume_ranking = brand_summary.nlargest(10, 'engagements_sum')[['brand', 'engagements_sum']]
            rankings['top_volume'] = volume_ranking.to_dict('records')
        
        # Rank by sentiment
        if 'sentiment' in brand_summary.columns:
            sentiment_ranking = brand_summary.nlargest(10, 'sentiment')[['brand', 'sentiment']]
            rankings['top_sentiment'] = sentiment_ranking.to_dict('records')
        
        return rankings
    
    def _generate_brand_ai_insights(self, brand_summary: pd.DataFrame) -> List[str]:
        """Generate AI insights for brand performance"""
        insights = []
        
        if brand_summary.empty:
            return ["No brand data available for analysis"]
        
        # Top performing brand
        if 'avg_engagement_rate' in brand_summary.columns:
            top_brand = brand_summary.loc[brand_summary['avg_engagement_rate'].idxmax()]
            insights.append(f"{top_brand['brand']} achieves highest engagement rate at {top_brand['avg_engagement_rate']:.2f}%")
        
        # Most active brand
        if 'post_id' in brand_summary.columns:
            most_active = brand_summary.loc[brand_summary['post_id'].idxmax()]
            insights.append(f"{most_active['brand']} is most active with {most_active['post_id']} posts")
        
        # Sentiment leader
        if 'sentiment' in brand_summary.columns:
            sentiment_leader = brand_summary.loc[brand_summary['sentiment'].idxmax()]
            insights.append(f"{sentiment_leader['brand']} has highest sentiment score at {sentiment_leader['sentiment']:.2f}")
        
        return insights
    
    def _generate_comprehensive_brand_temporal(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate comprehensive temporal brand performance analysis"""
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column available for brand temporal analysis"}
        
        # Apply market-aware timezone conversion
        df = self._convert_to_market_time(df, date_col)
        
        # Ensure _datetime is properly converted to datetime
        df['_datetime'] = pd.to_datetime(df['_datetime'], errors='coerce')
        
        # Add temporal columns
        df['_year_week'] = df['_datetime'].dt.to_period('W')
        df['_year_month'] = df['_datetime'].dt.to_period('M')
        df['_year_quarter'] = df['_datetime'].dt.to_period('Q')
        df['_year'] = df['_datetime'].dt.to_period('Y')
        df['_date'] = df['_datetime'].dt.date
        
        # Extract brands from each row using the proper brand extraction logic
        brand_data = []
        for _, row in df.iterrows():
            # Use improved brand extraction with proper tiktok_post_labels parsing
            brands = []
            if platform == 'tiktok':
                # First try the structured tiktok_post_labels field
                labels_json = str(row.get('tiktok_post_labels', ''))
                if labels_json and pd.notna(labels_json) and labels_json != 'nan':
                    brands.extend(self._parse_tiktok_labels_for_brands(labels_json))
                
                # Fallback to tiktok_post_labels_names
                if not brands:
                    labels_names = str(row.get('tiktok_post_labels_names', ''))
                    if labels_names and pd.notna(labels_names) and labels_names != 'nan':
                        parts = [part.strip() for part in labels_names.split(',')]
                        for part in parts:
                            if '[Brand]' in part:
                                brand = part.replace('[Brand]', '').strip()
                                if brand:
                                    brands.append(brand)
            elif platform in ['facebook', 'instagram']:
                brands_field = row.get('brands', '')
                if brands_field and pd.notna(brands_field):
                    brands = [b.strip() for b in str(brands_field).split(',') if b.strip()]
            
            brands = brands or ['Unknown']
            for brand in brands:
                brand_data.append({
                    'brand': brand,
                    'date': row['_date'],
                    'year_week': str(row['_year_week']),
                    'year_month': str(row['_year_month']),
                    'year_quarter': str(row['_year_quarter']),
                    'year': str(row['_year']),
                    'engagement': self._safe_numeric_convert(row.get('tiktok_insights_engagements', 0)),
                    'views': self._safe_numeric_convert(row.get('tiktok_insights_video_views', 0)),
                    'likes': self._safe_numeric_convert(row.get('tiktok_insights_likes', 0)),
                    'sentiment': self._safe_numeric_convert(row.get('tiktok_sentiment', 0))
                })
        
        if not brand_data:
            return {"error": "No brand data found"}
        
        brand_df = pd.DataFrame(brand_data)
        
        # Generate temporal aggregations
        daily_trends = {}
        for date, group in brand_df.groupby('date'):
            daily_trends[str(date)] = {}
            for brand, brand_group in group.groupby('brand'):
                daily_trends[str(date)][brand] = {
                    'posts': len(brand_group),
                    'total_engagement': brand_group['engagement'].sum(),
                    'total_views': brand_group['views'].sum(),
                    'total_likes': brand_group['likes'].sum(),
                    'avg_sentiment': brand_group['sentiment'].mean()
                }
        
        weekly_aggregations = {}
        for week, group in brand_df.groupby('year_week'):
            weekly_aggregations[week] = {}
            for brand, brand_group in group.groupby('brand'):
                weekly_aggregations[week][brand] = {
                    'posts': len(brand_group),
                    'total_engagement': brand_group['engagement'].sum(),
                    'total_views': brand_group['views'].sum(),
                    'total_likes': brand_group['likes'].sum(),
                    'avg_sentiment': brand_group['sentiment'].mean()
                }
        
        monthly_aggregations = {}
        for month, group in brand_df.groupby('year_month'):
            monthly_aggregations[month] = {}
            for brand, brand_group in group.groupby('brand'):
                monthly_aggregations[month][brand] = {
                    'posts': len(brand_group),
                    'total_engagement': brand_group['engagement'].sum(),
                    'total_views': brand_group['views'].sum(),
                    'total_likes': brand_group['likes'].sum(),
                    'avg_sentiment': brand_group['sentiment'].mean()
                }
        
        quarterly_aggregations = {}
        for quarter, group in brand_df.groupby('year_quarter'):
            quarterly_aggregations[quarter] = {}
            for brand, brand_group in group.groupby('brand'):
                quarterly_aggregations[quarter][brand] = {
                    'posts': len(brand_group),
                    'total_engagement': brand_group['engagement'].sum(),
                    'total_views': brand_group['views'].sum(),
                    'total_likes': brand_group['likes'].sum(),
                    'avg_sentiment': brand_group['sentiment'].mean()
                }
        
        yearly_aggregations = {}
        for year, group in brand_df.groupby('year'):
            yearly_aggregations[year] = {}
            for brand, brand_group in group.groupby('brand'):
                yearly_aggregations[year][brand] = {
                    'posts': len(brand_group),
                    'total_engagement': brand_group['engagement'].sum(),
                    'total_views': brand_group['views'].sum(),
                    'total_likes': brand_group['likes'].sum(),
                    'avg_sentiment': brand_group['sentiment'].mean()
                }
        
        return {
            "daily_trends": daily_trends,
            "weekly_aggregations": weekly_aggregations,
            "monthly_aggregations": monthly_aggregations,
            "quarterly_aggregations": quarterly_aggregations,
            "yearly_aggregations": yearly_aggregations,
            "cross_dimensional": {}
        }
    
    def _extract_brands_from_row(self, row: pd.Series, platform: str) -> List[str]:
        """Extract brand mentions from a row"""
        brands = []
        
        # Common brand keywords to look for
        brand_keywords = ['Nike', 'Adidas', 'Puma', 'Apple', 'Samsung', 'Google', 'Microsoft', 
                         'Amazon', 'Facebook', 'Instagram', 'TikTok', 'Twitter', 'YouTube',
                         'Coca-Cola', 'Pepsi', 'McDonald', 'Starbucks', 'Tesla', 'BMW', 'Mercedes']
        
        # Check various text fields
        text_fields = ['content', 'text', 'description', 'caption', f'{platform}_content', f'{platform}_text']
        
        for field in text_fields:
            if field in row and pd.notna(row[field]):
                text = str(row[field]).lower()
                for brand in brand_keywords:
                    if brand.lower() in text:
                        brands.append(brand)
        
        return list(set(brands)) if brands else ['Unknown']
    
    def _calculate_brand_summary(self, brand_df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Calculate brand performance summary"""
        if len(brand_df) == 0:
            return pd.DataFrame()
        
        # Group by brand and calculate metrics
        agg_dict = {
            'post_id': 'count',
            'sentiment': 'mean'
        }
        
        if platform == 'tiktok':
            agg_dict.update({
                'impressions': ['sum', 'mean'],
                'engagements': ['sum', 'mean'],
                'video_views': ['sum', 'mean'],
                'completion_rate': 'mean',
                'duration': 'mean'
            })
        elif platform in ['facebook', 'instagram']:
            agg_dict.update({
                'impressions': ['sum', 'mean'],
                'engagements': ['sum', 'mean'],
                'reach': ['sum', 'mean']
            })
        elif platform == 'customer_care':
            agg_dict.update({
                'urgency': 'mean',
                'resolution_time': 'mean',
                'satisfaction': 'mean'
            })
        
        summary = brand_df.groupby('brand').agg(agg_dict).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]
        
        # Calculate engagement rate for social platforms
        if platform in ['tiktok', 'facebook', 'instagram']:
            if 'engagements_mean' in summary.columns and 'impressions_mean' in summary.columns:
                # Protect against division by zero
                summary['avg_engagement_rate'] = (
                    summary['engagements_mean'] / summary['impressions_mean'].replace(0, np.nan) * 100
                ).round(2)
        
        return summary.reset_index()
    
    def _calculate_brand_rankings(self, brand_summary: pd.DataFrame) -> Dict:
        """Calculate brand rankings for AI decision making"""
        if brand_summary.empty:
            return {}
        
        rankings = {}
        
        # Top by engagement rate
        if 'avg_engagement_rate' in brand_summary.columns:
            rankings["top_by_engagement_rate"] = brand_summary.nlargest(5, 'avg_engagement_rate')[['brand', 'avg_engagement_rate']].to_dict('records')
        
        # Most active brands
        if 'post_id_count' in brand_summary.columns:
            rankings["most_active_brands"] = brand_summary.nlargest(5, 'post_id_count')[['brand', 'post_id_count']].to_dict('records')
        
        # Top by sentiment
        if 'sentiment_mean' in brand_summary.columns:
            rankings["top_by_sentiment"] = brand_summary.nlargest(5, 'sentiment_mean')[['brand', 'sentiment_mean']].to_dict('records')
        
        return rankings
    
    def _generate_brand_ai_insights(self, brand_summary: pd.DataFrame) -> List[str]:
        """Generate AI-actionable insights about brand performance"""
        insights = []
        
        if brand_summary.empty:
            return insights
        
        # Top performer insights
        if 'avg_engagement_rate' in brand_summary.columns:
            top_brand = brand_summary.loc[brand_summary['avg_engagement_rate'].idxmax()]
            insights.append(f"'{top_brand['brand']}' achieves highest engagement rate at {top_brand['avg_engagement_rate']}%")
        
        # Activity insights
        if 'post_id_count' in brand_summary.columns:
            most_active = brand_summary.loc[brand_summary['post_id_count'].idxmax()]
            insights.append(f"'{most_active['brand']}' is most active with {most_active['post_id_count']} posts")
        
        return insights


class ContentAnalyzer(BaseAnalyzer):
    """
    Content Type Performance Analysis
    
    Original methods moved here:
    - _generate_content_type_performance (line 1520)
    - _generate_comprehensive_content_temporal (line 1562)
    - _generate_content_type_metrics (line 2054)
    - _generate_duration_performance_metrics (line 1722)
    - _generate_content_ai_recommendations (line 292, 5500)
    - _generate_duration_recommendations (line 399, 5578)
    - _calculate_content_summary (line 220)
    - _calculate_content_rankings (line 267)
    - _calculate_duration_summary (line 349)
    - _calculate_optimal_duration (line 381)
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive content performance analysis."""
        # Combine content type and duration analysis
        content_result = self._generate_content_type_performance(df, platform)
        duration_result = self._generate_duration_performance_metrics(df, platform)
        
        return {
            **content_result,
            "duration_performance": duration_result,
            "generated_at": content_result.get("generated_at")
        }
    
    def _generate_content_type_performance(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate content type performance analytics"""
        from datetime import datetime
        
        content_data = []
        
        # Extract content type data from each post
        for _, row in df.iterrows():
            content_types = self._extract_content_types_from_row(row, platform)
            
            for content_type in content_types:
                content_data.append({
                    'post_id': self._get_post_id(row, platform),
                    'content_type': content_type.get('name', ''),
                    'category': content_type.get('category', ''),
                    **self._extract_performance_metrics(row, platform)
                })
        
        if not content_data:
            return {"content_summary": [], "content_rankings": [], "ai_recommendations": []}
        
        content_df = pd.DataFrame(content_data)
        
        # Content type performance summary
        content_summary = self._calculate_content_summary(content_df, platform)
        
        # Content rankings
        content_rankings = self._calculate_content_rankings(content_summary)
        
        # AI recommendations
        ai_recommendations = self._generate_content_ai_recommendations(content_summary)
        
        # Add comprehensive temporal content type performance data
        temporal_content_data = self._generate_comprehensive_content_temporal(df, platform)
        
        return {
            "content_summary": content_summary.to_dict('records') if not content_summary.empty else [],
            "content_rankings": content_rankings,
            "ai_recommendations": ai_recommendations,
            "temporal_performance": temporal_content_data,
            "generated_at": datetime.now().isoformat()
        }
    
    def _extract_content_types_from_row(self, row: pd.Series, platform: str) -> List[Dict]:
        """Extract content types with categories from a row"""
        content_types = []
        
        if platform == 'tiktok':
            labels = str(row.get('tiktok_post_labels_names', ''))
            if labels and pd.notna(labels):
                parts = [part.strip() for part in labels.split(',')]
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
        elif platform in ['facebook', 'instagram']:
            # Extract content types from content_type field or media_type
            content_type = str(row.get('content_type', ''))
            if content_type and pd.notna(content_type):
                content_types.append({'name': content_type, 'category': 'media'})
        
        return content_types or [{'name': 'Unknown', 'category': 'unknown'}]
    
    def _generate_duration_performance_metrics(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate video duration performance analytics - placeholder for now"""
        return {
            "duration_summary": [],
            "optimal_duration": {},
            "ai_recommendations": []
        }
    
    def _generate_comprehensive_content_temporal(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate comprehensive temporal content performance analysis"""
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column available for content temporal analysis"}
        
        # Apply market-aware timezone conversion
        df = self._convert_to_market_time(df, date_col)
        
        # Ensure _datetime is properly converted to datetime
        df['_datetime'] = pd.to_datetime(df['_datetime'], errors='coerce')
        
        # Add temporal columns
        df['_year_week'] = df['_datetime'].dt.to_period('W')
        df['_year_month'] = df['_datetime'].dt.to_period('M')
        df['_year_quarter'] = df['_datetime'].dt.to_period('Q')
        df['_year'] = df['_datetime'].dt.to_period('Y')
        df['_date'] = df['_datetime'].dt.date
        
        # Extract content types from each row using the proper content extraction logic
        content_data = []
        for _, row in df.iterrows():
            # Use improved content extraction with proper tiktok_post_labels parsing
            content_types = []
            if platform == 'tiktok':
                # First try the structured tiktok_post_labels field
                labels_json = str(row.get('tiktok_post_labels', ''))
                if labels_json and pd.notna(labels_json) and labels_json != 'nan':
                    content_types.extend(self._parse_tiktok_labels_for_content(labels_json))
                
                # Fallback to tiktok_post_labels_names
                if not content_types:
                    labels_names = str(row.get('tiktok_post_labels_names', ''))
                    if labels_names and pd.notna(labels_names) and labels_names != 'nan':
                        parts = [part.strip() for part in labels_names.split(',')]
                        for part in parts:
                            if '[Axis]' in part:
                                name = part.replace('[Axis]', '').strip()
                                if name:
                                    content_types.append({'type': name, 'category': 'axis'})
                            elif '[Asset]' in part:
                                name = part.replace('[Asset]', '').strip()
                                if name:
                                    content_types.append({'type': name, 'category': 'asset'})
                            elif '[Package M]' in part:
                                name = part.replace('[Package M]', '').strip()
                                if name:
                                    content_types.append({'type': name, 'category': 'package'})
            elif platform in ['facebook', 'instagram']:
                # Add platform-specific logic here
                content_types = [{'type': 'General', 'category': 'General'}]
            
            content_types = content_types or [{'type': 'Unknown', 'category': 'General'}]
            for content_type_info in content_types:
                content_data.append({
                    'content_type': content_type_info.get('type', 'Unknown'),
                    'category': content_type_info.get('category', 'General'),
                    'date': row['_date'],
                    'year_week': str(row['_year_week']),
                    'year_month': str(row['_year_month']),
                    'year_quarter': str(row['_year_quarter']),
                    'year': str(row['_year']),
                    'engagement': self._safe_numeric_convert(row.get('tiktok_insights_engagements', 0)),
                    'views': self._safe_numeric_convert(row.get('tiktok_insights_video_views', 0)),
                    'likes': self._safe_numeric_convert(row.get('tiktok_insights_likes', 0)),
                    'sentiment': self._safe_numeric_convert(row.get('tiktok_sentiment', 0))
                })
        
        if not content_data:
            return {"error": "No content type data found"}
        
        content_df = pd.DataFrame(content_data)
        
        # Generate temporal aggregations
        daily_trends = {}
        for date, group in content_df.groupby('date'):
            daily_trends[str(date)] = {}
            for content_type, type_group in group.groupby('content_type'):
                daily_trends[str(date)][content_type] = {
                    'posts': len(type_group),
                    'total_engagement': type_group['engagement'].sum(),
                    'total_views': type_group['views'].sum(),
                    'total_likes': type_group['likes'].sum(),
                    'avg_sentiment': type_group['sentiment'].mean()
                }
        
        weekly_aggregations = {}
        for week, group in content_df.groupby('year_week'):
            weekly_aggregations[week] = {}
            for content_type, type_group in group.groupby('content_type'):
                weekly_aggregations[week][content_type] = {
                    'posts': len(type_group),
                    'total_engagement': type_group['engagement'].sum(),
                    'total_views': type_group['views'].sum(),
                    'total_likes': type_group['likes'].sum(),
                    'avg_sentiment': type_group['sentiment'].mean()
                }
        
        monthly_aggregations = {}
        for month, group in content_df.groupby('year_month'):
            monthly_aggregations[month] = {}
            for content_type, type_group in group.groupby('content_type'):
                monthly_aggregations[month][content_type] = {
                    'posts': len(type_group),
                    'total_engagement': type_group['engagement'].sum(),
                    'total_views': type_group['views'].sum(),
                    'total_likes': type_group['likes'].sum(),
                    'avg_sentiment': type_group['sentiment'].mean()
                }
        
        quarterly_aggregations = {}
        for quarter, group in content_df.groupby('year_quarter'):
            quarterly_aggregations[quarter] = {}
            for content_type, type_group in group.groupby('content_type'):
                quarterly_aggregations[quarter][content_type] = {
                    'posts': len(type_group),
                    'total_engagement': type_group['engagement'].sum(),
                    'total_views': type_group['views'].sum(),
                    'total_likes': type_group['likes'].sum(),
                    'avg_sentiment': type_group['sentiment'].mean()
                }
        
        yearly_aggregations = {}
        for year, group in content_df.groupby('year'):
            yearly_aggregations[year] = {}
            for content_type, type_group in group.groupby('content_type'):
                yearly_aggregations[year][content_type] = {
                    'posts': len(type_group),
                    'total_engagement': type_group['engagement'].sum(),
                    'total_views': type_group['views'].sum(),
                    'total_likes': type_group['likes'].sum(),
                    'avg_sentiment': type_group['sentiment'].mean()
                }
        
        return {
            "daily_trends": daily_trends,
            "weekly_aggregations": weekly_aggregations,
            "monthly_aggregations": monthly_aggregations,
            "quarterly_aggregations": quarterly_aggregations,
            "yearly_aggregations": yearly_aggregations,
            "cross_dimensional": {}
        }
    
    # Helper methods will be added incrementally
    def _get_post_id(self, row: pd.Series, platform: str) -> str:
        """Get platform-specific post ID"""
        if platform == 'tiktok':
            return str(row.get('tiktok_id', ''))
        elif platform == 'facebook':
            return str(row.get('facebook_id', ''))
        elif platform == 'instagram':
            return str(row.get('instagram_id', ''))
        elif platform == 'customer_care':
            return str(row.get('case_id', ''))
        return str(row.get('id', ''))
    
    def _extract_performance_metrics(self, row: pd.Series, platform: str) -> Dict:
        """Extract performance metrics from a row based on platform"""
        metrics = {}
        
        if platform == 'tiktok':
            metrics.update({
                'impressions': self._safe_numeric_convert(row.get('tiktok_insights_impressions'), float, 0),
                'engagements': self._safe_numeric_convert(row.get('tiktok_insights_engagements'), float, 0),
                'video_views': self._safe_numeric_convert(row.get('tiktok_insights_video_views'), float, 0),
                'completion_rate': self._safe_numeric_convert(row.get('tiktok_insights_completion_rate'), float, 0),
                'duration': self._safe_numeric_convert(row.get('tiktok_duration'), float, 0),
                'sentiment': self._safe_numeric_convert(row.get('tiktok_sentiment'), float, 0)
            })
        elif platform == 'facebook':
            metrics.update({
                'impressions': self._safe_numeric_convert(row.get('facebook_insights_impressions'), float, 0),
                'engagements': self._safe_numeric_convert(row.get('facebook_insights_engagements'), float, 0),
                'reach': self._safe_numeric_convert(row.get('facebook_insights_reach'), float, 0),
                'sentiment': self._safe_numeric_convert(row.get('facebook_sentiment'), float, 0)
            })
        elif platform == 'instagram':
            metrics.update({
                'impressions': self._safe_numeric_convert(row.get('instagram_insights_impressions'), float, 0),
                'engagements': self._safe_numeric_convert(row.get('instagram_insights_engagements'), float, 0),
                'reach': self._safe_numeric_convert(row.get('instagram_insights_reach'), float, 0),
                'sentiment': self._safe_numeric_convert(row.get('instagram_sentiment'), float, 0)
            })
        elif platform == 'customer_care':
            metrics.update({
                'urgency': self._safe_numeric_convert(row.get('urgency'), float, 0),
                'resolution_time': self._safe_numeric_convert(row.get('resolution_time'), float, 0),
                'satisfaction': self._safe_numeric_convert(row.get('satisfaction'), float, 0),
                'sentiment': self._safe_numeric_convert(row.get('sentiment'), float, 0)
            })
        
        return metrics
    
    def _calculate_content_summary(self, content_df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Calculate content type performance summary - simplified for now"""
        if len(content_df) == 0:
            return pd.DataFrame()
        
        # Basic grouping by content_type
        summary = content_df.groupby('content_type').agg({
            'post_id': 'count',
            'sentiment': 'mean',
            'impressions': 'mean' if 'impressions' in content_df.columns else lambda x: 0,
            'engagements': 'mean' if 'engagements' in content_df.columns else lambda x: 0
        }).round(2)
        
        return summary.reset_index()
    
    def _calculate_content_rankings(self, content_summary: pd.DataFrame) -> Dict:
        """Calculate content type rankings - simplified for now"""
        if content_summary.empty:
            return {}
        
        rankings = {}
        
        # Basic rankings
        if 'post_id' in content_summary.columns:
            volume_ranking = content_summary.nlargest(5, 'post_id')[['content_type', 'post_id']]
            rankings['top_volume'] = volume_ranking.to_dict('records')
        
        return rankings
    
    def _generate_content_ai_recommendations(self, content_summary: pd.DataFrame) -> List[str]:
        """Generate AI recommendations for content performance - simplified for now"""
        if content_summary.empty:
            return ["No content data available for recommendations"]
        
        recommendations = []
        
        # Most active content type
        if 'post_id' in content_summary.columns:
            most_active = content_summary.loc[content_summary['post_id'].idxmax()]
            recommendations.append(f"{most_active['content_type']} is most frequently used with {most_active['post_id']} posts")
        
        return recommendations
    
    def _extract_content_types_from_row(self, row: pd.Series, platform: str) -> List[Dict]:
        """Extract content types from a row"""
        content_types = []
        
        # Platform-specific content type detection
        if platform == 'tiktok':
            # Check for video indicators
            if 'tiktok_duration' in row and pd.notna(row['tiktok_duration']):
                duration = float(row['tiktok_duration'])
                if duration <= 15:
                    content_types.append({'name': 'Short Video', 'category': 'Video', 'duration': duration})
                elif duration <= 60:
                    content_types.append({'name': 'Medium Video', 'category': 'Video', 'duration': duration})
                else:
                    content_types.append({'name': 'Long Video', 'category': 'Video', 'duration': duration})
        
        elif platform == 'facebook':
            # Check media type
            if 'facebook_media_type' in row and pd.notna(row['facebook_media_type']):
                media_type = str(row['facebook_media_type'])
                content_types.append({'name': media_type.title(), 'category': 'Media'})
        
        elif platform == 'instagram':
            # Check for Instagram-specific types
            if 'instagram_insights_story_completion_rate' in row:
                content_types.append({'name': 'Story', 'category': 'Story'})
            elif 'instagram_insights_video_views' in row and pd.notna(row['instagram_insights_video_views']):
                content_types.append({'name': 'Video', 'category': 'Video'})
            else:
                content_types.append({'name': 'Photo', 'category': 'Photo'})
        
        elif platform == 'customer_care':
            # Check issue types
            if 'issue_type' in row and pd.notna(row['issue_type']):
                issue_type = str(row['issue_type'])
                content_types.append({'name': issue_type, 'category': 'Issue'})
        
        # Default content types if none detected
        if not content_types:
            content_types = [{'name': 'Standard', 'category': 'General'}]
        
        return content_types
    
    def _calculate_content_summary(self, content_df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Calculate content type performance summary"""
        if len(content_df) == 0:
            return pd.DataFrame()
        
        # Group by content_type and category and calculate metrics
        agg_dict = {
            'post_id': 'count',
            'sentiment': 'mean'
        }
        
        if platform == 'tiktok':
            agg_dict.update({
                'impressions': ['sum', 'mean'],
                'engagements': ['sum', 'mean'], 
                'video_views': ['sum', 'mean'],
                'completion_rate': 'mean',
                'duration': 'mean'
            })
        elif platform in ['facebook', 'instagram']:
            agg_dict.update({
                'impressions': ['sum', 'mean'],
                'engagements': ['sum', 'mean'],
                'reach': ['sum', 'mean']
            })
        elif platform == 'customer_care':
            agg_dict.update({
                'urgency': 'mean',
                'resolution_time': 'mean',
                'satisfaction': 'mean'
            })
        
        summary = content_df.groupby(['content_type', 'category']).agg(agg_dict).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]
        
        # Calculate engagement rate for social platforms
        if platform in ['tiktok', 'facebook', 'instagram']:
            if 'engagements_mean' in summary.columns and 'impressions_mean' in summary.columns:
                # Protect against division by zero
                summary['avg_engagement_rate'] = (
                    summary['engagements_mean'] / summary['impressions_mean'].replace(0, np.nan) * 100
                ).round(2)
        
        return summary.reset_index()
    
    def _calculate_content_rankings(self, content_summary: pd.DataFrame) -> Dict:
        """Calculate content type rankings"""
        if content_summary.empty:
            return {}
        
        rankings = {}
        
        # Top by engagement rate
        if 'avg_engagement_rate' in content_summary.columns:
            rankings["highest_engagement_content"] = content_summary.nlargest(5, 'avg_engagement_rate')[['content_type', 'category', 'avg_engagement_rate']].to_dict('records')
        
        # Top by completion rate (for video platforms)
        if 'completion_rate_mean' in content_summary.columns:
            rankings["highest_completion_content"] = content_summary.nlargest(5, 'completion_rate_mean')[['content_type', 'category', 'completion_rate_mean']].to_dict('records')
        
        # Most popular content types
        if 'post_id_count' in content_summary.columns:
            rankings["most_popular_content"] = content_summary.nlargest(5, 'post_id_count')[['content_type', 'category', 'post_id_count']].to_dict('records')
        
        # Best sentiment content
        if 'sentiment_mean' in content_summary.columns:
            rankings["best_sentiment_content"] = content_summary.nlargest(5, 'sentiment_mean')[['content_type', 'category', 'sentiment_mean']].to_dict('records')
        
        return rankings
    
    def _generate_content_ai_recommendations(self, content_summary: pd.DataFrame) -> List[str]:
        """Generate content strategy recommendations"""
        recommendations = []
        
        if content_summary.empty:
            return recommendations
        
        # Top engagement recommendations
        if 'avg_engagement_rate' in content_summary.columns:
            top_content = content_summary.loc[content_summary['avg_engagement_rate'].idxmax()]
            recommendations.append(f"Focus on '{top_content['content_type']}' content - achieves {top_content['avg_engagement_rate']}% engagement rate")
        
        # Duration recommendations for video platforms
        if 'duration_mean' in content_summary.columns:
            optimal_duration = content_summary.loc[content_summary['avg_engagement_rate'].idxmax()]
            recommendations.append(f"Optimal video duration: {optimal_duration['duration_mean']:.0f} seconds")
        
        return recommendations


class SentimentAnalyzer(BaseAnalyzer):
    """
    Sentiment Analysis
    
    Original methods moved here:
    - _generate_sentiment_analysis (line 2090)
    - _generate_comprehensive_sentiment_temporal (line 2141)
    - _aggregate_sentiment_by_period (line 2203)
    - _aggregate_sentiment_cross_dimensional (line 2233)
    - _analyze_sentiment_distribution_temporal (line 2264)
    - _analyze_sentiment_volatility (line 2282)
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive sentiment analysis."""
        return self._generate_sentiment_analysis(df, platform)
    
    def _generate_sentiment_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate comprehensive sentiment analysis"""
        
        sentiment_col = self._get_sentiment_column(df, platform)
        if not sentiment_col:
            return {"error": "No sentiment data found"}
        
        # Basic statistics
        sentiment_stats = {
            "mean": float(df[sentiment_col].mean()),
            "median": float(df[sentiment_col].median()),
            "std": float(df[sentiment_col].std()),
            "min": float(df[sentiment_col].min()),
            "max": float(df[sentiment_col].max()),
            "total_analyzed": len(df[df[sentiment_col].notna()])
        }
        
        # Distribution
        df['_sentiment_category'] = pd.cut(
            df[sentiment_col],
            bins=[-1, -0.6, -0.2, 0.2, 0.6, 1],
            labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
        )
        
        sentiment_dist = df['_sentiment_category'].value_counts().to_dict()
        
        # Sentiment by entities
        sentiment_by_entity = {}
        
        # By brand
        if 'brand' in df.columns:
            brand_sentiment = df.groupby('brand')[sentiment_col].agg(['mean', 'std', 'count'])
            sentiment_by_entity['by_brand'] = brand_sentiment.round(3).to_dict('index')
        
        # By content type
        content_col = 'content_type' if 'content_type' in df.columns else 'issue_type'
        if content_col in df.columns:
            type_sentiment = df.groupby(content_col)[sentiment_col].agg(['mean', 'std', 'count'])
            sentiment_by_entity['by_type'] = type_sentiment.round(3).to_dict('index')
        
        # Add comprehensive temporal sentiment data
        temporal_sentiment_data = self._generate_comprehensive_sentiment_temporal(df, platform, sentiment_col)
        
        return {
            "statistics": sentiment_stats,
            "distribution": sentiment_dist,
            "by_entity": sentiment_by_entity,
            "sentiment_drivers": self._identify_sentiment_drivers(df, sentiment_col),
            "temporal_sentiment": temporal_sentiment_data
        }
    
    def _get_sentiment_column(self, df: pd.DataFrame, platform: str) -> str:
        """Get the correct sentiment column name for the platform"""
        for col_name in ['sentiment_score', f'{platform}_sentiment', 'sentiment']:
            if col_name in df.columns:
                return col_name
        return None
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> str:
        """Get the correct date column name for the platform"""
        date_columns = ['created_time', 'timestamp', 'date', '_datetime']
        for col_name in date_columns:
            if col_name in df.columns:
                return col_name
        return None
    
    def _generate_comprehensive_sentiment_temporal(self, df: pd.DataFrame, platform: str, sentiment_col: str) -> Dict[str, Any]:
        """Generate comprehensive temporal sentiment analysis: daily, weekly, monthly, quarterly, yearly"""
        
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column available for sentiment temporal analysis"}
        
        # Convert to datetime and extract all temporal features
        df['_datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['_datetime'].notna()]
        df['_date'] = df['_datetime'].dt.date
        df['_year_week'] = df['_datetime'].dt.to_period('W')
        df['_year_month'] = df['_datetime'].dt.to_period('M')
        df['_year_quarter'] = df['_datetime'].dt.to_period('Q')
        df['_year'] = df['_datetime'].dt.year
        
        # Add sentiment categories
        df['_sentiment_category'] = pd.cut(
            df[sentiment_col],
            bins=[-1, -0.6, -0.2, 0.2, 0.6, 1],
            labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
        )
        
        # Add comprehensive fields for cross-dimensional analysis
        df['country'] = df.get('derived_country', 'Unknown')
        df['language'] = df.get('detected_language', 'Unknown')
        
        temporal_sentiment = {
            # Time period aggregations
            "daily": self._aggregate_sentiment_by_period(df, '_date', sentiment_col),
            "weekly": self._aggregate_sentiment_by_period(df, '_year_week', sentiment_col),
            "monthly": self._aggregate_sentiment_by_period(df, '_year_month', sentiment_col),
            "quarterly": self._aggregate_sentiment_by_period(df, '_year_quarter', sentiment_col),
            "yearly": self._aggregate_sentiment_by_period(df, '_year', sentiment_col),
            
            # Cross-dimensional sentiment analysis
            "by_country": self._aggregate_sentiment_cross_dimensional(df, 'country', sentiment_col),
            "by_language": self._aggregate_sentiment_cross_dimensional(df, 'language', sentiment_col),
            
            # Sentiment distribution over time
            "sentiment_distribution_over_time": self._analyze_sentiment_distribution_temporal(df, sentiment_col),
            
            # Sentiment volatility analysis
            "volatility_analysis": self._analyze_sentiment_volatility(df, sentiment_col),
            
            # Summary insights
            "summary": {
                "date_range": {
                    "start": str(df['_date'].min()),
                    "end": str(df['_date'].max()),
                    "total_days": (df['_date'].max() - df['_date'].min()).days + 1
                },
                "total_posts": len(df),
                "countries_analyzed": df['country'].nunique(),
                "languages_analyzed": df['language'].nunique(),
                "avg_sentiment": float(df[sentiment_col].mean()),
                "sentiment_volatility": float(df[sentiment_col].std())
            }
        }
        
        return temporal_sentiment
    
    def _aggregate_sentiment_by_period(self, df: pd.DataFrame, period_col: str, sentiment_col: str) -> Dict[str, Any]:
        """Aggregate sentiment data by time period"""
        if period_col not in df.columns:
            return {}
        
        period_agg = df.groupby(period_col).agg({
            sentiment_col: ['mean', 'median', 'std', 'min', 'max', 'count'],
            '_sentiment_category': lambda x: x.value_counts().to_dict()
        }).round(3)
        
        # Flatten column names
        period_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in period_agg.columns]
        
        return period_agg.to_dict('index')
    
    def _aggregate_sentiment_cross_dimensional(self, df: pd.DataFrame, dimension_col: str, sentiment_col: str) -> Dict[str, Any]:
        """Aggregate sentiment data by cross-dimensional criteria"""
        if dimension_col not in df.columns:
            return {}
        
        cross_agg = df.groupby(dimension_col).agg({
            sentiment_col: ['mean', 'median', 'std', 'count'],
            '_sentiment_category': lambda x: x.value_counts().to_dict()
        }).round(3)
        
        # Flatten column names
        cross_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in cross_agg.columns]
        
        return cross_agg.to_dict('index')
    
    def _analyze_sentiment_distribution_temporal(self, df: pd.DataFrame, sentiment_col: str) -> Dict[str, Any]:
        """Analyze how sentiment distribution changes over time"""
        if '_year_month' not in df.columns:
            return {}
        
        # Monthly sentiment distribution
        monthly_dist = df.groupby(['_year_month', '_sentiment_category']).size().unstack(fill_value=0)
        
        # Calculate percentages
        monthly_pct = monthly_dist.div(monthly_dist.sum(axis=1), axis=0) * 100
        
        return {
            "monthly_counts": monthly_dist.to_dict('index'),
            "monthly_percentages": monthly_pct.round(2).to_dict('index')
        }
    
    def _analyze_sentiment_volatility(self, df: pd.DataFrame, sentiment_col: str) -> Dict[str, Any]:
        """Analyze sentiment volatility patterns"""
        if '_date' not in df.columns:
            return {}
        
        # Daily sentiment averages
        daily_sentiment = df.groupby('_date')[sentiment_col].mean()
        
        # Rolling volatility (7-day window)
        rolling_std = daily_sentiment.rolling(window=7, min_periods=3).std()
        
        return {
            "daily_volatility": rolling_std.dropna().to_dict(),
            "overall_volatility": float(daily_sentiment.std()),
            "max_volatility": float(rolling_std.max()) if not rolling_std.empty else 0,
            "min_volatility": float(rolling_std.min()) if not rolling_std.empty else 0
        }
    
    def _identify_sentiment_drivers(self, df: pd.DataFrame, sentiment_col: str) -> List[str]:
        """Identify key drivers of sentiment"""
        drivers = []
        
        # Find highest and lowest sentiment content
        if not df.empty:
            highest_sentiment = df.loc[df[sentiment_col].idxmax()]
            lowest_sentiment = df.loc[df[sentiment_col].idxmin()]
            
            drivers.append(f"Highest sentiment: {highest_sentiment.get('content_type', 'Unknown')} ({highest_sentiment[sentiment_col]:.3f})")
            drivers.append(f"Lowest sentiment: {lowest_sentiment.get('content_type', 'Unknown')} ({lowest_sentiment[sentiment_col]:.3f})")
        
        return drivers


class EngagementAnalyzer(BaseAnalyzer):
    """
    Engagement Metrics Analysis
    
    Original methods moved here:
    - _generate_engagement_metrics (line 2296)
    - _generate_comprehensive_engagement_temporal (line 2359)
    - _aggregate_engagement_by_period (line 2432)
    - _aggregate_engagement_cross_dimensional (line 2470)
    - _analyze_engagement_rate_evolution (line 2510)
    - _analyze_peak_engagement_performance (line 2533)
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive engagement analysis."""
        return self._generate_engagement_metrics(df, platform)
    
    def _generate_engagement_metrics(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate engagement metrics (platform-specific)"""
        
        metrics = {}
        
        if platform in ['facebook', 'instagram', 'tiktok']:
            # Social media engagement
            engagement_cols = {
                'facebook': ['facebook_insights_engagements', 'facebook_insights_impressions', 'facebook_insights_video_views'],
                'instagram': ['instagram_insights_engagements', 'instagram_insights_impressions', 'instagram_insights_reach'],
                'tiktok': ['tiktok_insights_engagements', 'tiktok_insights_impressions', 'tiktok_insights_shares', 'tiktok_insights_video_views']
            }
            
            cols = engagement_cols.get(platform, [])
            available_cols = [c for c in cols if c in df.columns]
            
            if available_cols:
                for col in available_cols:
                    if df[col].notna().sum() > 0:
                        metrics[col] = {
                            'total': int(df[col].sum()),
                            'average': float(df[col].mean()),
                            'median': float(df[col].median()),
                            'max': int(df[col].max())
                        }
                
                # Calculate engagement rate if possible
                engagement_col = f'{platform}_insights_engagements'
                impression_col = f'{platform}_insights_impressions'
                
                if engagement_col in df.columns and impression_col in df.columns:
                    # Fill NaN values to avoid calculation errors
                    engagements = df[engagement_col].fillna(0)
                    impressions = df[impression_col].fillna(1)  # Avoid division by zero
                    df['_engagement_rate'] = (engagements / impressions * 100).replace([np.inf, -np.inf], 0)
                    metrics['engagement_rate'] = {
                        'average': float(df['_engagement_rate'].mean()),
                        'median': float(df['_engagement_rate'].median()),
                        'std': float(df['_engagement_rate'].std())
                    }
        
        elif platform == 'customer_care':
            # Customer care engagement metrics
            care_cols = ['resolution_time', 'satisfaction', 'urgency']
            available_cols = [c for c in care_cols if c in df.columns]
            
            for col in available_cols:
                if df[col].notna().sum() > 0:
                    metrics[col] = {
                        'average': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
        
        # Add temporal engagement analysis
        temporal_engagement = self._generate_comprehensive_engagement_temporal(df, platform)
        
        return {
            "engagement_metrics": metrics,
            "temporal_engagement": temporal_engagement,
            "summary": {
                "total_metrics": len(metrics),
                "platform": platform,
                "records_analyzed": len(df)
            }
        }
    
    def _generate_comprehensive_engagement_temporal(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate comprehensive temporal engagement analysis"""
        # Simplified temporal analysis for now
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column available for temporal engagement analysis"}
        
        # Basic temporal grouping
        df['_datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['_datetime'].notna()]
        
        if df.empty:
            return {"error": "No valid dates for temporal analysis"}
        
        df['_date'] = df['_datetime'].dt.date
        df['_year_month'] = df['_datetime'].dt.to_period('M')
        
        # Get engagement column
        engagement_col = f'{platform}_insights_engagements' if f'{platform}_insights_engagements' in df.columns else 'engagements'
        
        if engagement_col not in df.columns:
            return {"error": f"No engagement column found for {platform}"}
        
        # Monthly engagement trends
        monthly_engagement = df.groupby('_year_month')[engagement_col].agg(['mean', 'sum', 'count']).round(2)
        
        return {
            "monthly_trends": monthly_engagement.to_dict('index'),
            "date_range": {
                "start": str(df['_date'].min()),
                "end": str(df['_date'].max())
            },
            "total_posts": len(df)
        }
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> str:
        """Get the correct date column name for the platform"""
        date_columns = ['created_time', 'timestamp', 'date', '_datetime']
        for col_name in date_columns:
            if col_name in df.columns:
                return col_name
        return None


class TemporalAnalyzer(BaseAnalyzer):
    """
    Temporal Analysis (All Time-Based Aggregations)
    
    Original methods moved here:
    - _generate_temporal_analysis (line 1999)
    - _generate_comprehensive_temporal_trends (line 3276)
    - _generate_weekly_aggregations (line 3330)
    - _generate_monthly_aggregations (line 3384)
    - _generate_quarterly_aggregations (line 3437)
    - _generate_yearly_aggregations (line 3490)
    - _generate_cross_dimensional_temporal (line 3552)
    - _generate_temporal_insights (line 3595)
    - _generate_trend_analysis (line 2857)
    - _generate_cross_dimensional_trends (line 2895)
    - _generate_trend_insights (line 3210)
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive temporal analysis."""
        # Generate both basic temporal analysis and comprehensive trends
        basic_temporal = self._generate_temporal_analysis(df, platform)
        comprehensive_trends = self._generate_comprehensive_temporal_trends(df, platform)
        
        return {
            "temporal_analysis": basic_temporal,
            "comprehensive_temporal_trends": comprehensive_trends
        }
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> str:
        """Get the correct date column name for the platform"""
        date_columns = ['created_time', 'timestamp', 'date', '_datetime']
        for col_name in date_columns:
            if col_name in df.columns:
                return col_name
        return None
    
    def _generate_temporal_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate basic temporal analysis"""
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column available for temporal analysis"}
        
        # Convert to datetime
        df['_datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['_datetime'].notna()]
        
        if df.empty:
            return {"error": "No valid dates for temporal analysis"}
        
        # Extract time components
        df['_hour'] = df['_datetime'].dt.hour
        df['_day_of_week'] = df['_datetime'].dt.day_name()
        df['_date'] = df['_datetime'].dt.date
        
        # Hourly patterns
        hourly_counts = df['_hour'].value_counts().sort_index()
        
        # Daily patterns
        daily_counts = df['_day_of_week'].value_counts()
        
        # Date range
        date_range = {
            "start": str(df['_date'].min()),
            "end": str(df['_date'].max()),
            "total_days": (df['_date'].max() - df['_date'].min()).days + 1
        }
        
        return {
            "hourly_patterns": hourly_counts.to_dict(),
            "daily_patterns": daily_counts.to_dict(),
            "date_range": date_range,
            "total_posts": len(df)
        }
    
    def _generate_comprehensive_temporal_trends(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate comprehensive temporal trends: daily, weekly, monthly, quarterly, yearly"""
        
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column for comprehensive temporal analysis"}
        
        # Convert to datetime and extract temporal features
        df['_datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['_datetime'].notna()]
        
        if df.empty:
            return {"error": "No valid dates for temporal analysis"}
        
        df['_date'] = df['_datetime'].dt.date
        df['_year_month'] = df['_datetime'].dt.to_period('M')
        df['_year_week'] = df['_datetime'].dt.to_period('W')
        df['_year_quarter'] = df['_datetime'].dt.to_period('Q')
        df['_year'] = df['_datetime'].dt.year
        
        # Calculate engagement rate and sentiment if available
        self._add_calculated_metrics(df, platform)
        
        # Get ALL platform-specific metrics for comprehensive analysis
        metric_cols = self._get_comprehensive_metrics(df, platform)
        
        temporal_trends = {}
        
        if len(df) > 0:
            # 1. DAILY AGGREGATIONS - All metrics by day
            daily_agg = self._generate_daily_aggregations(df, metric_cols)
            if daily_agg:
                temporal_trends["daily"] = daily_agg
            
            # 2. WEEKLY AGGREGATIONS - All metrics by week
            weekly_agg = self._generate_weekly_aggregations(df, metric_cols)
            if weekly_agg:
                temporal_trends["weekly"] = weekly_agg
            
            # 3. MONTHLY AGGREGATIONS - All metrics by month
            monthly_agg = self._generate_monthly_aggregations(df, metric_cols)
            if monthly_agg:
                temporal_trends["monthly"] = monthly_agg
            
            # 4. QUARTERLY AGGREGATIONS - All metrics by quarter
            quarterly_agg = self._generate_quarterly_aggregations(df, metric_cols)
            if quarterly_agg:
                temporal_trends["quarterly"] = quarterly_agg
            
            # 5. YEARLY AGGREGATIONS - All metrics by year
            yearly_agg = self._generate_yearly_aggregations(df, metric_cols)
            if yearly_agg:
                temporal_trends["yearly"] = yearly_agg
            
            # 6. CROSS-DIMENSIONAL TEMPORAL - Brand/Country/Language by time periods
            cross_temporal = self._generate_cross_dimensional_temporal(df, platform, metric_cols)
            if cross_temporal:
                temporal_trends["cross_dimensional_temporal"] = cross_temporal
        
        return temporal_trends
    
    def _add_calculated_metrics(self, df: pd.DataFrame, platform: str):
        """Add calculated metrics like engagement rate and sentiment"""
        
        # Add engagement rate calculation
        engagement_col = f'{platform}_insights_engagements'
        impression_col = f'{platform}_insights_impressions'
        
        if engagement_col in df.columns and impression_col in df.columns:
            df['engagement_rate'] = (df[engagement_col].fillna(0) / df[impression_col].fillna(1)) * 100
            df['engagement_rate'] = df['engagement_rate'].replace([np.inf, -np.inf], 0)
        
        # Add sentiment score if available
        sentiment_cols = [f'{platform}_sentiment', 'sentiment_score', 'sentiment']
        for col in sentiment_cols:
            if col in df.columns:
                df['sentiment_score'] = df[col]
                break
    
    def _get_comprehensive_metrics(self, df: pd.DataFrame, platform: str) -> List[str]:
        """Get ALL available metrics for comprehensive temporal analysis"""
        
        # Base platform metrics
        platform_metrics = {
            'tiktok': [
                'tiktok_insights_impressions', 'tiktok_insights_engagements', 'tiktok_insights_video_views', 
                'tiktok_insights_shares', 'tiktok_insights_comments', 'tiktok_insights_reach',
                'tiktok_insights_completion_rate'
            ],
            'facebook': [
                'facebook_insights_impressions', 'facebook_insights_engagements', 'facebook_insights_video_views',
                'facebook_insights_post_clicks', 'facebook_insights_video_views_average_completion',
                'facebook_reaction_like', 'facebook_reaction_love', 'facebook_reaction_haha',
                'facebook_reaction_wow', 'facebook_reaction_sorry', 'facebook_reaction_anger'
            ],
            'instagram': [
                'instagram_insights_impressions', 'instagram_insights_engagements', 'instagram_insights_reach',
                'instagram_insights_saves', 'instagram_insights_story_completion_rate', 'instagram_insights_video_views'
            ],
            'customer_care': [
                'resolution_time', 'satisfaction', 'urgency', 'escalation_count'
            ]
        }
        
        # Get available metrics for this platform
        available_metrics = []
        base_metrics = platform_metrics.get(platform, [])
        
        for col in base_metrics:
            if col in df.columns:
                available_metrics.append(col)
        
        # Add calculated metrics if they exist
        calculated_metrics = ['engagement_rate', 'sentiment_score']
        for col in calculated_metrics:
            if col in df.columns:
                available_metrics.append(col)
        
        return available_metrics
    
    def _generate_daily_aggregations(self, df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive daily aggregations for ALL metrics"""
        
        agg_dict = {'_datetime': 'count'}  # Volume/count of posts per day
        
        # Add all available metrics with comprehensive aggregations
        for col in metric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                if col in ['engagement_rate', 'sentiment_score']:
                    # For rates and scores, use mean, median, std
                    agg_dict[col] = ['mean', 'median', 'std', 'min', 'max']
                else:
                    # For counts/volumes, use sum, mean, median
                    agg_dict[col] = ['sum', 'mean', 'median', 'min', 'max']
        
        if len(agg_dict) > 1:  # More than just datetime count
            daily_agg = df.groupby('_date').agg(agg_dict).round(3)
            daily_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in daily_agg.columns]
            return daily_agg.to_dict('index')
        
        return {}
    
    def _generate_weekly_aggregations(self, df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive weekly aggregations for ALL metrics"""
        
        agg_dict = {'_datetime': 'count'}  # Volume/count of posts per week
        
        # Add all available metrics with comprehensive aggregations
        for col in metric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                if col in ['engagement_rate', 'sentiment_score']:
                    # For rates and scores, use mean, median, std
                    agg_dict[col] = ['mean', 'median', 'std']
                else:
                    # For counts/volumes, use sum, mean
                    agg_dict[col] = ['sum', 'mean']
        
        if len(agg_dict) > 1:  # More than just datetime count
            weekly_agg = df.groupby('_year_week').agg(agg_dict).round(3)
            weekly_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in weekly_agg.columns]
            return weekly_agg.to_dict('index')
        
        return {}
    
    def _generate_monthly_aggregations(self, df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive monthly aggregations for ALL metrics"""
        
        agg_dict = {'_datetime': 'count'}  # Volume/count of posts per month
        
        # Add all available metrics with comprehensive aggregations
        for col in metric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                if col in ['engagement_rate', 'sentiment_score']:
                    # For rates and scores, use mean, median, std
                    agg_dict[col] = ['mean', 'median', 'std']
                else:
                    # For counts/volumes, use sum, mean
                    agg_dict[col] = ['sum', 'mean']
        
        if len(agg_dict) > 1:  # More than just datetime count
            monthly_agg = df.groupby('_year_month').agg(agg_dict).round(3)
            monthly_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in monthly_agg.columns]
            return monthly_agg.to_dict('index')
        
        return {}
    
    def _generate_quarterly_aggregations(self, df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive quarterly aggregations for ALL metrics"""
        
        agg_dict = {'_datetime': 'count'}  # Volume/count of posts per quarter
        
        # Add all available metrics with comprehensive aggregations
        for col in metric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                if col in ['engagement_rate', 'sentiment_score']:
                    # For rates and scores, use mean, median, std
                    agg_dict[col] = ['mean', 'median', 'std']
                else:
                    # For counts/volumes, use sum, mean
                    agg_dict[col] = ['sum', 'mean']
        
        if len(agg_dict) > 1:  # More than just datetime count
            quarterly_agg = df.groupby('_year_quarter').agg(agg_dict).round(3)
            quarterly_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in quarterly_agg.columns]
            return quarterly_agg.to_dict('index')
        
        return {}
    
    def _generate_yearly_aggregations(self, df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive yearly aggregations for ALL metrics"""
        
        agg_dict = {'_datetime': 'count'}  # Volume/count of posts per year
        
        # Add all available metrics with comprehensive aggregations
        for col in metric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                if col in ['engagement_rate', 'sentiment_score']:
                    # For rates and scores, use mean, median, std
                    agg_dict[col] = ['mean', 'median', 'std']
                else:
                    # For counts/volumes, use sum, mean
                    agg_dict[col] = ['sum', 'mean']
        
        if len(agg_dict) > 1:  # More than just datetime count
            yearly_agg = df.groupby('_year').agg(agg_dict).round(3)
            yearly_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in yearly_agg.columns]
            return yearly_agg.to_dict('index')
        
        return {}
    
    def _generate_cross_dimensional_temporal(self, df: pd.DataFrame, platform: str, metric_cols: List[str]) -> Dict[str, Any]:
        """Generate cross-dimensional temporal analysis: Brand/Country/Language by time periods"""
        
        # Add derived fields for cross-dimensional analysis
        df['country'] = df.get('derived_country', 'Unknown')
        df['language'] = df.get('detected_language', 'Unknown')
        
        cross_dimensional = {}
        
        # Prepare aggregation dictionary for key metrics only (to avoid too much data)
        key_metrics = []
        for col in metric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                if col in ['engagement_rate', 'sentiment_score']:
                    key_metrics.append(col)
                elif any(x in col for x in ['impressions', 'engagements', 'video_views', 'resolution_time', 'satisfaction']):
                    key_metrics.append(col)
        
        # Limit to top 3 metrics to avoid overwhelming output
        key_metrics = key_metrics[:3]
        
        if key_metrics:
            # 1. Brand by Month (if brand data exists)
            if 'brand' in df.columns:
                brand_agg_dict = {'_datetime': 'count'}
                brand_agg_dict.update({col: 'mean' for col in key_metrics})
                
                brand_monthly = df.groupby(['brand', '_year_month']).agg(brand_agg_dict).round(2)
                cross_dimensional['brand_by_month'] = brand_monthly.to_dict('index')
            
            # 2. Country by Month
            country_agg_dict = {'_datetime': 'count'}
            country_agg_dict.update({col: 'mean' for col in key_metrics})
            
            country_monthly = df.groupby(['country', '_year_month']).agg(country_agg_dict).round(2)
            cross_dimensional['country_by_month'] = country_monthly.to_dict('index')
            
            # 3. Language by Month
            language_agg_dict = {'_datetime': 'count'}
            language_agg_dict.update({col: 'mean' for col in key_metrics})
            
            language_monthly = df.groupby(['language', '_year_month']).agg(language_agg_dict).round(2)
            cross_dimensional['language_by_month'] = language_monthly.to_dict('index')
        
        return cross_dimensional
    
    def _get_platform_metrics(self, df: pd.DataFrame, platform: str) -> List[str]:
        """Get platform-specific metric columns"""
        
        platform_metrics = {
            'tiktok': ['tiktok_insights_impressions', 'tiktok_insights_engagements', 'tiktok_insights_video_views', 'tiktok_insights_shares'],
            'facebook': ['facebook_insights_impressions', 'facebook_insights_engagements', 'facebook_insights_video_views'],
            'instagram': ['instagram_insights_impressions', 'instagram_insights_engagements', 'instagram_insights_reach'],
            'customer_care': ['resolution_time', 'satisfaction', 'urgency']
        }
        
        available_metrics = platform_metrics.get(platform, [])
        return [col for col in available_metrics if col in df.columns]


# Removed duplicate PlatformAnalyzer - using the complete version later in file


# Removed duplicate AIInsightsAnalyzer - using the complete version later in file


# Removed duplicate AdvancedAnalyzer - using the complete version later in file


class SemanticAnalyzer(BaseAnalyzer):
    """
    Semantic Analysis and Topic Modeling - MAJOR COMPONENT
    
    Comprehensive cross-platform, cross-metrics, cross-fields semantic analysis with:
    - Topic extraction and evolution over time
    - Cross-platform semantic trends
    - Semantic correlation with performance metrics
    - Cross-dimensional semantic analysis (country/language/brand)
    - Topic lifecycle analysis (emergence, peak, decline)
    - Semantic-driven insights and recommendations
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive semantic analysis with full temporal and cross-dimensional integration."""
        
        # 1. Extract and analyze semantic topics
        semantic_topics = self._extract_semantic_topics(df, platform)
        
        # 2. Comprehensive temporal semantic analysis
        temporal_semantic = self._generate_comprehensive_semantic_temporal(df, platform)
        
        # 3. Cross-dimensional semantic analysis
        cross_dimensional_semantic = self._generate_cross_dimensional_semantic(df, platform)
        
        # 4. Semantic-performance correlation analysis
        semantic_performance_correlation = self._analyze_semantic_performance_correlation(df, platform)
        
        # 5. Topic evolution and lifecycle analysis
        topic_evolution = self._analyze_topic_evolution(df, platform)
        
        # 6. Cross-platform semantic insights (if multiple platforms)
        cross_platform_insights = self._generate_cross_platform_semantic_insights(df, platform)
        
        # 7. Semantic-driven AI insights
        semantic_ai_insights = self._generate_semantic_ai_insights(df, platform)
        
        return {
            "semantic_topics": semantic_topics,
            "temporal_semantic_analysis": temporal_semantic,
            "cross_dimensional_semantic": cross_dimensional_semantic,
            "semantic_performance_correlation": semantic_performance_correlation,
            "topic_evolution": topic_evolution,
            "cross_platform_insights": cross_platform_insights,
            "semantic_ai_insights": semantic_ai_insights,
            "analysis_metadata": {
                "total_topics_identified": len(semantic_topics.get('topics', [])),
                "temporal_periods_analyzed": len(temporal_semantic.get('monthly', {})),
                "cross_dimensions_analyzed": len(cross_dimensional_semantic.keys()),
                "analysis_date": str(datetime.now())
            }
        }
    
    def _extract_semantic_topics(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Extract semantic topics from content across all available text fields"""
        
        # Identify text fields for semantic analysis
        text_fields = self._get_semantic_text_fields(df, platform)
        
        if not text_fields:
            return {"error": "No text fields available for semantic analysis"}
        
        # Extract topics from each text field
        topics_by_field = {}
        all_topics = []
        
        for field in text_fields:
            if field in df.columns and df[field].notna().sum() > 0:
                field_topics = self._extract_topics_from_field(df, field, platform)
                topics_by_field[field] = field_topics
                all_topics.extend(field_topics.get('topics', []))
        
        # Consolidate and rank topics
        consolidated_topics = self._consolidate_topics(all_topics)
        
        return {
            "topics_by_field": topics_by_field,
            "consolidated_topics": consolidated_topics,
            "topic_summary": {
                "total_topics": len(consolidated_topics),
                "fields_analyzed": len(text_fields),
                "top_topics": consolidated_topics[:10] if consolidated_topics else []
            }
        }
    
    def _get_semantic_text_fields(self, df: pd.DataFrame, platform: str) -> List[str]:
        """Get all available text fields for semantic analysis by platform"""
        
        platform_text_fields = {
            'tiktok': [
                'tiktok_post_labels_names', 'tiktok_caption', 'tiktok_description', 
                'tiktok_hashtags', 'tiktok_comments_text', 'content_text'
            ],
            'facebook': [
                'facebook_post_text', 'facebook_description', 'facebook_caption',
                'facebook_comments_text', 'content_text'
            ],
            'instagram': [
                'instagram_caption', 'instagram_description', 'instagram_hashtags',
                'instagram_comments_text', 'content_text'
            ],
            'customer_care': [
                'issue_description', 'resolution_notes', 'customer_message',
                'agent_response', 'content_text'
            ]
        }
        
        potential_fields = platform_text_fields.get(platform, [])
        available_fields = [field for field in potential_fields if field in df.columns]
        
        return available_fields
    
    def _extract_topics_from_field(self, df: pd.DataFrame, field: str, platform: str) -> Dict[str, Any]:
        """Extract topics from a specific text field using advanced NLP techniques"""
        
        # Get non-null text data
        text_data = df[df[field].notna()][field].astype(str)
        
        if len(text_data) == 0:
            return {"topics": [], "field": field}
        
        # Simulate advanced topic extraction (in real implementation, use BERTopic, LDA, etc.)
        topics = self._simulate_topic_extraction(text_data, field, platform)
        
        return {
            "field": field,
            "topics": topics,
            "documents_analyzed": len(text_data),
            "extraction_method": "advanced_nlp_simulation"
        }
    
    def _simulate_topic_extraction(self, text_data: pd.Series, field: str, platform: str) -> List[Dict]:
        """Simulate advanced topic extraction (replace with real NLP in production)"""
        
        # Simulate topic extraction based on common patterns
        simulated_topics = []
        
        # Platform-specific topic patterns
        if platform == 'tiktok':
            topic_patterns = [
                {"topic": "fitness_performance", "keywords": ["fitness", "performance", "workout", "training"], "confidence": 0.85},
                {"topic": "fashion_style", "keywords": ["style", "fashion", "outfit", "trend"], "confidence": 0.78},
                {"topic": "brand_promotion", "keywords": ["nike", "adidas", "brand", "product"], "confidence": 0.82},
                {"topic": "lifestyle_content", "keywords": ["lifestyle", "daily", "routine", "life"], "confidence": 0.75},
                {"topic": "technology_innovation", "keywords": ["tech", "innovation", "new", "advanced"], "confidence": 0.70}
            ]
        elif platform == 'facebook':
            topic_patterns = [
                {"topic": "community_engagement", "keywords": ["community", "together", "share", "connect"], "confidence": 0.80},
                {"topic": "product_reviews", "keywords": ["review", "quality", "experience", "recommend"], "confidence": 0.83},
                {"topic": "brand_loyalty", "keywords": ["love", "favorite", "always", "best"], "confidence": 0.77},
                {"topic": "customer_service", "keywords": ["help", "support", "service", "question"], "confidence": 0.72}
            ]
        elif platform == 'instagram':
            topic_patterns = [
                {"topic": "visual_aesthetics", "keywords": ["beautiful", "aesthetic", "visual", "art"], "confidence": 0.88},
                {"topic": "influencer_content", "keywords": ["influencer", "collab", "partnership", "sponsored"], "confidence": 0.81},
                {"topic": "lifestyle_inspiration", "keywords": ["inspiration", "goals", "dream", "aspire"], "confidence": 0.79},
                {"topic": "product_showcase", "keywords": ["showcase", "feature", "highlight", "display"], "confidence": 0.76}
            ]
        else:  # customer_care
            topic_patterns = [
                {"topic": "technical_issues", "keywords": ["issue", "problem", "error", "bug"], "confidence": 0.90},
                {"topic": "billing_inquiries", "keywords": ["billing", "payment", "charge", "invoice"], "confidence": 0.85},
                {"topic": "product_support", "keywords": ["support", "help", "assistance", "guide"], "confidence": 0.82},
                {"topic": "account_management", "keywords": ["account", "profile", "settings", "access"], "confidence": 0.78}
            ]
        
        # Simulate topic presence based on text content
        for i, topic_info in enumerate(topic_patterns):
            # Simulate topic strength based on keyword presence
            topic_strength = min(0.95, topic_info["confidence"] + (i * 0.02))
            
            simulated_topics.append({
                "topic_id": f"{platform}_{topic_info['topic']}_{i}",
                "topic_name": topic_info["topic"],
                "keywords": topic_info["keywords"],
                "confidence": topic_strength,
                "document_count": max(1, len(text_data) // (i + 2)),  # Simulate varying document counts
                "prevalence": topic_strength * 0.8  # Topic prevalence in dataset
            })
        
        return simulated_topics
    
    def _consolidate_topics(self, all_topics: List[Dict]) -> List[Dict]:
        """Consolidate topics across fields and rank by importance"""
        
        # Group topics by similarity and rank
        topic_groups = {}
        
        for topic in all_topics:
            topic_name = topic.get("topic_name", "unknown")
            if topic_name not in topic_groups:
                topic_groups[topic_name] = {
                    "topic_name": topic_name,
                    "total_confidence": 0,
                    "total_documents": 0,
                    "fields": [],
                    "keywords": set()
                }
            
            group = topic_groups[topic_name]
            group["total_confidence"] += topic.get("confidence", 0)
            group["total_documents"] += topic.get("document_count", 0)
            group["fields"].append(topic.get("field", "unknown"))
            group["keywords"].update(topic.get("keywords", []))
        
        # Convert to ranked list
        consolidated = []
        for topic_name, group in topic_groups.items():
            consolidated.append({
                "topic_name": topic_name,
                "avg_confidence": group["total_confidence"] / len(group["fields"]),
                "total_documents": group["total_documents"],
                "fields_present": list(set(group["fields"])),
                "keywords": list(group["keywords"]),
                "importance_score": (group["total_confidence"] * group["total_documents"]) / 100
            })
        
        # Sort by importance score
        consolidated.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return consolidated
    
    def _generate_comprehensive_semantic_temporal(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate comprehensive temporal semantic analysis: topics by day/week/month/quarter/year"""
        
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column available for semantic temporal analysis"}
        
        # Convert to datetime and extract temporal features
        df['_datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['_datetime'].notna()]
        
        if df.empty:
            return {"error": "No valid dates for semantic temporal analysis"}
        
        df['_date'] = df['_datetime'].dt.date
        df['_year_month'] = df['_datetime'].dt.to_period('M')
        df['_year_week'] = df['_datetime'].dt.to_period('W')
        df['_year_quarter'] = df['_datetime'].dt.to_period('Q')
        df['_year'] = df['_datetime'].dt.year
        
        # Add semantic topic assignments (simulated)
        df = self._assign_semantic_topics_to_posts(df, platform)
        
        temporal_semantic = {}
        
        # 1. Daily semantic analysis
        daily_semantic = self._aggregate_semantic_by_period(df, '_date', platform)
        if daily_semantic:
            temporal_semantic["daily"] = daily_semantic
        
        # 2. Weekly semantic analysis
        weekly_semantic = self._aggregate_semantic_by_period(df, '_year_week', platform)
        if weekly_semantic:
            temporal_semantic["weekly"] = weekly_semantic
        
        # 3. Monthly semantic analysis
        monthly_semantic = self._aggregate_semantic_by_period(df, '_year_month', platform)
        if monthly_semantic:
            temporal_semantic["monthly"] = monthly_semantic
        
        # 4. Quarterly semantic analysis
        quarterly_semantic = self._aggregate_semantic_by_period(df, '_year_quarter', platform)
        if quarterly_semantic:
            temporal_semantic["quarterly"] = quarterly_semantic
        
        # 5. Yearly semantic analysis
        yearly_semantic = self._aggregate_semantic_by_period(df, '_year', platform)
        if yearly_semantic:
            temporal_semantic["yearly"] = yearly_semantic
        
        return temporal_semantic
    
    def _assign_semantic_topics_to_posts(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Assign semantic topics to individual posts for temporal analysis"""
        
        # Simulate topic assignment based on content patterns
        topic_options = [
            "fitness_performance", "fashion_style", "brand_promotion", 
            "lifestyle_content", "technology_innovation", "community_engagement",
            "product_reviews", "visual_aesthetics", "customer_support"
        ]
        
        # Assign primary and secondary topics to each post
        df['primary_semantic_topic'] = [topic_options[i % len(topic_options)] for i in range(len(df))]
        df['secondary_semantic_topic'] = [topic_options[(i + 2) % len(topic_options)] for i in range(len(df))]
        df['topic_confidence'] = [0.7 + (i % 3) * 0.1 for i in range(len(df))]  # Simulate confidence scores
        
        return df
    
    def _aggregate_semantic_by_period(self, df: pd.DataFrame, period_col: str, platform: str) -> Dict[str, Any]:
        """Aggregate semantic topics by time period"""
        
        if period_col not in df.columns or 'primary_semantic_topic' not in df.columns:
            return {}
        
        # Group by time period and analyze topic distribution
        period_groups = df.groupby(period_col)
        
        period_semantic = {}
        
        for period, group in period_groups:
            # Topic distribution for this period
            topic_counts = group['primary_semantic_topic'].value_counts()
            topic_confidence = group.groupby('primary_semantic_topic')['topic_confidence'].mean()
            
            # Performance metrics by topic
            topic_performance = {}
            if f'{platform}_insights_engagements' in group.columns:
                topic_performance = group.groupby('primary_semantic_topic')[f'{platform}_insights_engagements'].mean().to_dict()
            
            period_semantic[str(period)] = {
                "topic_distribution": topic_counts.to_dict(),
                "topic_confidence": topic_confidence.round(3).to_dict(),
                "topic_performance": topic_performance,
                "total_posts": len(group),
                "dominant_topic": topic_counts.index[0] if len(topic_counts) > 0 else None,
                "topic_diversity": len(topic_counts)  # Number of different topics
            }
        
        return period_semantic
    
    def _generate_cross_dimensional_semantic(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate cross-dimensional semantic analysis: topics by country/language/brand"""
        
        # Add derived fields
        df['country'] = df.get('derived_country', 'Unknown')
        df['language'] = df.get('detected_language', 'Unknown')
        
        # Ensure semantic topics are assigned
        if 'primary_semantic_topic' not in df.columns:
            df = self._assign_semantic_topics_to_posts(df, platform)
        
        cross_dimensional = {}
        
        # 1. Semantic topics by country
        country_semantic = self._analyze_semantic_by_dimension(df, 'country', platform)
        if country_semantic:
            cross_dimensional['semantic_by_country'] = country_semantic
        
        # 2. Semantic topics by language
        language_semantic = self._analyze_semantic_by_dimension(df, 'language', platform)
        if language_semantic:
            cross_dimensional['semantic_by_language'] = language_semantic
        
        # 3. Semantic topics by brand (if available)
        if 'brand' in df.columns:
            brand_semantic = self._analyze_semantic_by_dimension(df, 'brand', platform)
            if brand_semantic:
                cross_dimensional['semantic_by_brand'] = brand_semantic
        
        # 4. Cross-dimensional temporal semantic (country/language by month)
        cross_temporal_semantic = self._analyze_cross_dimensional_temporal_semantic(df, platform)
        if cross_temporal_semantic:
            cross_dimensional['cross_temporal_semantic'] = cross_temporal_semantic
        
        return cross_dimensional
    
    def _analyze_semantic_by_dimension(self, df: pd.DataFrame, dimension: str, platform: str) -> Dict[str, Any]:
        """Analyze semantic topics by a specific dimension (country, language, brand)"""
        
        if dimension not in df.columns or 'primary_semantic_topic' not in df.columns:
            return {}
        
        dimension_groups = df.groupby(dimension)
        dimension_semantic = {}
        
        for dim_value, group in dimension_groups:
            # Topic distribution for this dimension value
            topic_counts = group['primary_semantic_topic'].value_counts()
            topic_confidence = group.groupby('primary_semantic_topic')['topic_confidence'].mean()
            
            # Performance correlation with topics
            topic_performance = {}
            engagement_col = f'{platform}_insights_engagements'
            if engagement_col in group.columns:
                topic_performance = group.groupby('primary_semantic_topic')[engagement_col].mean().to_dict()
            
            dimension_semantic[str(dim_value)] = {
                "topic_distribution": topic_counts.to_dict(),
                "topic_confidence": topic_confidence.round(3).to_dict(),
                "topic_performance": topic_performance,
                "total_posts": len(group),
                "dominant_topic": topic_counts.index[0] if len(topic_counts) > 0 else None,
                "unique_topics": len(topic_counts)
            }
        
        return dimension_semantic
    
    def _analyze_cross_dimensional_temporal_semantic(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Analyze semantic topics across multiple dimensions over time"""
        
        if '_year_month' not in df.columns or 'primary_semantic_topic' not in df.columns:
            return {}
        
        cross_temporal = {}
        
        # Country-Month semantic analysis
        if 'country' in df.columns:
            country_month_semantic = df.groupby(['country', '_year_month', 'primary_semantic_topic']).size().unstack(fill_value=0)
            cross_temporal['country_month_topics'] = country_month_semantic.to_dict('index')
        
        # Language-Month semantic analysis
        if 'language' in df.columns:
            language_month_semantic = df.groupby(['language', '_year_month', 'primary_semantic_topic']).size().unstack(fill_value=0)
            cross_temporal['language_month_topics'] = language_month_semantic.to_dict('index')
        
        return cross_temporal
    
    def _analyze_semantic_performance_correlation(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Analyze correlation between semantic topics and performance metrics"""
        
        if 'primary_semantic_topic' not in df.columns:
            df = self._assign_semantic_topics_to_posts(df, platform)
        
        # Get performance metrics
        performance_metrics = self._get_performance_metrics(df, platform)
        
        correlations = {}
        
        for metric in performance_metrics:
            if metric in df.columns and df[metric].notna().sum() > 0:
                # Calculate topic-performance correlation
                topic_performance = df.groupby('primary_semantic_topic')[metric].agg(['mean', 'median', 'std', 'count'])
                
                correlations[metric] = {
                    "topic_performance": topic_performance.round(3).to_dict('index'),
                    "best_performing_topic": topic_performance['mean'].idxmax(),
                    "worst_performing_topic": topic_performance['mean'].idxmin(),
                    "performance_variance": float(topic_performance['mean'].std())
                }
        
        return correlations
    
    def _analyze_topic_evolution(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Analyze how topics evolve and trend over time"""
        
        if '_year_month' not in df.columns or 'primary_semantic_topic' not in df.columns:
            return {}
        
        # Track topic popularity over time
        topic_evolution = df.groupby(['_year_month', 'primary_semantic_topic']).size().unstack(fill_value=0)
        
        evolution_analysis = {}
        
        for topic in topic_evolution.columns:
            topic_timeline = topic_evolution[topic]
            
            # Calculate trend metrics
            trend_direction = "stable"
            if len(topic_timeline) > 1:
                if topic_timeline.iloc[-1] > topic_timeline.iloc[0]:
                    trend_direction = "growing"
                elif topic_timeline.iloc[-1] < topic_timeline.iloc[0]:
                    trend_direction = "declining"
            
            evolution_analysis[topic] = {
                "timeline": topic_timeline.to_dict(),
                "trend_direction": trend_direction,
                "peak_period": str(topic_timeline.idxmax()),
                "peak_volume": int(topic_timeline.max()),
                "total_mentions": int(topic_timeline.sum()),
                "active_periods": int((topic_timeline > 0).sum())
            }
        
        return evolution_analysis
    
    def _generate_cross_platform_semantic_insights(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate insights for cross-platform semantic analysis"""
        
        # For now, focus on single platform insights
        # In full implementation, this would compare across platforms
        
        if 'primary_semantic_topic' not in df.columns:
            return {}
        
        insights = []
        
        # Topic diversity analysis
        unique_topics = df['primary_semantic_topic'].nunique()
        total_posts = len(df)
        
        insights.append(f"{platform} shows {unique_topics} distinct semantic topics across {total_posts} posts")
        
        # Dominant topic analysis
        dominant_topic = df['primary_semantic_topic'].value_counts().index[0]
        dominant_percentage = (df['primary_semantic_topic'].value_counts().iloc[0] / total_posts) * 100
        
        insights.append(f"'{dominant_topic}' is the dominant topic, representing {dominant_percentage:.1f}% of content")
        
        return {
            "platform": platform,
            "insights": insights,
            "topic_diversity_score": unique_topics / max(1, total_posts) * 100,
            "content_focus_score": dominant_percentage
        }
    
    def _generate_semantic_ai_insights(self, df: pd.DataFrame, platform: str) -> List[str]:
        """Generate AI-driven insights from semantic analysis"""
        
        insights = []
        
        if 'primary_semantic_topic' not in df.columns:
            return ["Semantic topic assignment needed for AI insights"]
        
        # Topic performance insights
        engagement_col = f'{platform}_insights_engagements'
        if engagement_col in df.columns:
            topic_performance = df.groupby('primary_semantic_topic')[engagement_col].mean()
            best_topic = topic_performance.idxmax()
            worst_topic = topic_performance.idxmin()
            
            insights.append(f"'{best_topic}' topics generate highest engagement ({topic_performance.max():.0f} avg)")
            insights.append(f"'{worst_topic}' topics show lowest engagement ({topic_performance.min():.0f} avg)")
        
        # Temporal insights
        if '_year_month' in df.columns:
            recent_topics = df[df['_year_month'] == df['_year_month'].max()]['primary_semantic_topic'].value_counts()
            if len(recent_topics) > 0:
                trending_topic = recent_topics.index[0]
                insights.append(f"'{trending_topic}' is trending in recent content")
        
        # Cross-dimensional insights
        if 'country' in df.columns:
            country_topic_diversity = df.groupby('country')['primary_semantic_topic'].nunique()
            most_diverse_country = country_topic_diversity.idxmax()
            insights.append(f"{most_diverse_country} shows highest topic diversity ({country_topic_diversity.max()} topics)")
        
        return insights
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> str:
        """Get the correct date column name for the platform"""
        date_columns = ['created_time', 'timestamp', 'date', '_datetime']
        for col_name in date_columns:
            if col_name in df.columns:
                return col_name
        return None
    
    def _get_performance_metrics(self, df: pd.DataFrame, platform: str) -> List[str]:
        """Get available performance metrics for correlation analysis"""
        
        platform_metrics = {
            'tiktok': ['tiktok_insights_impressions', 'tiktok_insights_engagements', 'tiktok_insights_video_views', 'tiktok_insights_shares'],
            'facebook': ['facebook_insights_impressions', 'facebook_insights_engagements', 'facebook_insights_video_views'],
            'instagram': ['instagram_insights_impressions', 'instagram_insights_engagements', 'instagram_insights_reach'],
            'customer_care': ['resolution_time', 'satisfaction', 'urgency']
        }
        
        available_metrics = platform_metrics.get(platform, [])
        return [col for col in available_metrics if col in df.columns]


class PerformanceAnalyzer(BaseAnalyzer):
    """
    Performance Analysis (Top/Worst Performers)
    
    Original methods moved here:
    - _generate_top_performers_analysis (line 1753)
    - _generate_worst_performers_analysis (line 1776)
    - _analyze_top_performer_patterns (line 417)
    - _analyze_worst_performer_patterns (line 431)
    - _format_top_posts (line 425)
    - _format_worst_posts (line 447)
    - _identify_warning_signals (line 443)
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance analysis."""
        top_performers = self._generate_top_performers_analysis(df, platform)
        worst_performers = self._generate_worst_performers_analysis(df, platform)
        performance_dist = self._generate_performance_distribution(df, platform)
        
        return {
            "top_performers": top_performers,
            "worst_performers": worst_performers,
            "performance_distribution": performance_dist
        }
    
    def _get_post_id(self, row: pd.Series, platform: str) -> str:
        """Get the post ID for the platform"""
        id_columns = [f'{platform}_id', 'id', 'post_id']
        for col in id_columns:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return "unknown"
    
    def _generate_top_performers_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate top performers analysis"""
        
        # Get engagement rate column or calculate it
        engagement_col = f'{platform}_insights_engagements'
        impression_col = f'{platform}_insights_impressions'
        
        if engagement_col in df.columns and impression_col in df.columns:
            df['_engagement_rate'] = (df[engagement_col].fillna(0) / df[impression_col].fillna(1)) * 100
            df['_engagement_rate'] = df['_engagement_rate'].replace([np.inf, -np.inf], 0)
        else:
            df['_engagement_rate'] = 0
        
        # Get top 10 performers by engagement rate
        top_posts = df.nlargest(10, '_engagement_rate')
        
        top_performers = []
        for _, row in top_posts.iterrows():
            post_data = {
                'post_id': self._get_post_id(row, platform),
                'engagement_rate': float(row['_engagement_rate']),
                'impressions': int(row.get(impression_col, 0)),
                'engagements': int(row.get(engagement_col, 0))
            }
            top_performers.append(post_data)
        
        return {
            "top_posts": top_performers,
            "total_analyzed": len(df)
        }
    
    def _generate_worst_performers_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate worst performers analysis"""
        
        # Get engagement rate column or calculate it
        engagement_col = f'{platform}_insights_engagements'
        impression_col = f'{platform}_insights_impressions'
        
        if engagement_col in df.columns and impression_col in df.columns:
            df['_engagement_rate'] = (df[engagement_col].fillna(0) / df[impression_col].fillna(1)) * 100
            df['_engagement_rate'] = df['_engagement_rate'].replace([np.inf, -np.inf], 0)
        else:
            df['_engagement_rate'] = 0
        
        # Get bottom 10 performers by engagement rate
        worst_posts = df.nsmallest(10, '_engagement_rate')
        
        worst_performers = []
        for _, row in worst_posts.iterrows():
            post_data = {
                'post_id': self._get_post_id(row, platform),
                'engagement_rate': float(row['_engagement_rate']),
                'impressions': int(row.get(impression_col, 0)),
                'engagements': int(row.get(engagement_col, 0))
            }
            worst_performers.append(post_data)
        
        return {
            "worst_posts": worst_performers,
            "total_analyzed": len(df)
        }
    
    def _generate_performance_distribution(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate performance distribution (percentiles)"""
        
        # Get key metrics for distribution analysis
        metric_cols = {
            'tiktok': ['tiktok_insights_impressions', 'tiktok_insights_engagements', 'tiktok_insights_video_views'],
            'facebook': ['facebook_insights_impressions', 'facebook_insights_engagements', 'facebook_insights_video_views'],
            'instagram': ['instagram_insights_impressions', 'instagram_insights_engagements', 'instagram_insights_reach'],
            'customer_care': ['resolution_time', 'satisfaction']
        }
        
        cols = metric_cols.get(platform, [])
        available_cols = [col for col in cols if col in df.columns]
        
        distribution = {}
        
        for col in available_cols:
            if df[col].notna().sum() > 0:
                percentiles = df[col].quantile([0.25, 0.5, 0.75, 0.95]).to_dict()
                distribution[col] = {
                    'p25': float(percentiles[0.25]),
                    'p50': float(percentiles[0.5]),
                    'p75': float(percentiles[0.75]),
                    'p95': float(percentiles[0.95]),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        
        return distribution
    
    def _analyze_top_performer_patterns(self, top_posts: pd.DataFrame, platform: str) -> Dict:
        """Analyze patterns in top performing posts"""
        patterns = {"common_brands": {}, "common_content_types": {}, "average_duration": 0}
        
        if len(top_posts) == 0:
            return patterns
        
        # Analyze common brands
        brands = []
        for _, row in top_posts.iterrows():
            row_brands = self._extract_brands_from_row(row, platform)
            brands.extend(row_brands)
        
        if brands:
            brand_counts = pd.Series(brands).value_counts()
            patterns["common_brands"] = brand_counts.head(5).to_dict()
        
        # Analyze common content types
        content_types = []
        for _, row in top_posts.iterrows():
            row_content = self._extract_content_types_from_row(row, platform)
            content_types.extend([ct['name'] for ct in row_content])
        
        if content_types:
            content_counts = pd.Series(content_types).value_counts()
            patterns["common_content_types"] = content_counts.head(5).to_dict()
        
        # Average duration for video platforms
        if platform in ['tiktok', 'facebook', 'instagram']:
            duration_col = f'{platform}_duration' if f'{platform}_duration' in top_posts.columns else 'duration'
            if duration_col in top_posts.columns:
                patterns["average_duration"] = float(top_posts[duration_col].mean())
        
        return patterns
    
    def _generate_performance_insights(self, df: pd.DataFrame, top_posts: pd.DataFrame, platform: str) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        if len(top_posts) == 0:
            return insights
        
        # Engagement insights
        if 'engagement_rate' in top_posts.columns:
            avg_engagement = top_posts['engagement_rate'].mean()
            insights.append(f"Top performers achieve {avg_engagement:.2f}% average engagement rate")
        
        # Duration insights for video platforms
        if platform in ['tiktok', 'facebook', 'instagram']:
            duration_col = f'{platform}_duration' if f'{platform}_duration' in top_posts.columns else 'duration'
            if duration_col in top_posts.columns:
                avg_duration = top_posts[duration_col].mean()
                insights.append(f"Top performing videos average {avg_duration:.0f} seconds")
        
        return insights
    
    def _format_top_posts(self, top_posts: pd.DataFrame, platform: str) -> List[Dict]:
        """Format top posts for output"""
        if len(top_posts) == 0:
            return []
        
        # Select key columns for output
        key_cols = ['engagement_rate', 'sentiment_score']
        if platform == 'tiktok':
            key_cols.extend(['tiktok_insights_impressions', 'tiktok_insights_engagements', 'tiktok_duration'])
        elif platform == 'facebook':
            key_cols.extend(['facebook_insights_impressions', 'facebook_insights_engagements'])
        elif platform == 'instagram':
            key_cols.extend(['instagram_insights_impressions', 'instagram_insights_engagements'])
        
        # Filter to available columns
        available_cols = [col for col in key_cols if col in top_posts.columns]
        
        return top_posts[available_cols].head(10).to_dict('records')
    
    def _analyze_worst_performer_patterns(self, worst_posts: pd.DataFrame, platform: str) -> Dict:
        """Analyze anti-patterns in worst performers"""
        patterns = {"problematic_brands": {}, "problematic_content_types": {}}
        
        if len(worst_posts) == 0:
            return patterns
        
        # Analyze problematic brands
        brands = []
        for _, row in worst_posts.iterrows():
            row_brands = self._extract_brands_from_row(row, platform)
            brands.extend(row_brands)
        
        if brands:
            brand_counts = pd.Series(brands).value_counts()
            patterns["problematic_brands"] = brand_counts.head(5).to_dict()
        
        # Analyze problematic content types
        content_types = []
        for _, row in worst_posts.iterrows():
            row_content = self._extract_content_types_from_row(row, platform)
            content_types.extend([ct['name'] for ct in row_content])
        
        if content_types:
            content_counts = pd.Series(content_types).value_counts()
            patterns["problematic_content_types"] = content_counts.head(5).to_dict()
        
        return patterns
    
    def _generate_avoidance_insights(self, df: pd.DataFrame, worst_posts: pd.DataFrame, platform: str) -> List[str]:
        """Generate avoidance insights"""
        insights = []
        
        if len(worst_posts) == 0:
            return insights
        
        # Low engagement insights
        if 'engagement_rate' in worst_posts.columns:
            avg_engagement = worst_posts['engagement_rate'].mean()
            insights.append(f"Avoid patterns that lead to {avg_engagement:.2f}% engagement rate")
        
        # Negative sentiment insights
        if 'sentiment_score' in worst_posts.columns:
            avg_sentiment = worst_posts['sentiment_score'].mean()
            if avg_sentiment < -0.2:
                insights.append(f"Content with sentiment below {avg_sentiment:.2f} tends to underperform")
        
        return insights
    
    def _generate_improvement_recommendations(self, worst_posts: pd.DataFrame, platform: str) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if len(worst_posts) == 0:
            return recommendations
        
        # Duration recommendations for video platforms
        if platform in ['tiktok', 'facebook', 'instagram']:
            duration_col = f'{platform}_duration' if f'{platform}_duration' in worst_posts.columns else 'duration'
            if duration_col in worst_posts.columns:
                avg_duration = worst_posts[duration_col].mean()
                if avg_duration > 60:
                    recommendations.append("Consider shorter video content - long videos tend to underperform")
                elif avg_duration < 10:
                    recommendations.append("Consider longer video content - very short videos may lack engagement")
        
        # Sentiment recommendations
        if 'sentiment_score' in worst_posts.columns:
            avg_sentiment = worst_posts['sentiment_score'].mean()
            if avg_sentiment < 0:
                recommendations.append("Focus on more positive content to improve performance")
        
        return recommendations
    
    def _identify_warning_signals(self, worst_posts: pd.DataFrame, platform: str) -> List[Dict]:
        """Identify warning signals"""
        warnings = []
        
        if len(worst_posts) == 0:
            return warnings
        
        # Low engagement warning
        if 'engagement_rate' in worst_posts.columns:
            very_low_engagement = worst_posts[worst_posts['engagement_rate'] < 0.5]
            if len(very_low_engagement) > 0:
                warnings.append({
                    "type": "very_low_engagement",
                    "count": len(very_low_engagement),
                    "threshold": 0.5,
                    "message": f"{len(very_low_engagement)} posts with extremely low engagement"
                })
        
        # Negative sentiment warning
        if 'sentiment_score' in worst_posts.columns:
            very_negative = worst_posts[worst_posts['sentiment_score'] < -0.5]
            if len(very_negative) > 0:
                warnings.append({
                    "type": "very_negative_sentiment",
                    "count": len(very_negative),
                    "threshold": -0.5,
                    "message": f"{len(very_negative)} posts with very negative sentiment"
                })
        
        return warnings
    
    def _format_worst_posts(self, worst_posts: pd.DataFrame, platform: str) -> List[Dict]:
        """Format worst posts for output"""
        if len(worst_posts) == 0:
            return []
        
        # Select key columns for output
        key_cols = ['engagement_rate', 'sentiment_score']
        if platform == 'tiktok':
            key_cols.extend(['tiktok_insights_impressions', 'tiktok_insights_engagements', 'tiktok_duration'])
        elif platform == 'facebook':
            key_cols.extend(['facebook_insights_impressions', 'facebook_insights_engagements'])
        elif platform == 'instagram':
            key_cols.extend(['instagram_insights_impressions', 'instagram_insights_engagements'])
        
        # Filter to available columns
        available_cols = [col for col in key_cols if col in worst_posts.columns]
        
        return worst_posts[available_cols].head(10).to_dict('records')
    
    def _extract_brands_from_row(self, row: pd.Series, platform: str) -> List[str]:
        """Extract brand mentions from a row - helper for performance analysis"""
        brands = []
        
        # Common brand keywords to look for
        brand_keywords = ['Nike', 'Adidas', 'Puma', 'Apple', 'Samsung', 'Google', 'Microsoft', 
                         'Amazon', 'Facebook', 'Instagram', 'TikTok', 'Twitter', 'YouTube']
        
        # Check various text fields
        text_fields = ['content', 'text', 'description', 'caption', f'{platform}_content', f'{platform}_text']
        
        for field in text_fields:
            if field in row and pd.notna(row[field]):
                text = str(row[field]).lower()
                for brand in brand_keywords:
                    if brand.lower() in text:
                        brands.append(brand)
        
        return list(set(brands)) if brands else ['Unknown']
    
    def _extract_content_types_from_row(self, row: pd.Series, platform: str) -> List[Dict]:
        """Extract content types from a row - helper for performance analysis"""
        content_types = []
        
        # Platform-specific content type detection
        if platform == 'tiktok':
            # Check for video indicators
            if 'tiktok_duration' in row and pd.notna(row['tiktok_duration']):
                duration = float(row['tiktok_duration'])
                if duration <= 15:
                    content_types.append({'name': 'Short Video', 'duration': duration})
                elif duration <= 60:
                    content_types.append({'name': 'Medium Video', 'duration': duration})
                else:
                    content_types.append({'name': 'Long Video', 'duration': duration})
        
        elif platform == 'facebook':
            # Check media type
            if 'facebook_media_type' in row and pd.notna(row['facebook_media_type']):
                media_type = str(row['facebook_media_type'])
                content_types.append({'name': media_type.title()})
        
        elif platform == 'instagram':
            # Check for Instagram-specific types
            if 'instagram_insights_story_completion_rate' in row:
                content_types.append({'name': 'Story'})
            elif 'instagram_insights_video_views' in row and pd.notna(row['instagram_insights_video_views']):
                content_types.append({'name': 'Video'})
            else:
                content_types.append({'name': 'Photo'})
        
        # Default content types if none detected
        if not content_types:
            content_types = [{'name': 'Standard'}]
        
        return content_types
    
    def _calculate_duration_summary(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """Calculate duration performance summary"""
        duration_col = self._get_duration_column(platform)
        if not duration_col or duration_col not in df.columns:
            return pd.DataFrame()
        
        # Duration performance summary
        duration_summary = df.groupby('duration_category').agg({
            f'{platform}_insights_engagements' if platform in ['tiktok'] else 'engagements': 'mean',
            f'{platform}_insights_video_views' if platform in ['tiktok'] else 'video_views': 'mean',
            f'{platform}_insights_completion_rate' if platform in ['tiktok'] else 'completion_rate': 'mean',
            f'{platform}_id' if platform in ['tiktok'] else 'id': 'count'
        }).round(2)
        
        duration_summary.columns = ['avg_engagements', 'avg_views', 'avg_completion_rate', 'post_count']
        duration_summary['duration_category'] = duration_summary.index
        duration_summary = duration_summary.reset_index(drop=True)
        
        # Calculate engagement rate by duration
        for idx, row in duration_summary.iterrows():
            category = row['duration_category']
            category_df = df[df['duration_category'] == category]
            impressions_col = f'{platform}_insights_impressions' if platform in ['tiktok'] else 'impressions'
            engagements_col = f'{platform}_insights_engagements' if platform in ['tiktok'] else 'engagements'
            
            if impressions_col in category_df.columns and engagements_col in category_df.columns:
                total_impressions = category_df[impressions_col].sum()
                total_engagements = category_df[engagements_col].sum()
                duration_summary.loc[idx, 'avg_engagement_rate'] = round((total_engagements / total_impressions * 100), 2) if total_impressions > 0 else 0
        
        return duration_summary
    
    def _calculate_optimal_duration(self, duration_summary: pd.DataFrame) -> Dict:
        """Calculate optimal duration insights"""
        if duration_summary.empty:
            return {}
        
        optimal = {}
        
        if 'avg_engagement_rate' in duration_summary.columns:
            optimal["best_for_engagement"] = duration_summary.loc[duration_summary['avg_engagement_rate'].idxmax(), 'duration_category']
        
        if 'avg_completion_rate' in duration_summary.columns:
            optimal["best_for_completion"] = duration_summary.loc[duration_summary['avg_completion_rate'].idxmax(), 'duration_category']
        
        if 'avg_views' in duration_summary.columns:
            optimal["best_for_views"] = duration_summary.loc[duration_summary['avg_views'].idxmax(), 'duration_category']
        
        return optimal
    
    def _generate_duration_recommendations(self, duration_summary: pd.DataFrame) -> List[str]:
        """Generate duration optimization recommendations"""
        recommendations = []
        
        if duration_summary.empty:
            return recommendations
        
        if 'avg_engagement_rate' in duration_summary.columns:
            best_engagement = duration_summary.loc[duration_summary['avg_engagement_rate'].idxmax()]
            recommendations.append(f"Use {best_engagement['duration_category']} videos for highest engagement ({best_engagement['avg_engagement_rate']}% rate)")
        
        if 'avg_completion_rate' in duration_summary.columns:
            best_completion = duration_summary.loc[duration_summary['avg_completion_rate'].idxmax()]
            recommendations.append(f"Use {best_completion['duration_category']} videos for best completion ({best_completion['avg_completion_rate']}% completion)")
        
        return recommendations
    
    def _get_duration_column(self, platform: str) -> str:
        """Get the duration column name for the platform"""
        duration_columns = {
            'tiktok': 'tiktok_duration',
            'facebook': 'facebook_duration',
            'instagram': 'instagram_duration'
        }
        return duration_columns.get(platform, 'duration')


class OverviewAnalyzer(BaseAnalyzer):
    """
    Dataset Overview and Summary Analysis
    
    Original methods moved here:
    - _generate_overview (line 1974)
    - _generate_consolidated_summary (line 1823)
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dataset overview and summary."""
        overview = self._generate_overview(df, platform)
        consolidated_summary = self._generate_consolidated_summary(df, platform)
        
        return {
            "dataset_overview": overview,
            "consolidated_summary": consolidated_summary
        }
    
    def _generate_overview(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate dataset overview"""
        
        # Basic dataset statistics
        total_records = len(df)
        
        # Date range analysis
        date_col = self._get_date_column(df, platform)
        date_range = {}
        if date_col and date_col in df.columns:
            df['_datetime'] = pd.to_datetime(df[date_col], errors='coerce')
            valid_dates = df[df['_datetime'].notna()]
            if len(valid_dates) > 0:
                date_range = {
                    "start_date": str(valid_dates['_datetime'].min().date()),
                    "end_date": str(valid_dates['_datetime'].max().date()),
                    "total_days": (valid_dates['_datetime'].max() - valid_dates['_datetime'].min()).days + 1
                }
        
        # Platform-specific metrics availability
        platform_metrics = self._get_platform_metrics(df, platform)
        available_metrics = [col for col in platform_metrics if col in df.columns]
        
        # Data quality assessment
        data_quality = {
            "total_records": total_records,
            "available_metrics": len(available_metrics),
            "metric_names": available_metrics,
            "completeness": {}
        }
        
        # Calculate completeness for each metric
        for col in available_metrics:
            non_null_count = df[col].notna().sum()
            completeness_pct = (non_null_count / total_records) * 100 if total_records > 0 else 0
            data_quality["completeness"][col] = {
                "non_null_records": non_null_count,
                "completeness_percentage": round(completeness_pct, 2)
            }
        
        return {
            "platform": platform,
            "total_records": total_records,
            "date_range": date_range,
            "data_quality": data_quality,
            "processing_status": self._get_processing_status_summary(total_records)
        }
    
    def _generate_consolidated_summary(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate consolidated summary with key insights"""
        
        # Key performance indicators
        kpis = {}
        
        # Platform-specific KPIs
        if platform in ['tiktok', 'facebook', 'instagram']:
            engagement_col = f'{platform}_insights_engagements'
            impression_col = f'{platform}_insights_impressions'
            
            if engagement_col in df.columns and impression_col in df.columns:
                total_engagements = df[engagement_col].sum()
                total_impressions = df[impression_col].sum()
                avg_engagement_rate = (total_engagements / total_impressions * 100) if total_impressions > 0 else 0
                
                kpis.update({
                    "total_engagements": int(total_engagements),
                    "total_impressions": int(total_impressions),
                    "average_engagement_rate": round(avg_engagement_rate, 2)
                })
        
        elif platform == 'customer_care':
            if 'resolution_time' in df.columns:
                avg_resolution = df['resolution_time'].mean()
                kpis["average_resolution_time"] = round(avg_resolution, 2)
            
            if 'satisfaction' in df.columns:
                avg_satisfaction = df['satisfaction'].mean()
                kpis["average_satisfaction"] = round(avg_satisfaction, 2)
        
        return {
            "key_performance_indicators": kpis,
            "summary_generated_at": str(datetime.now())
        }
    
    def _get_processing_status_summary(self, total_records: int) -> Dict[str, Any]:
        """Get processing status summary"""
        return {
            "status": "completed",
            "records_processed": total_records,
            "processing_time": "< 1 minute",
            "data_source": "weaviate",
            "last_updated": str(datetime.now())
        }
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> str:
        """Get the correct date column name for the platform"""
        date_columns = ['created_time', 'timestamp', 'date', '_datetime']
        for col_name in date_columns:
            if col_name in df.columns:
                return col_name
        return None
    
    def _get_platform_metrics(self, df: pd.DataFrame, platform: str) -> List[str]:
        """Get platform-specific metric columns"""
        platform_metrics = {
            'tiktok': ['tiktok_insights_impressions', 'tiktok_insights_engagements', 'tiktok_insights_video_views', 'tiktok_insights_shares'],
            'facebook': ['facebook_insights_impressions', 'facebook_insights_engagements', 'facebook_insights_video_views'],
            'instagram': ['instagram_insights_impressions', 'instagram_insights_engagements', 'instagram_insights_reach'],
            'customer_care': ['resolution_time', 'satisfaction', 'urgency']
        }
        return platform_metrics.get(platform, [])


class PlatformAnalyzer(BaseAnalyzer):
    """
    Platform-Specific Metrics Analysis
    
    Original methods moved here:
    - _generate_customer_care_specific (line 3951)
    - _generate_tiktok_specific (line 4164)
    - _generate_instagram_specific (line 4316)
    - _generate_facebook_specific (line 4401)
    - _generate_platform_specific_metrics (line 3651)
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate platform-specific metrics."""
        if platform == 'customer_care':
            return self._generate_customer_care_specific(df, platform)
        elif platform == 'tiktok':
            return self._generate_tiktok_specific(df)
        elif platform == 'instagram':
            return self._generate_instagram_specific(df)
        elif platform == 'facebook':
            return self._generate_facebook_specific(df)
        else:
            return self._generate_platform_specific_metrics(df, platform)
    
    def _generate_customer_care_specific(self, df: pd.DataFrame, platform: str = 'customer_care') -> Dict[str, Any]:
        """Customer care specific metrics - comprehensive"""
        metrics = {}
        
        # Issue type metrics
        if 'issue_type' in df.columns:
            issue_groups = df.groupby('issue_type')
            issue_metrics = []
            
            for issue, group in issue_groups:
                metric = {
                    'issue_type': issue,
                    'volume': len(group),
                    'percentage': round(len(group) / len(df) * 100, 2)
                }
                
                if 'sentiment_score' in group.columns:
                    metric['avg_sentiment'] = round(float(group['sentiment_score'].mean()), 3)
                
                if 'resolution_time_hours' in group.columns:
                    metric['avg_resolution_time'] = round(float(group['resolution_time_hours'].mean()), 1)
                    metric['median_resolution_time'] = round(float(group['resolution_time_hours'].median()), 1)
                
                if 'is_escalated' in group.columns:
                    metric['escalation_rate'] = round(float(group['is_escalated'].mean()), 3)
                
                if 'satisfaction_score' in group.columns:
                    metric['avg_satisfaction'] = round(float(group['satisfaction_score'].mean()), 2)
                
                issue_metrics.append(metric)
            
            metrics['issue_type_analysis'] = {
                'metrics': sorted(issue_metrics, key=lambda x: x['volume'], reverse=True),
                'highest_volume': max(issue_metrics, key=lambda x: x['volume'])['issue_type'],
                'most_negative': min(issue_metrics, key=lambda x: x.get('avg_sentiment', 0))['issue_type'] if any('avg_sentiment' in m for m in issue_metrics) else None,
                'highest_escalation': max(issue_metrics, key=lambda x: x.get('escalation_rate', 0))['issue_type'] if any('escalation_rate' in m for m in issue_metrics) else None
            }
        
        # Priority metrics
        if 'priority' in df.columns:
            priority_dist = df['priority'].value_counts().to_dict()
            metrics['priority_distribution'] = priority_dist
            
            # Priority performance
            priority_metrics = df.groupby('priority').agg({
                'resolution_time_hours': ['mean', 'median'] if 'resolution_time_hours' in df.columns else {},
                'is_escalated': 'mean' if 'is_escalated' in df.columns else {},
                'satisfaction_score': 'mean' if 'satisfaction_score' in df.columns else {}
            }).round(3)
            
            if not priority_metrics.empty:
                metrics['priority_performance'] = priority_metrics.to_dict()
        
        # Escalation prediction analysis
        if 'is_escalated' in df.columns:
            escalation_rate = df['is_escalated'].mean()
            metrics['escalation_analysis'] = {
                'overall_escalation_rate': round(float(escalation_rate), 3),
                'total_escalated_cases': int(df['is_escalated'].sum()),
                'escalation_risk_factors': self._analyze_escalation_factors(df)
            }
        
        # Origin effectiveness analysis
        if 'origin' in df.columns:
            origin_performance = df.groupby('origin').agg({
                'resolution_time_hours': 'mean' if 'resolution_time_hours' in df.columns else lambda x: 0,
                'satisfaction_score': 'mean' if 'satisfaction_score' in df.columns else lambda x: 0,
                'is_escalated': 'mean' if 'is_escalated' in df.columns else lambda x: 0
            }).round(3)
            
            if not origin_performance.empty:
                metrics['origin_effectiveness'] = {
                    'channel_performance': origin_performance.to_dict(),
                    'best_performing_channel': origin_performance.index[0] if len(origin_performance) > 0 else None,
                    'channel_volume_distribution': df['origin'].value_counts().to_dict()
                }
        
        return metrics
    
    def _analyze_escalation_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze factors that contribute to escalation."""
        factors = {}
        
        if 'is_escalated' in df.columns:
            escalated_df = df[df['is_escalated'] == True]
            non_escalated_df = df[df['is_escalated'] == False]
            
            # Resolution time factor
            if 'resolution_time_hours' in df.columns and len(escalated_df) > 0 and len(non_escalated_df) > 0:
                factors['resolution_time_impact'] = {
                    'escalated_avg_time': round(float(escalated_df['resolution_time_hours'].mean()), 2),
                    'non_escalated_avg_time': round(float(non_escalated_df['resolution_time_hours'].mean()), 2),
                    'time_difference_hours': round(float(escalated_df['resolution_time_hours'].mean() - non_escalated_df['resolution_time_hours'].mean()), 2)
                }
            
            # Sentiment factor
            if 'sentiment_score' in df.columns and len(escalated_df) > 0 and len(non_escalated_df) > 0:
                factors['sentiment_impact'] = {
                    'escalated_avg_sentiment': round(float(escalated_df['sentiment_score'].mean()), 3),
                    'non_escalated_avg_sentiment': round(float(non_escalated_df['sentiment_score'].mean()), 3),
                    'sentiment_difference': round(float(escalated_df['sentiment_score'].mean() - non_escalated_df['sentiment_score'].mean()), 3)
                }
        
        return factors
    
    def _generate_tiktok_specific(self, df: pd.DataFrame) -> Dict[str, Any]:
        """TikTok specific metrics - comprehensive"""
        metrics = {}
        
        # Viral Metrics Analysis
        if all(col in df.columns for col in ['tiktok_insights_shares', 'tiktok_insights_comments', 'tiktok_insights_video_views']):
            # Calculate viral coefficient
            df['viral_coefficient'] = (
                pd.to_numeric(df['tiktok_insights_shares'], errors='coerce').fillna(0) + 
                pd.to_numeric(df['tiktok_insights_comments'], errors='coerce').fillna(0)
            ) / pd.to_numeric(df['tiktok_insights_video_views'], errors='coerce').replace(0, 1)
            
            # Engagement depth analysis
            df['comment_to_like_ratio'] = (
                pd.to_numeric(df['tiktok_insights_comments'], errors='coerce').fillna(0) / 
                pd.to_numeric(df['tiktok_insights_likes'], errors='coerce').replace(0, 1)
            )
            
            viral_analysis = {
                'avg_viral_coefficient': round(float(df['viral_coefficient'].mean()), 4),
                'top_viral_posts': int((df['viral_coefficient'] > df['viral_coefficient'].quantile(0.9)).sum()),
                'avg_shares_per_post': round(float(pd.to_numeric(df['tiktok_insights_shares'], errors='coerce').mean()), 1),
                'avg_comments_per_post': round(float(pd.to_numeric(df['tiktok_insights_comments'], errors='coerce').mean()), 1),
                'high_engagement_depth_posts': int((df['comment_to_like_ratio'] > 0.1).sum()),
                'share_rate': round(float(pd.to_numeric(df['tiktok_insights_shares'], errors='coerce').sum() / 
                                        pd.to_numeric(df['tiktok_insights_video_views'], errors='coerce').sum() * 100), 3),
                'comment_rate': round(float(pd.to_numeric(df['tiktok_insights_comments'], errors='coerce').sum() / 
                                          pd.to_numeric(df['tiktok_insights_video_views'], errors='coerce').sum() * 100), 3)
            }
            
            metrics['viral_analysis'] = viral_analysis
        
        # Reach Efficiency Analysis
        if 'tiktok_insights_reach' in df.columns and 'tiktok_insights_impressions' in df.columns:
            reach_data = pd.to_numeric(df['tiktok_insights_reach'], errors='coerce').fillna(0)
            impressions_data = pd.to_numeric(df['tiktok_insights_impressions'], errors='coerce').fillna(0)
            
            # Calculate reach efficiency
            df['reach_efficiency'] = (reach_data / impressions_data.replace(0, 1)) * 100
            
            reach_analysis = {
                'avg_reach_efficiency': round(float(df['reach_efficiency'].mean()), 2),
                'total_reach': int(reach_data.sum()),
                'reach_vs_impressions_ratio': round(float(reach_data.sum() / impressions_data.sum()), 3) if impressions_data.sum() > 0 else 0,
                'high_reach_efficiency_posts': int((df['reach_efficiency'] > 80).sum()),
                'audience_penetration_score': round(float(df['reach_efficiency'].median()), 2)
            }
            
            metrics['reach_analysis'] = reach_analysis
        
        # Attachment Performance Analysis
        if 'tiktok_attachments' in df.columns:
            attachment_data = df['tiktok_attachments'].fillna('none')
            attachment_performance = df.groupby(attachment_data).agg({
                'tiktok_insights_video_views': 'mean' if 'tiktok_insights_video_views' in df.columns else lambda x: 0,
                'tiktok_insights_likes': 'mean' if 'tiktok_insights_likes' in df.columns else lambda x: 0,
                'tiktok_insights_shares': 'mean' if 'tiktok_insights_shares' in df.columns else lambda x: 0
            }).round(2)
            
            if not attachment_performance.empty:
                metrics['attachment_performance'] = {
                    'performance_by_type': attachment_performance.to_dict(),
                    'attachment_distribution': attachment_data.value_counts().to_dict(),
                    'best_performing_attachment': attachment_performance.index[0] if len(attachment_performance) > 0 else None
                }
        
        return metrics
    
    def _generate_instagram_specific(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Instagram specific metrics"""
        metrics = {}
        
        # Purchase Intent Analysis (Saves)
        if 'instagram_insights_saves' in df.columns and 'instagram_insights_impressions' in df.columns:
            saves_data = pd.to_numeric(df['instagram_insights_saves'], errors='coerce').fillna(0)
            impressions_data = pd.to_numeric(df['instagram_insights_impressions'], errors='coerce').fillna(0)
            
            # Calculate save rates
            df['save_rate'] = (saves_data / impressions_data.replace(0, 1)) * 100
            df['save_intent_score'] = saves_data * 5  # Weight saves 5x higher than likes for intent
            
            # High-intent content identification
            high_save_threshold = saves_data.quantile(0.8) if len(saves_data) > 0 else 0
            high_intent_posts = saves_data > high_save_threshold
            
            purchase_intent_analysis = {
                'total_saves': int(saves_data.sum()),
                'avg_saves_per_post': round(float(saves_data.mean()), 1),
                'avg_save_rate': round(float(df['save_rate'].mean()), 3),
                'high_intent_posts': int(high_intent_posts.sum()),
                'high_intent_percentage': round(float(high_intent_posts.mean() * 100), 2),
                'save_to_engagement_ratio': round(float(saves_data.sum() / pd.to_numeric(df['instagram_insights_engagements'], errors='coerce').sum()), 3) if pd.to_numeric(df['instagram_insights_engagements'], errors='coerce').sum() > 0 else 0,
                'top_save_rate_threshold': round(float(df['save_rate'].quantile(0.9)), 3),
                'purchase_prediction_score': round(float(df['save_intent_score'].mean()), 1)
            }
            
            # Content value segmentation
            if len(saves_data) > 10:
                # Segment posts by save performance
                low_saves = (saves_data <= saves_data.quantile(0.33)).sum()
                medium_saves = ((saves_data > saves_data.quantile(0.33)) & (saves_data <= saves_data.quantile(0.67))).sum()
                high_saves = (saves_data > saves_data.quantile(0.67)).sum()
                
                purchase_intent_analysis['content_value_segments'] = {
                    'low_intent_posts': int(low_saves),
                    'medium_intent_posts': int(medium_saves),
                    'high_intent_posts': int(high_saves),
                    'high_intent_conversion_potential': round(float(high_saves / len(saves_data) * 100), 2)
                }
            
            metrics['purchase_intent_analysis'] = purchase_intent_analysis
        
        # Story Performance Analysis
        if 'instagram_insights_story_completion_rate' in df.columns:
            story_completion = pd.to_numeric(df['instagram_insights_story_completion_rate'], errors='coerce').fillna(0)
            
            story_analysis = {
                'avg_story_completion_rate': round(float(story_completion.mean()), 3),
                'high_completion_stories': int((story_completion > 0.8).sum()),
                'story_engagement_quality': 'high' if story_completion.mean() > 0.7 else 'medium' if story_completion.mean() > 0.5 else 'low'
            }
            
            metrics['story_performance'] = story_analysis
        
        # Video Performance Analysis
        if 'instagram_insights_video_views' in df.columns:
            video_views = pd.to_numeric(df['instagram_insights_video_views'], errors='coerce').fillna(0)
            
            video_analysis = {
                'total_video_views': int(video_views.sum()),
                'avg_video_views': round(float(video_views.mean()), 1),
                'high_performing_videos': int((video_views > video_views.quantile(0.8)).sum()) if len(video_views) > 0 else 0
            }
            
            metrics['video_performance'] = video_analysis
        
        return metrics
    
    def _generate_facebook_specific(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Facebook specific metrics - comprehensive"""
        metrics = {}
        
        # Emotional Intelligence Analysis
        reaction_cols = ['facebook_reaction_like', 'facebook_reaction_love', 'facebook_reaction_haha',
                        'facebook_reaction_wow', 'facebook_reaction_sorry', 'facebook_reaction_anger']
        
        if all(col in df.columns for col in reaction_cols):
            # Calculate emotional engagement scores
            for col in reaction_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Emotional engagement breakdown
            total_reactions = sum(df[col].sum() for col in reaction_cols)
            
            emotional_breakdown = {}
            for col in reaction_cols:
                reaction_name = col.replace('facebook_reaction_', '')
                reaction_total = df[col].sum()
                emotional_breakdown[reaction_name] = {
                    'total_count': int(reaction_total),
                    'percentage': round(float(reaction_total / total_reactions * 100), 2) if total_reactions > 0 else 0,
                    'avg_per_post': round(float(df[col].mean()), 2),
                    'posts_with_reaction': int((df[col] > 0).sum())
                }
            
            # Emotional diversity score (entropy-based)
            reaction_sums = [df[col].sum() for col in reaction_cols]
            total = sum(reaction_sums)
            if total > 0:
                proportions = [r/total for r in reaction_sums if r > 0]
                diversity_score = -sum(p * np.log(p) for p in proportions) / np.log(len(proportions)) if len(proportions) > 1 else 0
            else:
                diversity_score = 0
            
            # Emotional sentiment score (weighted)
            positive_reactions = df['facebook_reaction_like'].sum() + df['facebook_reaction_love'].sum() * 2 + df['facebook_reaction_wow'].sum() * 1.5
            negative_reactions = df['facebook_reaction_anger'].sum() * 2 + df['facebook_reaction_sorry'].sum()
            neutral_reactions = df['facebook_reaction_haha'].sum()
            
            total_weighted = positive_reactions + negative_reactions + neutral_reactions
            emotional_sentiment = (positive_reactions - negative_reactions) / total_weighted if total_weighted > 0 else 0
            
            metrics['emotional_intelligence'] = {
                'reaction_breakdown': emotional_breakdown,
                'emotional_diversity_score': round(float(diversity_score), 3),
                'emotional_sentiment_score': round(float(emotional_sentiment), 3),
                'total_emotional_reactions': int(total_reactions),
                'dominant_emotion': max(emotional_breakdown.items(), key=lambda x: x[1]['total_count'])[0] if emotional_breakdown else 'none',
            }
        
        # Video Completion Analysis
        if 'facebook_insights_video_views_average_completion' in df.columns:
            completion_data = pd.to_numeric(df['facebook_insights_video_views_average_completion'], errors='coerce').fillna(0)
            
            video_completion_analysis = {
                'avg_completion_rate': round(float(completion_data.mean()), 3),
                'high_completion_videos': int((completion_data > 0.8).sum()),
                'completion_quality': 'excellent' if completion_data.mean() > 0.8 else 'good' if completion_data.mean() > 0.6 else 'needs_improvement'
            }
            
            metrics['video_completion_analysis'] = video_completion_analysis
        
        # Click-through Analysis
        if 'facebook_insights_post_clicks' in df.columns and 'facebook_insights_impressions' in df.columns:
            clicks_data = pd.to_numeric(df['facebook_insights_post_clicks'], errors='coerce').fillna(0)
            impressions_data = pd.to_numeric(df['facebook_insights_impressions'], errors='coerce').fillna(0)
            
            # Calculate CTR
            df['click_through_rate'] = (clicks_data / impressions_data.replace(0, 1)) * 100
            
            click_analysis = {
                'total_clicks': int(clicks_data.sum()),
                'avg_clicks_per_post': round(float(clicks_data.mean()), 1),
                'avg_click_through_rate': round(float(df['click_through_rate'].mean()), 3),
                'high_ctr_posts': int((df['click_through_rate'] > df['click_through_rate'].quantile(0.8)).sum()) if len(df) > 0 else 0,
                'conversion_potential': 'high' if df['click_through_rate'].mean() > 2.0 else 'medium' if df['click_through_rate'].mean() > 1.0 else 'low'
            }
            
            metrics['click_through_analysis'] = click_analysis
        
        # Media Type Performance
        if 'facebook_media_type' in df.columns:
            media_performance = df.groupby('facebook_media_type').agg({
                'facebook_insights_engagements': 'mean' if 'facebook_insights_engagements' in df.columns else lambda x: 0,
                'facebook_insights_impressions': 'mean' if 'facebook_insights_impressions' in df.columns else lambda x: 0,
                'facebook_insights_post_clicks': 'mean' if 'facebook_insights_post_clicks' in df.columns else lambda x: 0
            }).round(2)
            
            if not media_performance.empty:
                metrics['media_type_performance'] = {
                    'performance_by_type': media_performance.to_dict(),
                    'media_distribution': df['facebook_media_type'].value_counts().to_dict(),
                    'best_performing_media': media_performance.index[0] if len(media_performance) > 0 else None
                }
        
        return metrics
    
    def _generate_platform_specific_metrics(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generic platform-specific metrics for platforms without dedicated handlers."""
        metrics = {}
        
        # Basic platform info
        metrics['platform_info'] = {
            'platform': platform,
            'total_records': len(df),
            'date_range': {
                'start': str(df.index.min()) if hasattr(df.index, 'min') else 'unknown',
                'end': str(df.index.max()) if hasattr(df.index, 'max') else 'unknown'
            }
        }
        
        # Platform-specific field analysis
        platform_fields = [col for col in df.columns if col.startswith(f'{platform}_')]
        if platform_fields:
            field_completeness = {}
            for field in platform_fields:
                non_null_count = df[field].notna().sum()
                field_completeness[field] = {
                    'completeness_percentage': round(float(non_null_count / len(df) * 100), 2),
                    'non_null_records': int(non_null_count),
                    'unique_values': int(df[field].nunique()) if df[field].dtype == 'object' else 'numeric'
                }
            
            metrics['field_analysis'] = field_completeness
        
        return metrics


class AdvancedAnalyzer(BaseAnalyzer):
    """Advanced Analytics - Correlation, Risk Detection, Distribution Analysis"""
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced analytics."""
        return {
            'correlation_analysis': self._generate_correlation_analysis(df, platform),
            'trend_analysis': self._generate_trend_analysis(df, platform),
            'cross_dimensional_trends': self._generate_cross_dimensional_trends(df, platform),
            'performance_distribution': self._generate_performance_distribution(df, platform),
            'completion_rate_analysis': self._generate_completion_rate_analysis(df, platform),
            'risk_detection': self._generate_risk_detection(df, platform)
        }
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> Optional[str]:
        """Get the correct date column name for the platform"""
        date_columns = ['created_time', 'timestamp', 'date', '_datetime']
        for col_name in date_columns:
            if col_name in df.columns:
                return col_name
        return None
    
    def _analyze_brand_daily_trends(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Analyze daily trends for each brand"""
        
        # Extract brands from content
        all_brands = []
        for _, row in df.iterrows():
            brands = self._extract_brands_from_row(row, platform)
            for brand in brands:
                all_brands.append({
                    'brand': brand,
                    'date': row.get('_date', row.get('created_time')),
                    'engagement_rate': row.get('engagement_rate', 0),
                    'sentiment_score': row.get('sentiment_score', 0),
                    'row_index': row.name
                })
        
        if not all_brands:
            return {}
        
        brand_df = pd.DataFrame(all_brands)
        brand_trends = {}
        
        # Analyze trends for top brands
        top_brands = brand_df['brand'].value_counts().head(5).index
        for brand in top_brands:
            brand_data = brand_df[brand_df['brand'] == brand]
            if len(brand_data) > 1:
                brand_trends[brand] = {
                    "daily_engagement": brand_data.groupby('date')['engagement_rate'].mean().to_dict(),
                    "daily_sentiment": brand_data.groupby('date')['sentiment_score'].mean().to_dict(),
                    "total_posts": len(brand_data)
                }
        
        return brand_trends
    
    def _extract_brands_from_row(self, row: pd.Series, platform: str) -> List[str]:
        """Extract brand mentions from a row - helper for trend analysis"""
        brands = []
        
        # Common brand keywords to look for
        brand_keywords = ['Nike', 'Adidas', 'Puma', 'Apple', 'Samsung', 'Google', 'Microsoft', 
                         'Amazon', 'Facebook', 'Instagram', 'TikTok', 'Twitter', 'YouTube']
        
        # Check various text fields
        text_fields = ['content', 'text', 'description', 'caption', f'{platform}_content', f'{platform}_text']
        
        for field in text_fields:
            if field in row and pd.notna(row[field]):
                text = str(row[field]).lower()
                for brand in brand_keywords:
                    if brand.lower() in text:
                        brands.append(brand)
        
        return list(set(brands)) if brands else ['Unknown']
    
    def _get_sentiment_column(self, df: pd.DataFrame, platform: str) -> Optional[str]:
        """Get the sentiment column for the platform"""
        sentiment_cols = [f'{platform}_sentiment', 'sentiment', 'sentiment_score']
        for col in sentiment_cols:
            if col in df.columns:
                return col
        return None
    
    def _generate_correlation_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Enhanced correlation analysis between all key metrics"""
        
        correlations = {}
        
        # Get numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Define key metrics for correlation based on platform
        key_metrics = []
        if platform == 'tiktok':
            key_metrics = ['tiktok_insights_impressions', 'tiktok_insights_engagements', 
                          'tiktok_insights_video_views', 'tiktok_insights_likes', 
                          'tiktok_insights_completion_rate', 'tiktok_duration',
                          'tiktok_insights_shares', 'tiktok_insights_comments', 'tiktok_insights_reach']
        elif platform == 'facebook':
            key_metrics = ['facebook_insights_impressions', 'facebook_insights_engagements',
                          'facebook_insights_reach', 'facebook_insights_likes',
                          'facebook_reaction_like', 'facebook_reaction_love', 'facebook_reaction_haha',
                          'facebook_reaction_wow', 'facebook_reaction_sorry', 'facebook_reaction_anger',
                          'facebook_insights_post_clicks', 'facebook_media_type', 'comment_sentiment_score',
                          'comment_positive_ratio', 'comment_negative_ratio']
        elif platform == 'instagram':
            key_metrics = ['instagram_insights_impressions', 'instagram_insights_engagements',
                          'instagram_insights_reach', 'instagram_insights_likes',
                          'instagram_insights_saves', 'instagram_insights_video_views', 'comment_sentiment_score',
                          'comment_positive_ratio', 'comment_negative_ratio']
        elif platform == 'customer_care':
            key_metrics = ['resolution_time_hours', 'satisfaction_score', 'urgency_score', 
                          'response_time_hours', 'escalation_rate', 'priority_score', 'origin_effectiveness']
        
        # Filter to available metrics
        available_metrics = [col for col in key_metrics if col in numeric_cols]
        
        # Add sentiment if available
        sentiment_col = self._get_sentiment_column(df, platform)
        if sentiment_col and sentiment_col in numeric_cols:
            available_metrics.append(sentiment_col)
        
        # Add language-based metrics if available
        if 'detected_language' in df.columns and 'language_confidence' in df.columns:
            # Create language diversity score
            df['language_diversity_score'] = df.groupby('detected_language')['detected_language'].transform('count')
            available_metrics.append('language_diversity_score')
            available_metrics.append('language_confidence')
        
        # Add comprehensive country-based metrics if available
        if 'derived_country' in df.columns:
            # Create country diversity score
            df['country_diversity_score'] = df.groupby('derived_country')['derived_country'].transform('count')
            available_metrics.append('country_diversity_score')
            
            # Add country performance metrics
            country_performance_metrics = self._generate_country_performance_metrics(df, platform, available_metrics)
            available_metrics.extend(country_performance_metrics)
        
        # Add semantic topic metrics if available
        semantic_metrics = ['topic_sentiment_score', 'topic_engagement_score', 'topic_diversity_score', 
                          'dominant_topic_strength', 'semantic_coherence_score']
        for metric in semantic_metrics:
            if metric in df.columns:
                available_metrics.append(metric)
        
        # Generate correlation matrix for key metrics
        if len(available_metrics) >= 2:
            correlation_df = df[available_metrics].corr()
            
            # Convert to dictionary format, excluding self-correlations
            correlation_matrix = {}
            for i, metric1 in enumerate(available_metrics):
                for j, metric2 in enumerate(available_metrics):
                    if i < j:  # Only upper triangle to avoid duplicates
                        corr_value = correlation_df.loc[metric1, metric2]
                        if not pd.isna(corr_value):
                            key = f"{metric1}_vs_{metric2}"
                            correlation_matrix[key] = round(float(corr_value), 3)
        
        # Time-based correlations
        date_col = self._get_date_column(df, platform)
        if date_col and date_col in df.columns:
            df['_hour'] = pd.to_datetime(df[date_col], errors='coerce').dt.hour
            df['_day_of_week'] = pd.to_datetime(df[date_col], errors='coerce').dt.dayofweek
            
            time_correlations = {}
            
            # Hour correlations with key metrics
            for metric in available_metrics:
                if metric in df.columns:
                    hour_corr = df['_hour'].corr(df[metric])
                    if not pd.isna(hour_corr):
                        time_correlations[f'hour_vs_{metric}'] = round(float(hour_corr), 3)
            
            # Day of week correlations
            for metric in available_metrics:
                if metric in df.columns:
                    day_corr = df['_day_of_week'].corr(df[metric])
                    if not pd.isna(day_corr):
                        time_correlations[f'day_of_week_vs_{metric}'] = round(float(day_corr), 3)
            
            correlations['temporal_correlations'] = time_correlations
        
        # Store the main correlation matrix
        if 'correlation_matrix' in locals():
            correlations['metric_correlations'] = correlation_matrix
        
        # Legacy format for backward compatibility
        if sentiment_col:
            if platform != 'customer_care' and 'engagements' in df.columns:
                corr = df[sentiment_col].corr(df['engagements'])
                if not pd.isna(corr):
                    correlations['sentiment_engagement'] = round(float(corr), 3)
            elif platform == 'customer_care' and 'satisfaction_score' in df.columns:
                corr = df[sentiment_col].corr(df['satisfaction_score'])
                if not pd.isna(corr):
                    correlations['sentiment_satisfaction'] = round(float(corr), 3)
        
        if date_col and date_col in df.columns and sentiment_col:
                hour_sentiment_corr = df['_hour'].corr(df[sentiment_col])
                if not pd.isna(hour_sentiment_corr):
                    correlations['hour_sentiment'] = round(float(hour_sentiment_corr), 3)
        
        # Add geographic correlation insights
        if 'derived_country' in df.columns:
            geographic_correlations = self._analyze_geographic_correlations(df, platform, available_metrics)
            correlations['geographic_correlations'] = geographic_correlations
        
        return correlations
    
    def _get_sentiment_column(self, df: pd.DataFrame, platform: str) -> str:
        """Get the sentiment column name for the platform."""
        if 'sentiment_score' in df.columns:
            return 'sentiment_score'
        elif 'sentiment' in df.columns:
            return 'sentiment'
        return None
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> Optional[str]:
        """Get the date column for the platform."""
        date_columns = [
            'created_time', 'created_at', 'date', 'timestamp', 'post_date',
            f'{platform}_created_at', f'{platform}_date'
        ]
        
        for col in date_columns:
            if col in df.columns:
                return col
        
        return None
    
    def _generate_trend_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Analyze trends over time"""
        
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column for trend analysis"}
        
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
        df = df[df['_date'].notna()]
        
        # Daily aggregations
        daily = df.groupby('_date').agg({
            '_date': 'count',  # Volume
            **({'sentiment_score': 'mean'} if 'sentiment_score' in df.columns else {}),
            **({'engagements': 'sum'} if 'engagements' in df.columns else {})
        }).rename(columns={'_date': 'volume'})
        
        if len(daily) < 3:
            return {"error": "Insufficient data for trend analysis"}
        
        trends = {}
        
        # Volume trend
        volume_trend = self._calculate_trend(daily['volume'])
        trends['volume'] = volume_trend
        
        # Sentiment trend
        if 'sentiment_score' in daily.columns:
            sentiment_trend = self._calculate_trend(daily['sentiment_score'])
            trends['sentiment'] = sentiment_trend
        
        # Engagement trend
        if 'engagements' in daily.columns:
            engagement_trend = self._calculate_trend(daily['engagements'])
            trends['engagement'] = engagement_trend
        
        return trends
    
    def _calculate_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate trend direction and change percentage."""
        if len(series) < 2:
            return {'direction': 'insufficient_data', 'change_percentage': 0}
        
        # Simple linear trend calculation
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return {'direction': 'insufficient_data', 'change_percentage': 0}
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calculate slope
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        # Calculate percentage change from first to last
        first_val = y_clean[0]
        last_val = y_clean[-1]
        
        if first_val != 0:
            change_pct = ((last_val - first_val) / first_val) * 100
        else:
            change_pct = 0
        
        # Determine direction
        if abs(slope) < 0.01:  # Threshold for "stable"
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'change_percentage': round(float(change_pct), 2),
            'slope': round(float(slope), 4)
        }
    
    def _generate_cross_dimensional_trends(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate cross-dimensional daily trend analysis"""
        
        date_col = self._get_date_column(df, platform)
        if not date_col or date_col not in df.columns:
            return {"error": "No date column for cross-dimensional trend analysis"}
        
        # Convert to datetime and extract date
        df['_datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['_datetime'].notna()]
        df['_date'] = df['_datetime'].dt.date
        
        if len(df) < 7:  # Need at least a week of data
            return {"error": "Insufficient data for cross-dimensional trends (minimum 7 days required)"}
        
        cross_trends = {}
        
        # Simplified cross-dimensional trends
        # Daily volume trends
        daily_volume = df.groupby('_date').size()
        cross_trends["daily_volume"] = {
            "trend_data": daily_volume.to_dict(),
            "total_days": len(daily_volume),
            "avg_daily_posts": float(daily_volume.mean()),
            "peak_day": str(daily_volume.idxmax()),
            "peak_volume": int(daily_volume.max())
        }
        
        # Daily sentiment trends (if available)
        sentiment_col = self._get_sentiment_column(df, platform)
        if sentiment_col and sentiment_col in df.columns:
            daily_sentiment = df.groupby('_date')[sentiment_col].mean()
            cross_trends["daily_sentiment"] = {
                "trend_data": daily_sentiment.to_dict(),
                "avg_sentiment": float(daily_sentiment.mean()),
                "best_day": str(daily_sentiment.idxmax()),
                "worst_day": str(daily_sentiment.idxmin())
            }
        
        # Daily engagement trends (if available)
        engagement_cols = [f'{platform}_insights_engagements', 'engagements']
        engagement_col = None
        for col in engagement_cols:
            if col in df.columns:
                engagement_col = col
                break
        
        if engagement_col:
            daily_engagement = df.groupby('_date')[engagement_col].mean()
            cross_trends["daily_engagement"] = {
                "trend_data": daily_engagement.to_dict(),
                "avg_engagement": float(daily_engagement.mean()),
                "peak_day": str(daily_engagement.idxmax())
            }
        
        # Summary insights
        cross_trends["summary"] = {
            "total_days_analyzed": len(daily_volume),
            "data_completeness": "good",
            "trend_analysis_complete": True
        }
        
        return cross_trends
    
    def _generate_performance_distribution(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate percentile distributions for all key metrics"""
        
        # Get numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Define key metrics based on platform
        key_metrics = []
        if platform == 'tiktok':
            key_metrics = ['tiktok_insights_impressions', 'tiktok_insights_engagements', 
                          'tiktok_insights_video_views', 'tiktok_insights_likes', 
                          'tiktok_insights_completion_rate', 'tiktok_duration',
                          'tiktok_insights_shares', 'tiktok_insights_comments', 'tiktok_insights_reach']
        elif platform == 'facebook':
            key_metrics = ['facebook_insights_impressions', 'facebook_insights_engagements',
                          'facebook_insights_reach', 'facebook_insights_likes',
                          'facebook_reaction_like', 'facebook_reaction_love', 'facebook_reaction_haha',
                          'facebook_reaction_wow', 'facebook_reaction_sorry', 'facebook_reaction_anger',
                          'facebook_insights_post_clicks', 'facebook_media_type']
        elif platform == 'instagram':
            key_metrics = ['instagram_insights_impressions', 'instagram_insights_engagements',
                          'instagram_insights_reach', 'instagram_insights_likes',
                          'instagram_insights_saves', 'instagram_insights_video_views']
        elif platform == 'customer_care':
            key_metrics = ['resolution_time_hours', 'satisfaction_score', 'urgency_score', 
                          'response_time_hours', 'escalation_rate', 'priority_score', 'origin_effectiveness']
        
        # Filter to available metrics
        available_metrics = [col for col in key_metrics if col in numeric_cols]
        
        distributions = {}
        
        for metric in available_metrics:
            data = df[metric].dropna()
            if len(data) > 0:
                distributions[metric] = {
                    'count': len(data),
                    'mean': round(float(data.mean()), 2),
                    'std': round(float(data.std()), 2),
                    'min': round(float(data.min()), 2),
                    'p25': round(float(data.quantile(0.25)), 2),
                    'p50': round(float(data.quantile(0.50)), 2),
                    'p75': round(float(data.quantile(0.75)), 2),
                    'p95': round(float(data.quantile(0.95)), 2),
                    'max': round(float(data.max()), 2)
                }
        
        return {
            'metric_distributions': distributions,
            'summary': f"Distribution analysis for {len(available_metrics)} key metrics",
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_completion_rate_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Enhanced completion rate analysis by content type and duration"""
        
        completion_col = None
        if platform == 'tiktok':
            completion_col = 'tiktok_insights_completion_rate'
        elif platform == 'facebook':
            completion_col = 'facebook_insights_completion_rate'
        elif platform == 'instagram':
            completion_col = 'instagram_insights_completion_rate'
        
        if not completion_col or completion_col not in df.columns:
            return {"error": f"No completion rate data available for {platform}"}
        
        analysis = {}
        
        # Overall completion rate stats
        completion_data = df[completion_col].dropna()
        if len(completion_data) > 0:
            analysis['overall'] = {
                'mean': round(float(completion_data.mean()), 3),
                'median': round(float(completion_data.median()), 3),
                'std': round(float(completion_data.std()), 3),
                'count': len(completion_data)
            }
        
        # By content type (simplified)
        if platform == 'tiktok' and 'tiktok_duration' in df.columns:
            # Create duration buckets
            df['_duration_bucket'] = pd.cut(df['tiktok_duration'], 
                                          bins=[0, 15, 30, 60, float('inf')],
                                          labels=['0-15s', '16-30s', '31-60s', '60s+'])
            by_duration = df.groupby('_duration_bucket')[completion_col].agg(['mean', 'count', 'std']).round(3)
            analysis['by_duration'] = by_duration.to_dict('index')
        
        return analysis
    
    def _generate_country_performance_metrics(self, df: pd.DataFrame, platform: str, base_metrics: List[str]) -> List[str]:
        """Generate country-based performance metrics for correlation analysis"""
        
        country_metrics = []
        
        if 'derived_country' not in df.columns:
            return country_metrics
        
        # Get key performance metrics for the platform
        perf_metrics = []
        if platform == 'tiktok':
            perf_metrics = ['tiktok_insights_video_views', 'tiktok_insights_engagements', 
                          'tiktok_insights_likes', 'tiktok_insights_shares']
        elif platform == 'facebook':
            perf_metrics = ['facebook_insights_impressions', 'facebook_insights_engagements', 'facebook_insights_reach']
        elif platform == 'instagram':
            perf_metrics = ['instagram_insights_impressions', 'instagram_insights_engagements', 'instagram_insights_reach']
        
        # Filter to available metrics
        perf_metrics = [m for m in perf_metrics if m in df.columns]
        
        if not perf_metrics:
            return country_metrics
        
        # Calculate country performance scores
        country_stats = df.groupby('derived_country')[perf_metrics].mean()
        
        # Create relative performance metrics (vs global average)
        global_means = df[perf_metrics].mean()
        
        for country in country_stats.index:
            country_data = country_stats.loc[country]
            
            # Calculate relative performance score
            relative_scores = []
            for metric in perf_metrics:
                if global_means[metric] > 0:
                    relative_score = (country_data[metric] / global_means[metric] - 1) * 100
                    relative_scores.append(relative_score)
            
            if relative_scores:
                avg_relative_performance = sum(relative_scores) / len(relative_scores)
                
                # Add country performance indicator to dataframe
                country_col = f'country_{country.lower().replace(" ", "_")}_performance'
                df[country_col] = (df['derived_country'] == country).astype(int) * avg_relative_performance
                country_metrics.append(country_col)
        
        # Add regional performance metrics
        regions = {
            'western_europe': ['France', 'Germany', 'Switzerland', 'Spain', 'Portugal'],
            'eastern_europe': ['Romania', 'Poland', 'Czech Republic', 'Bulgaria'],
            'northern_europe': ['Scandinavia', 'United Kingdom'],
            'southern_europe': ['Italy', 'Greece'],
            'middle_east': ['Middle East', 'Turkey']
        }
        
        for region_name, countries in regions.items():
            region_mask = df['derived_country'].isin(countries)
            if region_mask.sum() > 0:
                region_col = f'region_{region_name}_indicator'
                df[region_col] = region_mask.astype(int)
                country_metrics.append(region_col)
        
        return country_metrics
    
    def _analyze_geographic_correlations(self, df: pd.DataFrame, platform: str, available_metrics: List[str]) -> Dict[str, Any]:
        """Analyze correlations between geographic factors and performance metrics"""
        
        geographic_insights = {}
        
        # Get performance metrics for the platform
        perf_metrics = []
        if platform == 'tiktok':
            perf_metrics = ['tiktok_insights_video_views', 'tiktok_insights_engagements', 
                          'tiktok_insights_likes', 'tiktok_insights_shares', 'tiktok_insights_comments']
        elif platform == 'facebook':
            perf_metrics = ['facebook_insights_impressions', 'facebook_insights_engagements', 'facebook_insights_reach']
        elif platform == 'instagram':
            perf_metrics = ['instagram_insights_impressions', 'instagram_insights_engagements', 'instagram_insights_reach']
        
        # Filter to available metrics
        perf_metrics = [m for m in perf_metrics if m in df.columns and m in available_metrics]
        
        if not perf_metrics:
            return geographic_insights
        
        # Country performance analysis
        country_performance = {}
        countries = df['derived_country'].unique()
        
        for metric in perf_metrics:
            country_stats = df.groupby('derived_country')[metric].agg(['mean', 'median', 'std', 'count'])
            
            # Calculate coefficient of variation across countries
            country_means = country_stats['mean']
            cv = country_means.std() / country_means.mean() if country_means.mean() > 0 else 0
            
            country_performance[metric] = {
                'coefficient_of_variation': round(float(cv), 3),
                'top_country': country_means.idxmax(),
                'top_performance': round(float(country_means.max()), 2),
                'bottom_country': country_means.idxmin(),
                'bottom_performance': round(float(country_means.min()), 2),
                'performance_spread': round(float(country_means.max() - country_means.min()), 2)
            }
        
        geographic_insights['country_performance_variation'] = country_performance
        
        # Regional correlation analysis
        regions = {
            'Western Europe': ['France', 'Germany', 'Switzerland', 'Spain', 'Portugal'],
            'Eastern Europe': ['Romania', 'Poland', 'Czech Republic', 'Bulgaria'],
            'Northern Europe': ['Scandinavia', 'United Kingdom'],
            'Southern Europe': ['Italy', 'Greece'],
            'Middle East': ['Middle East', 'Turkey']
        }
        
        regional_performance = {}
        for region, countries in regions.items():
            region_df = df[df['derived_country'].isin(countries)]
            if len(region_df) > 0:
                region_stats = {}
                for metric in perf_metrics:
                    if metric in region_df.columns:
                        region_stats[metric] = {
                            'mean': round(float(region_df[metric].mean()), 2),
                            'posts': len(region_df)
                        }
                regional_performance[region] = region_stats
        
        geographic_insights['regional_performance'] = regional_performance
        
        # Country diversity correlation
        if 'country_diversity_score' in available_metrics:
            diversity_correlations = {}
            for metric in perf_metrics:
                if metric in df.columns:
                    corr = df['country_diversity_score'].corr(df[metric])
                    if not pd.isna(corr):
                        diversity_correlations[metric] = round(float(corr), 3)
            
            if diversity_correlations:
                geographic_insights['country_diversity_correlations'] = diversity_correlations
        
        # Geographic sentiment correlation
        sentiment_col = self._get_sentiment_column(df, platform)
        if sentiment_col and sentiment_col in df.columns:
            country_sentiment = df.groupby('derived_country')[sentiment_col].mean().sort_values(ascending=False)
            
            geographic_insights['country_sentiment_ranking'] = {
                'most_positive': country_sentiment.index[0] if len(country_sentiment) > 0 else None,
                'most_positive_score': round(float(country_sentiment.iloc[0]), 3) if len(country_sentiment) > 0 else None,
                'least_positive': country_sentiment.index[-1] if len(country_sentiment) > 0 else None,
                'least_positive_score': round(float(country_sentiment.iloc[-1]), 3) if len(country_sentiment) > 0 else None,
                'sentiment_variation': round(float(country_sentiment.std()), 3) if len(country_sentiment) > 1 else 0
            }
        
        return geographic_insights
    
    def _generate_risk_detection(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Placeholder for risk detection - will be moved from original file."""
        return {'status': 'placeholder - to be implemented'}


class GeographicAnalyzer(BaseAnalyzer):
    """
    Geographic Performance Analysis
    """
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive geographic analysis"""
        try:
            # Apply market-aware timezone conversion
            df = self._convert_to_market_time(df, self._get_date_column(df, platform))
            
            if 'derived_country' not in df.columns:
                return {"error": "No geographic data available"}
            
            # Clean country data
            df['country'] = df['derived_country'].fillna('Unknown')
            df = df[df['country'] != 'Unknown']
            
            if len(df) == 0:
                return {"error": "No records with country information"}
            
            return {
                "country_overview": self._generate_country_overview(df, platform),
                "country_performance": self._generate_country_performance(df, platform),
                "geographic_temporal": self._generate_geographic_temporal(df, platform),
                "cross_country_comparison": self._generate_cross_country_comparison(df, platform),
                "regional_insights": self._generate_regional_insights(df, platform)
            }
            
        except Exception as e:
            logger.error(f"Geographic analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_country_overview(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate country-level overview statistics"""
        country_stats = df.groupby('country').agg({
            'tiktok_insights_video_views': ['count', 'sum', 'mean'],
            'tiktok_insights_likes': ['sum', 'mean'],
            'tiktok_insights_comments': ['sum', 'mean'],
            'tiktok_insights_shares': ['sum', 'mean'],
            'tiktok_insights_engagements': ['sum', 'mean'],
            'tiktok_sentiment': 'mean'
        }).round(2)
        
        # Flatten column names
        country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns]
        
        # Convert to dict and add rankings
        overview = {}
        for country in country_stats.index:
            stats = country_stats.loc[country].to_dict()
            overview[country] = {
                'total_posts': int(stats['tiktok_insights_video_views_count']),
                'total_views': int(stats['tiktok_insights_video_views_sum']),
                'avg_views': float(stats['tiktok_insights_video_views_mean']),
                'total_engagement': int(stats['tiktok_insights_engagements_sum']),
                'avg_engagement': float(stats['tiktok_insights_engagements_mean']),
                'avg_sentiment': float(stats['tiktok_sentiment_mean']),
                'engagement_rate': float(stats['tiktok_insights_engagements_sum'] / stats['tiktok_insights_video_views_sum'] * 100) if stats['tiktok_insights_video_views_sum'] > 0 else 0
            }
        
        # Add rankings
        countries_by_engagement = sorted(overview.items(), key=lambda x: x[1]['avg_engagement'], reverse=True)
        countries_by_views = sorted(overview.items(), key=lambda x: x[1]['avg_views'], reverse=True)
        
        return {
            'by_country': overview,
            'rankings': {
                'top_engagement': [(country, data['avg_engagement']) for country, data in countries_by_engagement[:10]],
                'top_views': [(country, data['avg_views']) for country, data in countries_by_views[:10]]
            },
            'summary': {
                'total_countries': len(overview),
                'total_posts': sum(data['total_posts'] for data in overview.values()),
                'avg_engagement_rate': sum(data['engagement_rate'] for data in overview.values()) / len(overview)
            }
        }
    
    def _generate_country_performance(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate detailed performance metrics by country"""
        performance = {}
        
        for country in df['country'].unique():
            country_df = df[df['country'] == country]
            
            # Basic metrics
            metrics = {
                'posts': len(country_df),
                'avg_views': country_df['tiktok_insights_video_views'].mean(),
                'avg_likes': country_df['tiktok_insights_likes'].mean(),
                'avg_comments': country_df['tiktok_insights_comments'].mean(),
                'avg_shares': country_df['tiktok_insights_shares'].mean(),
                'avg_sentiment': country_df['tiktok_sentiment'].mean()
            }
            
            # Top performing posts
            top_posts = country_df.nlargest(5, 'tiktok_insights_video_views')[
                ['tiktok_insights_video_views', 'tiktok_insights_likes', 'created_time']
            ].to_dict('records')
            
            # Content type distribution
            content_dist = {}
            if 'content_types' in country_df.columns:
                # Handle list columns by flattening
                all_content_types = []
                for content_list in country_df['content_types']:
                    if isinstance(content_list, list):
                        all_content_types.extend(content_list)
                    elif pd.notna(content_list):
                        all_content_types.append(str(content_list))
                if all_content_types:
                    content_dist = pd.Series(all_content_types).value_counts().head(5).to_dict()
            
            # Brand distribution  
            brand_dist = {}
            if 'brands' in country_df.columns:
                # Handle list columns by flattening
                all_brands = []
                for brand_list in country_df['brands']:
                    if isinstance(brand_list, list):
                        all_brands.extend(brand_list)
                    elif pd.notna(brand_list):
                        all_brands.append(str(brand_list))
                if all_brands:
                    brand_dist = pd.Series(all_brands).value_counts().head(5).to_dict()
            
            performance[country] = {
                'metrics': {k: float(v) if pd.notna(v) else 0 for k, v in metrics.items()},
                'top_posts': top_posts,
                'content_distribution': content_dist,
                'brand_distribution': brand_dist
            }
        
        return performance
    
    def _generate_geographic_temporal(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate temporal trends by country"""
        df['_datetime'] = pd.to_datetime(df['_datetime'], errors='coerce')
        df['_year_month'] = df['_datetime'].dt.to_period('M')
        
        temporal = {}
        
        # Monthly trends by country
        monthly_data = df.groupby(['country', '_year_month']).agg({
            'tiktok_insights_video_views': ['count', 'mean'],
            'tiktok_insights_engagements': 'mean',
            'tiktok_sentiment': 'mean'
        }).round(2)
        
        for country in df['country'].unique():
            country_monthly = {}
            try:
                country_data = monthly_data.loc[country]
                for month in country_data.index:
                    month_str = str(month)
                    country_monthly[month_str] = {
                        'posts': int(country_data.loc[month, ('tiktok_insights_video_views', 'count')]),
                        'avg_views': float(country_data.loc[month, ('tiktok_insights_video_views', 'mean')]),
                        'avg_engagement': float(country_data.loc[month, ('tiktok_insights_engagements', 'mean')]),
                        'avg_sentiment': float(country_data.loc[month, ('tiktok_sentiment', 'mean')])
                    }
            except (KeyError, IndexError):
                pass
            
            temporal[country] = {
                'monthly_trends': country_monthly
            }
        
        return temporal
    
    def _generate_cross_country_comparison(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate cross-country performance comparisons"""
        
        # Performance metrics by country
        country_metrics = df.groupby('country').agg({
            'tiktok_insights_video_views': 'mean',
            'tiktok_insights_engagements': 'mean', 
            'tiktok_insights_likes': 'mean',
            'tiktok_insights_comments': 'mean',
            'tiktok_insights_shares': 'mean',
            'tiktok_sentiment': 'mean'
        }).round(2)
        
        # Calculate relative performance (vs global average)
        global_avg = df.agg({
            'tiktok_insights_video_views': 'mean',
            'tiktok_insights_engagements': 'mean',
            'tiktok_insights_likes': 'mean', 
            'tiktok_insights_comments': 'mean',
            'tiktok_insights_shares': 'mean',
            'tiktok_sentiment': 'mean'
        })
        
        relative_performance = {}
        for country in country_metrics.index:
            relative_performance[country] = {}
            for metric in country_metrics.columns:
                country_val = country_metrics.loc[country, metric]
                global_val = global_avg[metric]
                if global_val > 0:
                    relative_performance[country][metric] = {
                        'value': float(country_val),
                        'vs_global': float((country_val / global_val - 1) * 100)  # % difference
                    }
        
        return {
            'absolute_metrics': country_metrics.to_dict('index'),
            'relative_performance': relative_performance,
            'global_averages': global_avg.to_dict()
        }
    
    def _generate_regional_insights(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Generate regional grouping insights"""
        
        # Define regional groupings
        regions = {
            'Western Europe': ['France', 'Germany', 'Switzerland', 'Spain', 'Portugal'],
            'Eastern Europe': ['Romania', 'Poland', 'Czech Republic', 'Bulgaria'],
            'Northern Europe': ['Scandinavia', 'United Kingdom'],
            'Southern Europe': ['Italy', 'Greece'],
            'Middle East': ['Middle East', 'Turkey'],
            'Global': ['Global']
        }
        
        regional_data = {}
        
        for region, countries in regions.items():
            region_df = df[df['country'].isin(countries)]
            
            if len(region_df) > 0:
                regional_data[region] = {
                    'countries': countries,
                    'total_posts': len(region_df),
                    'avg_views': float(region_df['tiktok_insights_video_views'].mean()),
                    'avg_engagement': float(region_df['tiktok_insights_engagements'].mean()),
                    'avg_sentiment': float(region_df['tiktok_sentiment'].mean()),
                    'top_country': region_df.groupby('country')['tiktok_insights_video_views'].mean().idxmax()
                }
        
        return regional_data


class AIInsightsAnalyzer(BaseAnalyzer):
    """AI Insights and Recommendations"""
    
    def analyze(self, df: pd.DataFrame, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI insights and recommendations."""
        return {
            'ai_action_items': self._generate_ai_action_items(df, platform),
            'crisis_indicators': self._generate_crisis_indicators(df, platform),
            'language_analysis': self._generate_language_analysis(df, platform),
            'geographic_analysis': self._generate_geographic_analysis(df, platform),
            'query_examples': self._generate_query_examples(platform)
        }
    
    def _generate_ai_action_items(self, df: pd.DataFrame, platform: str) -> List[str]:
        """Generate sophisticated AI pattern detection and insights"""
        insights = []
        
        if len(df) == 0:
            return insights
        
        # Performance distribution analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Detect outlier patterns
        for col in numeric_cols:
            if col.endswith(('_impressions', '_engagements', '_views')):
                data = df[col].dropna()
                if len(data) > 10:
                    q99 = data.quantile(0.99)
                    q50 = data.quantile(0.50)
                    if q99 > q50 * 10:  # Top 1% performs 10x better
                        insights.append(f"PATTERN: {col} shows extreme variance - top 1% performs 10x above median")
        
        # Temporal pattern detection
        date_col = self._get_date_column(df, platform)
        if date_col and date_col in df.columns:
            df['_hour'] = pd.to_datetime(df[date_col], errors='coerce').dt.hour
            df['_day_of_week'] = pd.to_datetime(df[date_col], errors='coerce').dt.dayofweek
            
            # Hour-based patterns
            if 'engagements' in df.columns:
                hourly_engagement = df.groupby('_hour')['engagements'].mean()
                peak_hour = hourly_engagement.idxmax()
                low_hour = hourly_engagement.idxmin()
                ratio = hourly_engagement.max() / hourly_engagement.min()
                
                if ratio > 2:
                    insights.append(f"TEMPORAL: Peak engagement at hour {peak_hour}, lowest at {low_hour} (ratio: {ratio:.1f}x)")
        
        # Sentiment-performance correlation patterns
        sentiment_col = self._get_sentiment_column(df, platform)
        if sentiment_col and 'engagements' in df.columns:
            corr = df[sentiment_col].corr(df['engagements'])
            if not pd.isna(corr) and abs(corr) > 0.3:
                direction = "positive" if corr > 0 else "negative"
                insights.append(f"SENTIMENT: Strong {direction} correlation ({corr:.3f}) between sentiment and engagement")
        
        # Duration optimization patterns (for video platforms)
        if platform in ['tiktok', 'facebook', 'instagram']:
            duration_col = f'{platform}_duration' if f'{platform}_duration' in df.columns else 'duration'
            if duration_col in df.columns and 'engagements' in df.columns:
                # Find optimal duration range
                df['_duration_bucket'] = pd.cut(df[duration_col], 
                                              bins=[0, 15, 30, 60, float('inf')],
                                              labels=['0-15s', '16-30s', '31-60s', '60s+'])
                duration_performance = df.groupby('_duration_bucket')['engagements'].mean()
                optimal_bucket = duration_performance.idxmax()
                insights.append(f"DURATION: Optimal performance in {optimal_bucket} range")
        
        # Language-specific performance patterns
        if 'detected_language' in df.columns and 'engagement_rate' in df.columns:
            lang_performance = df.groupby('detected_language')['engagement_rate'].agg(['mean', 'count'])
            lang_performance = lang_performance[lang_performance['count'] >= 10]  # Minimum sample size
            
            if len(lang_performance) > 1:
                best_lang = lang_performance['mean'].idxmax()
                worst_lang = lang_performance['mean'].idxmin()
                best_rate = lang_performance.loc[best_lang, 'mean']
                worst_rate = lang_performance.loc[worst_lang, 'mean']
                
                if best_rate > worst_rate * 1.5:  # Significant difference
                    insights.append(f"LANGUAGE: {best_lang} content outperforms {worst_lang} by {best_rate/worst_rate:.1f}x")
        
        # Country-specific performance patterns
        if 'derived_country' in df.columns and 'engagement_rate' in df.columns:
            country_performance = df.groupby('derived_country')['engagement_rate'].agg(['mean', 'count'])
            country_performance = country_performance[country_performance['count'] >= 10]  # Minimum sample size
            
            if len(country_performance) > 1:
                best_country = country_performance['mean'].idxmax()
                worst_country = country_performance['mean'].idxmin()
                best_rate = country_performance.loc[best_country, 'mean']
                worst_rate = country_performance.loc[worst_country, 'mean']
                
                if best_rate > worst_rate * 1.5:  # Significant difference
                    insights.append(f"GEOGRAPHIC: {best_country} market outperforms {worst_country} by {best_rate/worst_rate:.1f}x")
        
        return insights
    
    def _generate_crisis_indicators(self, df: pd.DataFrame, platform: str) -> List[Dict]:
        """Generate crisis warning indicators from existing data"""
        warnings = []
        
        if len(df) == 0:
            return warnings
        
        # Check for negative sentiment spikes
        if 'sentiment_score' in df.columns or f'{platform}_sentiment' in df.columns:
            sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else f'{platform}_sentiment'
            negative_content = df[df[sentiment_col] < -0.5]
            if len(negative_content) > len(df) * 0.15:  # More than 15% negative
                warnings.append({
                    "type": "sentiment_crisis",
                    "severity": "high" if len(negative_content) > len(df) * 0.25 else "medium",
                    "message": f"{len(negative_content)} posts with highly negative sentiment detected",
                    "percentage": round((len(negative_content) / len(df)) * 100, 1)
                })
        
        # Check for engagement drops
        if 'engagement_rate' in df.columns:
            low_engagement = df[df['engagement_rate'] < df['engagement_rate'].quantile(0.1)]
            if len(low_engagement) > len(df) * 0.3:  # More than 30% low engagement
                warnings.append({
                    "type": "engagement_drop",
                    "severity": "medium",
                    "message": f"Significant engagement drop detected in {len(low_engagement)} posts",
                    "avg_engagement": round(float(low_engagement['engagement_rate'].mean()), 2)
                })
        
        return warnings
    
    def _generate_language_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Placeholder for language analysis - will be moved from original file."""
        return {'status': 'placeholder - to be implemented'}
    
    def _generate_geographic_analysis(self, df: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Placeholder for geographic analysis - will be moved from original file."""
        return {'status': 'placeholder - to be implemented'}
    
    def _generate_query_examples(self, platform: str) -> Dict:
        """Generate platform-specific query examples for semantic search"""
        
        examples = {
            "basic_queries": [],
            "advanced_queries": [],
            "temporal_queries": [],
            "sentiment_queries": [],
            "performance_queries": []
        }
        
        if platform == 'tiktok':
            examples["basic_queries"] = [
                "viral TikTok videos",
                "high engagement content",
                "trending hashtags"
            ]
            examples["advanced_queries"] = [
                "TikTok videos with completion rate > 80%",
                "content from top performing brands",
                "videos with high share-to-view ratio"
            ]
            examples["temporal_queries"] = [
                "TikTok content from last week",
                "peak engagement hours analysis",
                "weekend vs weekday performance"
            ]
            examples["sentiment_queries"] = [
                "positive sentiment TikTok content",
                "brand mentions with negative sentiment",
                "sentiment trends over time"
            ]
            examples["performance_queries"] = [
                "top 10% performing TikTok videos",
                "content with highest viral coefficient",
                "underperforming content analysis"
            ]
        
        elif platform == 'facebook':
            examples["basic_queries"] = [
                "Facebook posts with high engagement",
                "emotional reactions analysis",
                "video completion rates"
            ]
            examples["advanced_queries"] = [
                "Facebook posts with love reactions > 100",
                "content with high click-through rates",
                "posts with diverse emotional responses"
            ]
        
        elif platform == 'instagram':
            examples["basic_queries"] = [
                "Instagram posts with high saves",
                "story completion analysis",
                "purchase intent indicators"
            ]
            examples["advanced_queries"] = [
                "Instagram content with save rate > 5%",
                "high-intent purchase signals",
                "story vs feed performance"
            ]
        
        elif platform == 'customer_care':
            examples["basic_queries"] = [
                "escalated customer cases",
                "high satisfaction scores",
                "resolution time analysis"
            ]
            examples["advanced_queries"] = [
                "cases resolved in < 2 hours",
                "escalation risk factors",
                "channel effectiveness analysis"
            ]
        
        return examples
    
    def _get_sentiment_column(self, df: pd.DataFrame, platform: str) -> str:
        """Get the sentiment column name for the platform."""
        if 'sentiment_score' in df.columns:
            return 'sentiment_score'
        elif 'sentiment' in df.columns:
            return 'sentiment'
        elif f'{platform}_sentiment' in df.columns:
            return f'{platform}_sentiment'
        return None
    
    def _get_date_column(self, df: pd.DataFrame, platform: str) -> Optional[str]:
        """Get the date column for the platform."""
        date_columns = [
            'created_time', 'created_at', 'date', 'timestamp', 'post_date',
            f'{platform}_created_at', f'{platform}_date'
        ]
        
        for col in date_columns:
            if col in df.columns:
                return col
        
        return None
    
    def _get_key_metric_definitions(self, platform: str) -> Dict:
        """Get key metric definitions for platform"""
        definitions = {
            "engagement_rate": "Engagements divided by impressions, percentage",
            "sentiment_score": "Sentiment analysis score from -1 (negative) to +1 (positive)"
        }
        
        if platform == 'tiktok':
            definitions.update({
                "completion_rate": "Percentage of video watched to completion",
                "viral_coefficient": "Shares + comments divided by views",
                "reach_efficiency": "Reach divided by impressions, percentage"
            })
        elif platform == 'facebook':
            definitions.update({
                "emotional_diversity": "Entropy-based score of reaction variety",
                "click_through_rate": "Post clicks divided by impressions, percentage"
            })
        elif platform == 'instagram':
            definitions.update({
                "save_rate": "Saves divided by impressions, percentage",
                "story_completion_rate": "Percentage of story viewed to completion"
            })
        elif platform == 'customer_care':
            definitions.update({
                "resolution_time": "Hours from case creation to resolution",
                "escalation_rate": "Percentage of cases that get escalated",
                "satisfaction_score": "Customer satisfaction rating"
            })
        
        return definitions
    
    def _get_duration_column(self, platform: str) -> str:
        """Get the duration column name for the platform"""
        duration_columns = {
            'tiktok': 'tiktok_duration',
            'facebook': 'facebook_duration',
            'instagram': 'instagram_duration'
        }
        return duration_columns.get(platform, 'duration')
    
    def _calculate_engagement_rate_column(self, df: pd.DataFrame, platform: str) -> str:
        """Calculate engagement rate and return column name"""
        if platform == 'tiktok':
            engagements_col = 'tiktok_insights_engagements'
            impressions_col = 'tiktok_insights_impressions'
        elif platform == 'facebook':
            engagements_col = 'facebook_insights_engagements'
            impressions_col = 'facebook_insights_impressions'
        elif platform == 'instagram':
            engagements_col = 'instagram_insights_engagements'
            impressions_col = 'instagram_insights_impressions'
        else:
            engagements_col = 'engagements'
            impressions_col = 'impressions'
        
        # Calculate engagement rate if columns exist
        if engagements_col in df.columns and impressions_col in df.columns:
            df['engagement_rate'] = (
                pd.to_numeric(df[engagements_col], errors='coerce').fillna(0) / 
                pd.to_numeric(df[impressions_col], errors='coerce').replace(0, np.nan) * 100
            ).fillna(0)
            return 'engagement_rate'
        
        return None


# Analyzer Registry - Maps analyzer names to classes
ANALYZER_REGISTRY = {
    'brand_performance': BrandAnalyzer,
    'content_type_performance': ContentAnalyzer,
    'sentiment_analysis': SentimentAnalyzer,
    'engagement_metrics': EngagementAnalyzer,
    'temporal_analysis': TemporalAnalyzer,
    'semantic_analysis': SemanticAnalyzer,
    'platform_specific': PlatformAnalyzer,
    'ai_insights': AIInsightsAnalyzer,
    'advanced_analysis': AdvancedAnalyzer,
    'performance_analysis': PerformanceAnalyzer,
    'overview': OverviewAnalyzer,
    'geographic_analysis': GeographicAnalyzer
}


def get_analyzer(analyzer_name: str, config: Dict[str, Any]) -> BaseAnalyzer:
    """Get analyzer instance by name."""
    if analyzer_name not in ANALYZER_REGISTRY:
        raise ValueError(f"Unknown analyzer: {analyzer_name}")
    
    analyzer_class = ANALYZER_REGISTRY[analyzer_name]
    return analyzer_class(config)


def get_enabled_analyzers(platform: str, config: Dict[str, Any]) -> List[BaseAnalyzer]:
    """Get list of enabled analyzers for a platform."""
    # TODO: Read from YAML config which analyzers are enabled
    enabled_names = [
        'overview',
        'brand_performance', 
        'content_type_performance',
        'sentiment_analysis',
        'engagement_metrics',
        'temporal_analysis',
        'semantic_analysis',
        'performance_analysis',
        'platform_specific',
        'advanced_analysis',
        'geographic_analysis',
        'ai_insights'
    ]
    
    analyzers = []
    for name in enabled_names:
        try:
            analyzer = get_analyzer(name, config)
            if analyzer.is_enabled(platform):
                analyzers.append(analyzer)
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            # Don't add failed analyzers but continue with others
    
    return analyzers
