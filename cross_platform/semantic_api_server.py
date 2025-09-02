#!/usr/bin/env python3
"""
FastAPI server for enterprise semantic search
Integrates with existing Weaviate architecture
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import asyncio
import json
from pathlib import Path

from semantic_search_engine import (
    EnterpriseSemanticSearchEngine, 
    SemanticQueryInterface,
    SearchType, 
    SearchScope, 
    SearchFilters
)

app = FastAPI(
    title="Sephora Social Intelligence API",
    description="Enterprise semantic search across 1.2M+ social media and customer care records",
    version="1.0.0"
)

# CORS middleware for web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================== REQUEST MODELS ===================

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    search_type: str = Field("hybrid", description="Search type: semantic, hybrid, keyword, analytics")
    scope: str = Field("all", description="Search scope: social, care, all, cross")
    platforms: Optional[List[str]] = Field(None, description="Filter by platforms: tiktok, facebook, instagram")
    brands: Optional[List[str]] = Field(None, description="Filter by brands")
    content_types: Optional[List[str]] = Field(None, description="Filter by content types")
    date_range: Optional[List[str]] = Field(None, description="Date range [start_date, end_date] in ISO format")
    engagement_threshold: Optional[float] = Field(None, description="Minimum engagement rate")
    viral_only: bool = Field(False, description="Show only viral content")
    limit: int = Field(50, ge=1, le=500, description="Maximum results")
    include_analytics: bool = Field(True, description="Include analytics insights")
    # New multilingual and transparency features
    languages: Optional[List[str]] = Field(None, description="Filter by languages: en, es, pl, ro, fr, de, it, pt")
    countries: Optional[List[str]] = Field(None, description="Filter by countries: France, Global, USA, UK, Spain, etc.")
    topic_ids: Optional[List[str]] = Field(None, description="Filter by semantic topic IDs")
    topic_labels: Optional[List[str]] = Field(None, description="Filter by semantic topic labels")
    sentiment_range: Optional[List[float]] = Field(None, description="Sentiment range [min, max] between -1.0 and 1.0")
    cross_lingual: bool = Field(False, description="Enable cross-language search")
    cluster_results: bool = Field(True, description="Enable result clustering to reduce redundancy")
    include_explanations: bool = Field(True, description="Include match explanations and confidence scores")

class ContentDiscoveryRequest(BaseModel):
    content_themes: List[str] = Field(..., description="Content themes to discover")
    performance_criteria: Dict[str, float] = Field({}, description="Performance criteria filters")
    platforms: Optional[List[str]] = Field(None, description="Platform filters")
    limit: int = Field(100, ge=1, le=1000)

class CompetitiveIntelRequest(BaseModel):
    competitor_keywords: List[str] = Field(..., description="Competitor keywords to analyze")
    time_window_days: int = Field(30, ge=1, le=365, description="Analysis time window")
    platforms: Optional[List[str]] = Field(None, description="Platform filters")

class CrisisDetectionRequest(BaseModel):
    sentiment_threshold: float = Field(-0.5, description="Negative sentiment threshold")
    engagement_spike_threshold: float = Field(200.0, description="Engagement spike threshold")
    time_window_hours: int = Field(24, ge=1, le=168, description="Detection time window")

class TrendAnalysisRequest(BaseModel):
    emerging_topics: List[str] = Field(..., description="Topics to analyze for trends")
    time_windows: List[int] = Field([7, 14, 30], description="Time windows in days")
    platforms: Optional[List[str]] = Field(None, description="Platform filters")

class NaturalQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")

class MultiDimensionalResearchRequest(BaseModel):
    """Advanced multi-dimensional research combining country, semantics, and time"""
    research_question: str = Field(..., description="Research question or hypothesis")
    countries: Optional[List[str]] = Field(None, description="Countries to analyze")
    semantic_topics: Optional[List[str]] = Field(None, description="Semantic topics to focus on")
    time_periods: Optional[List[str]] = Field(None, description="Time periods [start, end] in ISO format")
    platforms: Optional[List[str]] = Field(None, description="Platforms to include")
    languages: Optional[List[str]] = Field(None, description="Languages to analyze")
    sentiment_focus: Optional[str] = Field(None, description="Sentiment focus: positive, negative, neutral, mixed")
    engagement_threshold: Optional[float] = Field(None, description="Minimum engagement rate")
    cross_dimensional_analysis: bool = Field(True, description="Enable cross-dimensional correlation analysis")
    include_temporal_trends: bool = Field(True, description="Include temporal trend analysis")
    limit: int = Field(200, ge=1, le=1000, description="Maximum results per dimension")

# =================== API ENDPOINTS ===================

@app.get("/")
async def root():
    return {
        "service": "Sephora Social Intelligence API",
        "version": "1.0.0",
        "data_scale": "1.2M+ records",
        "platforms": ["TikTok", "Facebook", "Instagram", "Customer Care"],
        "endpoints": {
            "/search": "General semantic search with multilingual support",
            "/content-discovery": "Advanced content discovery",
            "/competitive-intel": "Competitive intelligence",
            "/crisis-detection": "Crisis detection",
            "/trend-analysis": "Trend analysis",
            "/natural-query": "Natural language queries",
            "/language-analysis": "Language distribution analysis",
            "/multi-dimensional-research": "Advanced research combining country + semantics + time"
        },
        "new_features": {
            "multilingual_search": "Cross-language search capabilities",
            "result_clustering": "Automatic grouping of similar results",
            "match_explanations": "Detailed explanations for search results",
            "confidence_scoring": "Confidence scores for result relevance",
            "language_detection": "Automatic language detection for content"
        }
    }

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """General semantic search endpoint"""
    try:
        # Convert request to SearchFilters
        filters = SearchFilters(
            platforms=request.platforms,
            brands=request.brands,
            content_types=request.content_types,
            date_range=tuple(request.date_range) if request.date_range and len(request.date_range) == 2 else None,
            engagement_threshold=request.engagement_threshold,
            viral_only=request.viral_only,
            languages=request.languages,
            countries=request.countries,
            topic_ids=request.topic_ids,
            topic_labels=request.topic_labels,
            sentiment_range=tuple(request.sentiment_range) if request.sentiment_range and len(request.sentiment_range) == 2 else None,
            cross_lingual=request.cross_lingual
        )
        
        with EnterpriseSemanticSearchEngine() as search_engine:
            results = await search_engine.search(
                query=request.query,
                search_type=SearchType(request.search_type),
                scope=SearchScope(request.scope),
                filters=filters,
                limit=request.limit,
                include_analytics=request.include_analytics
            )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/customer-care")
async def search_customer_care_endpoint(request: SearchRequest):
    """Convenience endpoint for Customer Care search (forces scope='care')."""
    try:
        filters = SearchFilters(
            platforms=request.platforms,
            brands=request.brands,
            content_types=request.content_types,
            date_range=tuple(request.date_range) if request.date_range and len(request.date_range) == 2 else None,
            engagement_threshold=request.engagement_threshold,
            viral_only=request.viral_only
        )

        with EnterpriseSemanticSearchEngine() as search_engine:
            results = await search_engine.search(
                query=request.query,
                search_type=SearchType(request.search_type),
                scope=SearchScope.CUSTOMER_CARE_ONLY,
                filters=filters,
                limit=request.limit,
                include_analytics=request.include_analytics
            )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Customer care search failed: {str(e)}")

@app.post("/content-discovery")
async def content_discovery_endpoint(request: ContentDiscoveryRequest):
    """Advanced content discovery endpoint"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            results = await search_engine.content_discovery_search(
                content_themes=request.content_themes,
                performance_criteria=request.performance_criteria,
                platforms=request.platforms,
                limit=request.limit
            )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content discovery failed: {str(e)}")

@app.post("/competitive-intel")
async def competitive_intel_endpoint(request: CompetitiveIntelRequest):
    """Competitive intelligence endpoint"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            results = await search_engine.competitive_intelligence_search(
                competitor_keywords=request.competitor_keywords,
                time_window_days=request.time_window_days,
                platforms=request.platforms
            )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Competitive intelligence failed: {str(e)}")

@app.post("/crisis-detection") 
async def crisis_detection_endpoint(request: CrisisDetectionRequest):
    """Crisis detection endpoint"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            results = await search_engine.crisis_detection_search(
                sentiment_threshold=request.sentiment_threshold,
                engagement_spike_threshold=request.engagement_spike_threshold,
                time_window_hours=request.time_window_hours
            )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crisis detection failed: {str(e)}")

@app.post("/trend-analysis")
async def trend_analysis_endpoint(request: TrendAnalysisRequest):
    """Trend analysis endpoint"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            results = await search_engine.trend_analysis_search(
                emerging_topics=request.emerging_topics,
                time_windows=request.time_windows,
                platforms=request.platforms
            )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@app.post("/natural-query")
async def natural_query_endpoint(request: NaturalQueryRequest):
    """Natural language query processing"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            query_interface = SemanticQueryInterface(search_engine)
            results = await query_interface.process_natural_query(request.query)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Natural query processing failed: {str(e)}")

@app.post("/language-analysis")
async def language_analysis_endpoint(
    platforms: Optional[List[str]] = Body(None),
    limit: int = Body(1000, ge=100, le=5000)
):
    """Analyze language distribution across platforms"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            # Get sample of content for language analysis
            results = await search_engine.search(
                query="content analysis sample",
                search_type=SearchType.KEYWORD,
                scope=SearchScope.SOCIAL_ONLY,
                filters=SearchFilters(platforms=platforms),
                limit=limit,
                include_analytics=False
            )
            
            # Analyze language distribution
            language_stats = {}
            total_analyzed = 0
            
            for result in results.get("results", []):
                detected_lang = result.get("detected_language")
                lang_name = result.get("language_name", "Unknown")
                confidence = result.get("language_confidence", 0)
                
                if detected_lang:
                    if detected_lang not in language_stats:
                        language_stats[detected_lang] = {
                            "language_code": detected_lang,
                            "language_name": lang_name,
                            "count": 0,
                            "avg_confidence": 0,
                            "platforms": {}
                        }
                    
                    language_stats[detected_lang]["count"] += 1
                    language_stats[detected_lang]["avg_confidence"] += confidence
                    
                    platform = result.get("platform", "unknown")
                    if platform not in language_stats[detected_lang]["platforms"]:
                        language_stats[detected_lang]["platforms"][platform] = 0
                    language_stats[detected_lang]["platforms"][platform] += 1
                    
                    total_analyzed += 1
            
            # Calculate percentages and averages
            for lang_data in language_stats.values():
                lang_data["percentage"] = round((lang_data["count"] / max(total_analyzed, 1)) * 100, 2)
                lang_data["avg_confidence"] = round(lang_data["avg_confidence"] / max(lang_data["count"], 1), 3)
            
            # Sort by count
            sorted_languages = sorted(language_stats.values(), key=lambda x: x["count"], reverse=True)
            
            return {
                "total_content_analyzed": total_analyzed,
                "languages_detected": len(language_stats),
                "language_distribution": sorted_languages,
                "platforms_analyzed": platforms or ["all"],
                "analysis_timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Language analysis failed: {str(e)}")

# =================== ANALYTICS ENDPOINTS ===================

@app.get("/analytics/dataset-overview")
async def get_dataset_overview():
    """Get overview of all datasets in the system"""
    try:
        # Load latest metrics summaries
        metrics_dirs = {
            "tiktok": Path("./metrics/tiktok"),
            "facebook": Path("./metrics/facebook"), 
            "instagram": Path("./metrics/instagram"),
            "cross_platform": Path("./metrics/cross_platform")
        }
        
        overview = {}
        for platform, metrics_dir in metrics_dirs.items():
            summary_file = metrics_dir / f"latest_metrics_summary_{platform}.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        data = json.load(f)
                    overview[platform] = {
                        "dataset_id": data.get("dataset_id"),
                        "total_posts": data.get("quick_access", {}).get("dataset_overview", {}).get("total_posts", 0),
                        "avg_engagement_rate": data.get("quick_access", {}).get("dataset_overview", {}).get("avg_engagement_rate", 0),
                        "last_updated": data.get("last_updated")
                    }
                except Exception as e:
                    overview[platform] = {"error": f"Failed to load: {e}"}
            else:
                overview[platform] = {"status": "no_data"}
        
        return {
            "platforms": overview,
            "total_records": sum([p.get("total_posts", 0) for p in overview.values()]),
            "system_status": "operational",
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overview generation failed: {str(e)}")

@app.get("/analytics/platform/{platform}")
async def get_platform_analytics(
    platform: str,
    metric_type: str = Query("summary", description="Metric type: summary, brand, content, temporal, top, worst")
):
    """Get specific platform analytics"""
    try:
        metrics_dir = Path(f"./metrics/{platform}")
        
        if metric_type == "summary":
            summary_file = metrics_dir / f"latest_metrics_summary_{platform}.json"
        else:
            # Find latest file for this metric type
            pattern = f"{platform}_{metric_type}_*.json"
            files = list(metrics_dir.glob(pattern))
            if not files:
                raise HTTPException(status_code=404, detail=f"No {metric_type} metrics found for {platform}")
            summary_file = max(files, key=lambda f: f.stat().st_mtime)
        
        if not summary_file.exists():
            raise HTTPException(status_code=404, detail=f"Analytics file not found: {summary_file}")
        
        with open(summary_file) as f:
            data = json.load(f)
        
        return {
            "platform": platform,
            "metric_type": metric_type,
            "data": data,
            "file_path": str(summary_file),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Platform analytics failed: {str(e)}")

@app.get("/health")
async def health_check():
    """System health check"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            # Test basic connectivity
            test_result = await search_engine.search("test", limit=1)
            
        return {
            "status": "healthy",
            "weaviate_connection": "ok",
            "search_engine": "operational",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =================== ADVANCED ANALYTICS ENDPOINTS ===================

@app.post("/analytics/roi-analysis")
async def roi_analysis_endpoint(
    campaign_keywords: List[str] = Body(...),
    business_metrics: Dict[str, float] = Body(...),
    platforms: Optional[List[str]] = Body(None)
):
    """ROI analysis for specific campaigns"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            results = await search_engine.roi_analysis_search(
                campaign_keywords=campaign_keywords,
                business_metrics=business_metrics,
                platforms=platforms
            )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ROI analysis failed: {str(e)}")

@app.post("/analytics/audience-behavior")
async def audience_behavior_endpoint(
    behavior_patterns: List[str] = Body(...),
    engagement_types: List[str] = Body(["likes", "comments", "shares", "saves"]),
    platforms: Optional[List[str]] = Body(None)
):
    """Audience behavior analysis"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            results = await search_engine.audience_behavior_search(
                behavior_patterns=behavior_patterns,
                engagement_types=engagement_types,
                platforms=platforms
            )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audience behavior analysis failed: {str(e)}")

# =================== BATCH PROCESSING ENDPOINTS ===================

@app.post("/batch/multi-query")
async def batch_multi_query(
    queries: List[str] = Body(...),
    search_type: str = Body("hybrid"),
    scope: str = Body("all"),
    limit_per_query: int = Body(25)
):
    """Process multiple queries in batch"""
    try:
        results = {}
        
        with EnterpriseSemanticSearchEngine() as search_engine:
            for i, query in enumerate(queries):
                try:
                    result = await search_engine.search(
                        query=query,
                        search_type=SearchType(search_type),
                        scope=SearchScope(scope),
                        limit=limit_per_query
                    )
                    results[f"query_{i+1}"] = {
                        "query": query,
                        "result": result
                    }
                except Exception as e:
                    results[f"query_{i+1}"] = {
                        "query": query,
                        "error": str(e)
                    }
        
        return {
            "batch_results": results,
            "total_queries": len(queries),
            "successful_queries": len([r for r in results.values() if "error" not in r]),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# =================== INSIGHTS GENERATION ===================

@app.post("/insights/content-optimization")
async def content_optimization_insights(
    underperforming_threshold: float = Body(2.0, description="Engagement rate threshold for underperforming content"),
    platforms: Optional[List[str]] = Body(None)
):
    """Generate content optimization insights"""
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            # Find underperforming content
            underperforming = await search_engine.search(
                query="low engagement underperforming content",
                search_type=SearchType.KEYWORD,
                scope=SearchScope.SOCIAL_ONLY,
                filters=SearchFilters(
                    platforms=platforms,
                    engagement_threshold=0.0  # Include all for analysis
                ),
                limit=200
            )
            
            # Filter to truly underperforming
            truly_underperforming = [
                r for r in underperforming.get("results", []) 
                if r.get("engagement_rate", 0) < underperforming_threshold
            ]
            
            # Find high-performing content for comparison
            high_performing = await search_engine.search(
                query="high engagement successful content",
                search_type=SearchType.KEYWORD,
                scope=SearchScope.SOCIAL_ONLY,
                filters=SearchFilters(
                    platforms=platforms,
                    engagement_threshold=10.0  # High performers only
                ),
                limit=100
            )
            
            # Generate optimization recommendations
            optimization_insights = _generate_optimization_recommendations(
                truly_underperforming, 
                high_performing.get("results", [])
            )
        
        return {
            "analysis_criteria": {
                "underperforming_threshold": underperforming_threshold,
                "platforms": platforms or ["all"]
            },
            "underperforming_count": len(truly_underperforming),
            "high_performing_count": len(high_performing.get("results", [])),
            "optimization_insights": optimization_insights,
            "underperforming_examples": truly_underperforming[:10],
            "high_performing_examples": high_performing.get("results", [])[:10],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content optimization failed: {str(e)}")

# =================== UTILITY FUNCTIONS ===================

def _generate_optimization_recommendations(underperforming: List[Dict], high_performing: List[Dict]) -> Dict:
    """Generate content optimization recommendations"""
    if not underperforming or not high_performing:
        return {"recommendations": [], "insights": []}
    
    recommendations = []
    insights = []
    
    # Platform analysis
    under_platforms = {}
    high_platforms = {}
    
    for item in underperforming:
        platform = item.get("platform", "unknown")
        under_platforms[platform] = under_platforms.get(platform, 0) + 1
    
    for item in high_performing:
        platform = item.get("platform", "unknown") 
        high_platforms[platform] = high_platforms.get(platform, 0) + 1
    
    # Generate platform-specific recommendations
    for platform in under_platforms:
        if platform in high_platforms:
            high_avg = sum([r.get("engagement_rate", 0) for r in high_performing if r.get("platform") == platform]) / high_platforms[platform]
            under_avg = sum([r.get("engagement_rate", 0) for r in underperforming if r.get("platform") == platform]) / under_platforms[platform]
            
            if high_avg > under_avg * 2:
                recommendations.append(f"Optimize {platform} content strategy - high performers average {high_avg:.1f}% vs underperformers at {under_avg:.1f}%")
    
    # Content type analysis
    high_content_types = {}
    for item in high_performing:
        labels = item.get("labels", "")
        if "[Axis]" in labels:
            content_type = labels.split("[Axis]")[1].split(",")[0].strip()
            high_content_types[content_type] = high_content_types.get(content_type, 0) + 1
    
    if high_content_types:
        top_content_type = max(high_content_types.items(), key=lambda x: x[1])[0]
        recommendations.append(f"Increase '{top_content_type}' content production - appears in {high_content_types[top_content_type]} high-performing posts")
    
    insights.append(f"Analyzed {len(underperforming)} underperforming vs {len(high_performing)} high-performing posts")
    
    return {
        "recommendations": recommendations,
        "insights": insights,
        "platform_performance_gap": {platform: high_platforms.get(platform, 0) / max(under_platforms.get(platform, 1), 1) for platform in under_platforms}
    }


@app.post("/multi-dimensional-research")
async def multi_dimensional_research_endpoint(request: MultiDimensionalResearchRequest):
    """Advanced multi-dimensional research combining country, semantics, and time analysis"""
    try:
        # Convert sentiment focus to range
        sentiment_range = None
        if request.sentiment_focus:
            if request.sentiment_focus.lower() == "positive":
                sentiment_range = (0.2, 1.0)
            elif request.sentiment_focus.lower() == "negative":
                sentiment_range = (-1.0, -0.2)
            elif request.sentiment_focus.lower() == "neutral":
                sentiment_range = (-0.2, 0.2)
            # mixed = no filter
        
        # Build comprehensive filters
        filters = SearchFilters(
            platforms=request.platforms,
            countries=request.countries,
            languages=request.languages,
            topic_labels=request.semantic_topics,
            date_range=tuple(request.time_periods) if request.time_periods and len(request.time_periods) == 2 else None,
            sentiment_range=sentiment_range,
            engagement_threshold=request.engagement_threshold
        )
        
        with EnterpriseSemanticSearchEngine() as search_engine:
            # Get comprehensive results
            results = await search_engine.search(
                query=request.research_question,
                search_type=SearchType.HYBRID,
                scope=SearchScope.ALL_DATA,
                filters=filters,
                limit=request.limit
            )
            
            # Multi-dimensional analysis
            analysis = {
                "research_question": request.research_question,
                "total_results": len(results),
                "filters_applied": {
                    "countries": request.countries,
                    "semantic_topics": request.semantic_topics,
                    "time_periods": request.time_periods,
                    "platforms": request.platforms,
                    "languages": request.languages,
                    "sentiment_focus": request.sentiment_focus
                }
            }
            
            if results:
                # Country distribution
                if request.countries or request.cross_dimensional_analysis:
                    country_dist = {}
                    for result in results:
                        country = result.get('derived_country', 'Unknown')
                        country_dist[country] = country_dist.get(country, 0) + 1
                    analysis['country_distribution'] = dict(sorted(country_dist.items(), key=lambda x: x[1], reverse=True))
                
                # Platform performance
                platform_performance = {}
                for result in results:
                    platform = result.get('platform', 'Unknown')
                    if platform not in platform_performance:
                        platform_performance[platform] = {
                            'count': 0,
                            'avg_engagement': 0,
                            'avg_sentiment': 0
                        }
                    platform_performance[platform]['count'] += 1
                    platform_performance[platform]['avg_engagement'] += result.get('engagement_rate', 0)
                    platform_performance[platform]['avg_sentiment'] += result.get('sentiment', 0)
                
                # Calculate averages
                for platform in platform_performance:
                    count = platform_performance[platform]['count']
                    if count > 0:
                        platform_performance[platform]['avg_engagement'] = round(
                            platform_performance[platform]['avg_engagement'] / count, 2
                        )
                        platform_performance[platform]['avg_sentiment'] = round(
                            platform_performance[platform]['avg_sentiment'] / count, 3
                        )
                
                analysis['platform_performance'] = platform_performance
                
                # Cross-dimensional insights
                if request.cross_dimensional_analysis:
                    insights = []
                    
                    # Country-platform insights
                    if len(analysis.get('country_distribution', {})) > 1 and len(platform_performance) > 1:
                        top_country = max(analysis['country_distribution'], key=analysis['country_distribution'].get)
                        best_platform = max(platform_performance, key=lambda x: platform_performance[x]['avg_engagement'])
                        insights.append(f"Top performing combination: {top_country} + {best_platform}")
                    
                    # Engagement-sentiment correlation
                    high_engagement = [r for r in results if r.get('engagement_rate', 0) > 5.0]
                    if high_engagement:
                        avg_sentiment_high_eng = sum(r.get('sentiment', 0) for r in high_engagement) / len(high_engagement)
                        insights.append(f"High engagement content has {avg_sentiment_high_eng:.2f} average sentiment")
                    
                    analysis['cross_dimensional_insights'] = insights
            
            return analysis
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-dimensional research failed: {str(e)}")

# =================== STARTUP ===================

@app.on_event("startup")
async def startup_event():
    """Initialize connections and validate system"""
    print("Starting Sephora Social Intelligence API...")
    
    # Test Weaviate connection
    try:
        with EnterpriseSemanticSearchEngine() as search_engine:
            print("Weaviate connection: OK")
    except Exception as e:
        print(f"Warning: Weaviate connection failed: {e}")
    
    print("API server ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "semantic_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )