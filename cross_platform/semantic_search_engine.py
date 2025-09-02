#!/usr/bin/env python3
"""
Enterprise Semantic Search Engine for Social Media Analytics
Integrates with existing Weaviate architecture for million-scale semantic queries
"""

import json
import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import weaviate
import weaviate.classes.query as wvq
import pandas as pd
import numpy as np

# Language detection and multilingual support
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Multilingual embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SearchScope(Enum):
    SOCIAL_ONLY = "social"
    CUSTOMER_CARE_ONLY = "care" 
    ALL_DATA = "all"
    CROSS_PLATFORM = "cross"


class SearchType(Enum):
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    ANALYTICS = "analytics"
    TOPIC = "topic"  # New search type for topic-based queries


@dataclass
class SearchFilters:
    platforms: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    engagement_threshold: Optional[float] = None
    sentiment_range: Optional[Tuple[float, float]] = None
    viral_only: bool = False
    dataset_ids: Optional[List[str]] = None
    topic_ids: Optional[List[str]] = None  # Filter by semantic topic IDs
    topic_labels: Optional[List[str]] = None  # Filter by topic labels
    languages: Optional[List[str]] = None  # Filter by detected languages (e.g., ['en', 'es', 'pl'])
    countries: Optional[List[str]] = None  # Filter by countries (e.g., ['France', 'Global', 'USA'])
    cross_lingual: bool = False  # Enable cross-language search


@dataclass
class SearchResult:
    content_id: str
    platform: str
    content_text: str
    labels: str
    engagement_rate: float
    impressions: int
    engagements: int
    created_time: str
    semantic_score: float
    url: Optional[str] = None
    brand: Optional[str] = None
    content_type: Optional[str] = None
    sentiment: Optional[float] = None
    additional_metrics: Optional[Dict] = None
    # Enhanced fields for transparency and multilingual support
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    match_explanation: Optional[Dict] = None
    confidence_score: Optional[float] = None
    result_cluster_id: Optional[str] = None


class EnterpriseSemanticSearchEngine:
    """Million-scale semantic search engine for social media analytics"""

    def __init__(self, weaviate_host: str = "http://localhost:8080"):
        self.weaviate_host = weaviate_host
        self.client = None
        self._platform_collections = {
            "tiktok": "TikTokPost",
            "facebook": "FacebookPost", 
            "instagram": "InstagramPost"
        }
        self._care_collection = "CustomerCareCase"  # If implemented
        self._analytics_collection = "SocialAnalyticsDoc"
        
        # Initialize multilingual capabilities
        self._multilingual_model = None
        self._language_map = {
            'en': 'English', 'es': 'Spanish', 'pl': 'Polish', 'ro': 'Romanian',
            'fr': 'French', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese'
        }
        
        # Initialize multilingual embeddings if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._multilingual_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except Exception as e:
                print(f"Warning: Could not load multilingual model: {e}")
                self._multilingual_model = None

    def __enter__(self):
        self.client = weaviate.connect_to_local()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    # =================== CORE SEARCH API ===================

    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        scope: SearchScope = SearchScope.ALL_DATA,
        filters: Optional[SearchFilters] = None,
        limit: int = 50,
        include_analytics: bool = True
    ) -> Dict[str, Any]:
        """Main search interface supporting multiple query types and scopes"""
        
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'with' statement.")

        start_time = datetime.now()
        results = []
        
        # Route to appropriate search method
        if search_type == SearchType.SEMANTIC:
            results = await self._semantic_search(query, scope, filters, limit)
        elif search_type == SearchType.HYBRID:
            results = await self._hybrid_search(query, scope, filters, limit)
        elif search_type == SearchType.KEYWORD:
            results = await self._keyword_search(query, scope, filters, limit)
        elif search_type == SearchType.ANALYTICS:
            results = await self._analytics_search(query, scope, filters, limit)
        elif search_type == SearchType.TOPIC:
            results = await self._topic_search(query, scope, filters, limit)

        # Enhance results with language detection and explanations
        if results:
            # Add language information
            results = self._enhance_results_with_language_info(results)
            
            # Add match explanations and confidence scores
            for result in results:
                result["match_explanation"] = self._generate_match_explanation(query, result)
                result["confidence_score"] = self._calculate_confidence_score(result, query)
            
            # Apply result clustering to reduce redundancy
            if len(results) > 5:  # Only cluster if we have enough results
                results = self._cluster_similar_results(results)
            
            # Apply language filtering if specified
            if filters and filters.languages:
                results = [r for r in results if r.get("detected_language") in filters.languages]

        # Add analytics insights if requested
        analytics_insights = []
        if include_analytics and results:
            analytics_insights = await self._generate_search_analytics(query, results, scope)

        search_time = (datetime.now() - start_time).total_seconds()

        return {
            "query": query,
            "search_type": search_type.value,
            "scope": scope.value,
            "filters": asdict(filters) if filters else None,
            "total_results": len(results),
            "clustered_results": len([r for r in results if r.get("cluster_size", 1) > 1]),
            "languages_detected": list(set(r.get("detected_language") for r in results if r.get("detected_language"))),
            "search_time_seconds": round(search_time, 3),
            "results": results,
            "analytics_insights": analytics_insights,
            "generated_at": datetime.now().isoformat()
        }

    # =================== SPECIALIZED SEARCH TYPES ===================

    async def _semantic_search(self, query: str, scope: SearchScope, filters: Optional[SearchFilters], limit: int) -> List[Dict]:
        """Pure semantic similarity search using vector embeddings with cross-lingual support"""
        results = []
        
        # Detect query language and prepare for cross-lingual search
        query_lang, query_lang_confidence = self._detect_language(query)
        cross_lingual_enabled = filters and filters.cross_lingual
        
        # If cross-lingual search is enabled and we have multilingual model, enhance the query
        enhanced_queries = [query]
        if cross_lingual_enabled and self._multilingual_model and query_lang:
            # Add translated versions or semantic variations
            enhanced_queries.extend(self._generate_cross_lingual_queries(query, query_lang))
        
        # Determine number of sources for fair limit allocation
        include_social = scope in [SearchScope.SOCIAL_ONLY, SearchScope.ALL_DATA, SearchScope.CROSS_PLATFORM]
        include_care = scope in [SearchScope.CUSTOMER_CARE_ONLY, SearchScope.ALL_DATA, SearchScope.CROSS_PLATFORM]
        sources_count = (len(self._platform_collections) if include_social else 0) + (1 if include_care else 0)
        
        # Increase per-source limits for better coverage (minimum 100 per source)
        per_source_limit = max(100, limit // max(sources_count, 1))
        
        # For cross-platform searches, ensure we get substantial results from each platform
        if scope == SearchScope.CROSS_PLATFORM and sources_count > 1:
            per_source_limit = max(200, limit // sources_count)
        
        if include_social:
            for platform, collection_name in self._platform_collections.items():
                if filters and filters.platforms and platform not in filters.platforms:
                    continue
                    
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # Build semantic query with cross-lingual support
                    if cross_lingual_enabled and len(enhanced_queries) > 1:
                        # Use the first enhanced query for primary search
                        primary_query = enhanced_queries[0]
                        query_builder = collection.query.near_text(
                            query=primary_query,
                            limit=per_source_limit
                        )
                    else:
                        query_builder = collection.query.near_text(
                            query=query,
                            limit=per_source_limit
                        )
                    
                    # Apply filters
                    where_filter = self._build_platform_where_filter(platform, filters)
                    if where_filter:
                        query_builder = query_builder.where(where_filter)
                    
                    response = query_builder.return_metadata(wvq.MetadataQuery(score=True))
                    
                    for obj in response.objects:
                        result = self._format_social_result(obj, platform)
                        if result:
                            results.append(result)
                            
                except Exception as e:
                    print(f"Error searching {platform}: {e}")
                    continue

        # Customer Care semantic search
        if include_care:
            try:
                collection = self.client.collections.get(self._care_collection)
                query_builder = collection.query.near_text(
                    query=query,
                    limit=per_source_limit
                )
                care_where = self._build_care_where_filter(filters)
                if care_where:
                    query_builder = query_builder.where(care_where)
                response = query_builder.return_metadata(wvq.MetadataQuery(score=True))
                for obj in response.objects:
                    care_result = self._format_care_result(obj)
                    if care_result:
                        results.append(care_result)
            except Exception as e:
                print(f"Error searching customer care: {e}")

        # Sort by semantic score
        results.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
        return results[:limit]

    async def _hybrid_search(self, query: str, scope: SearchScope, filters: Optional[SearchFilters], limit: int) -> List[Dict]:
        """Hybrid search combining semantic similarity with keyword matching"""
        
        # Extract keywords for hybrid query
        keywords = self._extract_keywords(query)
        
        results = []
        include_social = scope in [SearchScope.SOCIAL_ONLY, SearchScope.ALL_DATA, SearchScope.CROSS_PLATFORM]
        include_care = scope in [SearchScope.CUSTOMER_CARE_ONLY, SearchScope.ALL_DATA, SearchScope.CROSS_PLATFORM]
        sources_count = (len(self._platform_collections) if include_social else 0) + (1 if include_care else 0)
        # Increase per-source limits for better coverage (minimum 50 per source)
        per_source_limit = max(50, limit // max(sources_count, 1))

        if include_social:
            for platform, collection_name in self._platform_collections.items():
                if filters and filters.platforms and platform not in filters.platforms:
                    continue
                    
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # Hybrid query combining semantic + keyword
                    query_builder = collection.query.hybrid(
                        query=query,
                        alpha=0.7,  # Weight toward semantic (0.7) vs keyword (0.3)
                        limit=per_source_limit
                    )
                    
                    where_filter = self._build_platform_where_filter(platform, filters)
                    if where_filter:
                        query_builder = query_builder.where(where_filter)
                    
                    response = query_builder.return_metadata(wvq.MetadataQuery(score=True))
                    
                    for obj in response.objects:
                        result = self._format_social_result(obj, platform)
                        if result:
                            results.append(result)
                            
                except Exception as e:
                    print(f"Error in hybrid search for {platform}: {e}")
                    continue

        # Customer Care hybrid search
        if include_care:
            try:
                collection = self.client.collections.get(self._care_collection)
                query_builder = collection.query.hybrid(
                    query=query,
                    alpha=0.7,
                    limit=per_source_limit
                )
                care_where = self._build_care_where_filter(filters)
                if care_where:
                    query_builder = query_builder.where(care_where)
                response = query_builder.return_metadata(wvq.MetadataQuery(score=True))
                for obj in response.objects:
                    care_result = self._format_care_result(obj)
                    if care_result:
                        results.append(care_result)
            except Exception as e:
                print(f"Error in hybrid search for customer care: {e}")

        # Include analytics search if scope allows
        if scope in [SearchScope.ALL_DATA]:
            analytics_results = await self._search_analytics_docs(query, filters, limit // 4)
            results.extend(analytics_results)

        # Re-rank by combined score
        results.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
        return results[:limit]

    async def _keyword_search(self, query: str, scope: SearchScope, filters: Optional[SearchFilters], limit: int) -> List[Dict]:
        """Traditional keyword-based search with advanced filtering"""
        
        results = []
        keywords = self._extract_keywords(query)
        include_social = scope in [SearchScope.SOCIAL_ONLY, SearchScope.ALL_DATA, SearchScope.CROSS_PLATFORM]
        include_care = scope in [SearchScope.CUSTOMER_CARE_ONLY, SearchScope.ALL_DATA, SearchScope.CROSS_PLATFORM]
        sources_count = (len(self._platform_collections) if include_social else 0) + (1 if include_care else 0)
        # Increase per-source limits for better coverage (minimum 50 per source)
        per_source_limit = max(50, limit // max(sources_count, 1))
        
        if include_social:
            for platform, collection_name in self._platform_collections.items():
                if filters and filters.platforms and platform not in filters.platforms:
                    continue
                    
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # Build keyword query
                    where_conditions = []
                    
                    # Text content conditions
                    content_field = f"{platform}_content"
                    labels_field = f"{platform}_post_labels_names"
                    
                    for keyword in keywords:
                        where_conditions.append(
                            wvq.Filter.any_of([
                                wvq.Filter.by_property(content_field).contains_any([keyword]),
                                wvq.Filter.by_property(labels_field).contains_any([keyword])
                            ])
                        )
                    
                    # Apply additional filters
                    platform_filter = self._build_platform_where_filter(platform, filters)
                    if platform_filter:
                        where_conditions.append(platform_filter)
                    
                    final_filter = wvq.Filter.all_of(where_conditions) if where_conditions else None
                    
                    query_builder = collection.query.fetch_objects(limit=per_source_limit)
                    if final_filter:
                        query_builder = query_builder.where(final_filter)
                    
                    response = query_builder
                    
                    for obj in response.objects:
                        result = self._format_social_result(obj, platform)
                        if result:
                            # Calculate keyword relevance score
                            result['semantic_score'] = self._calculate_keyword_score(result, keywords)
                            results.append(result)
                            
                except Exception as e:
                    print(f"Error in keyword search for {platform}: {e}")
                    continue

        # Customer Care keyword search
        if include_care:
            try:
                collection = self.client.collections.get(self._care_collection)
                # Build care keyword conditions across multiple text fields
                where_conditions = []
                care_text_fields = [
                    "subject", "description", "comments", "content_summary", "keywords", "issue_category"
                ]
                for kw in keywords:
                    where_conditions.append(
                        wvq.Filter.any_of([
                            wvq.Filter.by_property(field).contains_any([kw]) for field in care_text_fields
                        ])
                    )
                care_filter = self._build_care_where_filter(filters)
                if care_filter:
                    where_conditions.append(care_filter)
                final_filter = wvq.Filter.all_of(where_conditions) if where_conditions else None

                query_builder = collection.query.fetch_objects(limit=per_source_limit)
                if final_filter:
                    query_builder = query_builder.where(final_filter)
                response = query_builder
                for obj in response.objects:
                    care_result = self._format_care_result(obj)
                    if care_result:
                        care_result['semantic_score'] = self._calculate_keyword_score(
                            {
                                "content_text": care_result.get("content_text", ""),
                                "labels": care_result.get("labels", "")
                            },
                            keywords
                        )
                        results.append(care_result)
            except Exception as e:
                print(f"Error in keyword search for customer care: {e}")

        results.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
        return results[:limit]

    async def _analytics_search(self, query: str, scope: SearchScope, filters: Optional[SearchFilters], limit: int) -> List[Dict]:
        """Search through analytics insights and reports"""
        
        try:
            collection = self.client.collections.get(self._analytics_collection)
            
            # Semantic search through analytics documents
            query_builder = collection.query.near_text(
                query=query,
                limit=limit
            )
            
            # Filter by platforms if specified
            where_conditions = []
            if filters and filters.platforms:
                where_conditions.append(
                    wvq.Filter.by_property("platform").contains_any(filters.platforms)
                )
            
            if filters and filters.dataset_ids:
                where_conditions.append(
                    wvq.Filter.by_property("dataset_id").contains_any(filters.dataset_ids)
                )
            
            if where_conditions:
                query_builder = query_builder.where(wvq.Filter.all_of(where_conditions))
            
            response = query_builder.return_metadata(wvq.MetadataQuery(score=True))
            
            analytics_results = []
            for obj in response.objects:
                analytics_results.append({
                    "result_type": "analytics",
                    "platform": obj.properties.get("platform"),
                    "doc_type": obj.properties.get("doc_type"),
                    "title": obj.properties.get("title"),
                    "text": obj.properties.get("text", "")[:500],  # Truncate for display
                    "file_path": obj.properties.get("file_path"),
                    "semantic_score": obj.metadata.score if obj.metadata else 0,
                    "created_at": obj.properties.get("created_at")
                })
            
            return analytics_results
            
        except Exception as e:
            print(f"Error in analytics search: {e}")
            return []
    
    async def _topic_search(self, query: str, scope: SearchScope, filters: Optional[SearchFilters], limit: int) -> List[Dict]:
        """Search for posts by semantic topic"""
        
        results = []
        
        # First, find relevant topics
        try:
            topic_collection = self.client.collections.get("SemanticTopic")
            
            # Search topics by query
            topic_query = topic_collection.query.near_text(
                query=query,
                limit=10  # Get top 10 matching topics
            )
            
            # Apply platform filter if specified
            if filters and filters.platforms:
                topic_query = topic_query.where(
                    wvq.Filter.by_property("platform").contains_any(filters.platforms)
                )
            
            topic_response = topic_query.return_metadata(wvq.MetadataQuery(score=True))
            
            # Get topic IDs from results
            relevant_topic_ids = []
            topic_info = {}
            
            for obj in topic_response.objects:
                topic_id = obj.properties.get("topic_id")
                if topic_id:
                    relevant_topic_ids.append(topic_id)
                    topic_info[topic_id] = {
                        "label": obj.properties.get("label"),
                        "avg_sentiment": obj.properties.get("avg_sentiment"),
                        "avg_engagement_rate": obj.properties.get("avg_engagement_rate"),
                        "trend": obj.properties.get("trend"),
                        "risk_score": obj.properties.get("risk_score"),
                        "size": obj.properties.get("size"),
                        "semantic_score": obj.metadata.score if obj.metadata else 0
                    }
            
            # Now find posts in these topics via PostTopicMapping
            if relevant_topic_ids:
                mapping_collection = self.client.collections.get("PostTopicMapping")
                
                # Get post IDs for these topics
                mapping_query = mapping_collection.query.fetch_objects(
                    limit=limit
                ).where(
                    wvq.Filter.by_property("topic_id").contains_any(relevant_topic_ids)
                )
                
                mapping_response = mapping_query
                
                # Group posts by platform and topic
                posts_by_platform = {}
                for obj in mapping_response.objects:
                    platform = obj.properties.get("platform")
                    post_id = obj.properties.get("post_id")
                    topic_id = obj.properties.get("topic_id")
                    
                    if platform not in posts_by_platform:
                        posts_by_platform[platform] = []
                    posts_by_platform[platform].append({
                        "post_id": post_id,
                        "topic_id": topic_id,
                        "topic_info": topic_info.get(topic_id, {})
                    })
                
                # Fetch actual posts from each platform
                for platform, post_mappings in posts_by_platform.items():
                    if platform in self._platform_collections:
                        collection_name = self._platform_collections[platform]
                        collection = self.client.collections.get(collection_name)
                        
                        # Get unique post IDs
                        post_ids = list(set(m["post_id"] for m in post_mappings))
                        
                        # Create mapping lookup
                        topic_by_post = {m["post_id"]: m["topic_info"] for m in post_mappings}
                        
                        # Fetch posts
                        id_field = f"{platform}_id" if platform != "tiktok" else "post_id"
                        
                        post_query = collection.query.fetch_objects(
                            limit=limit // len(posts_by_platform)  # Distribute limit
                        ).where(
                            wvq.Filter.by_property(id_field).contains_any(post_ids[:50])  # Limit to 50 IDs
                        )
                        
                        post_response = post_query
                        
                        for obj in post_response.objects:
                            result = self._format_social_result(obj, platform)
                            if result:
                                post_id = result.get("content_id")
                                topic_data = topic_by_post.get(post_id, {})
                                
                                # Add topic information to result
                                result["topic_label"] = topic_data.get("label", "")
                                result["topic_trend"] = topic_data.get("trend", "")
                                result["topic_risk_score"] = topic_data.get("risk_score", 0)
                                result["topic_match_score"] = topic_data.get("semantic_score", 0)
                                
                                results.append(result)
            
            # Sort by topic match score
            results.sort(key=lambda x: x.get("topic_match_score", 0), reverse=True)
            
        except Exception as e:
            print(f"Error in topic search: {e}")
        
        return results[:limit]

    # =================== ADVANCED SEARCH METHODS ===================

    async def content_discovery_search(
        self,
        content_themes: List[str],
        performance_criteria: Dict[str, float],
        platforms: Optional[List[str]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Advanced content discovery based on themes and performance criteria"""
        
        results = []
        
        for theme in content_themes:
            # Search for content matching this theme
            theme_results = await self._semantic_search(
                theme,
                SearchScope.SOCIAL_ONLY,
                SearchFilters(platforms=platforms),
                limit // len(content_themes)
            )
            
            # Filter by performance criteria
            filtered_results = []
            for result in theme_results:
                meets_criteria = True
                
                if "min_engagement_rate" in performance_criteria:
                    if result.get("engagement_rate", 0) < performance_criteria["min_engagement_rate"]:
                        meets_criteria = False
                
                if "min_impressions" in performance_criteria:
                    if result.get("impressions", 0) < performance_criteria["min_impressions"]:
                        meets_criteria = False
                
                if meets_criteria:
                    result["matched_theme"] = theme
                    filtered_results.append(result)
            
            results.extend(filtered_results)
        
        # Analyze patterns across discovered content
        patterns = self._analyze_content_patterns(results)
        
        return {
            "themes_searched": content_themes,
            "performance_criteria": performance_criteria,
            "total_discovered": len(results),
            "results": results,
            "content_patterns": patterns,
            "generated_at": datetime.now().isoformat()
        }

    async def competitive_intelligence_search(
        self,
        competitor_keywords: List[str],
        time_window_days: int = 30,
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search for competitive mentions and analyze market positioning"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)
        
        competitive_results = []
        
        for keyword in competitor_keywords:
            # Search across platforms for competitor mentions
            platform_results = await self._hybrid_search(
                f"competitor {keyword} brand comparison mention",
                SearchScope.SOCIAL_ONLY,
                SearchFilters(
                    platforms=platforms,
                    date_range=(start_date.isoformat(), end_date.isoformat())
                ),
                50
            )
            
            for result in platform_results:
                result["competitor_keyword"] = keyword
                competitive_results.append(result)
        
        # Analyze competitive positioning
        competitive_analysis = self._analyze_competitive_landscape(competitive_results)
        
        return {
            "competitor_keywords": competitor_keywords,
            "time_window_days": time_window_days,
            "platforms_analyzed": platforms or ["all"],
            "total_mentions": len(competitive_results),
            "results": competitive_results,
            "competitive_analysis": competitive_analysis,
            "generated_at": datetime.now().isoformat()
        }

    async def crisis_detection_search(
        self,
        sentiment_threshold: float = -0.5,
        engagement_spike_threshold: float = 200.0,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Advanced crisis detection through sentiment and engagement anomalies"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        crisis_indicators = []
        
        # Search for high-engagement negative sentiment content
        for platform, collection_name in self._platform_collections.items():
            try:
                collection = self.client.collections.get(collection_name)
                
                # Query for potential crisis content
                where_filter = wvq.Filter.all_of([
                    wvq.Filter.by_property("created_time").greater_than(start_time.isoformat()),
                    wvq.Filter.by_property(f"{platform}_sentiment").less_than(sentiment_threshold)
                ])
                
                query_builder = collection.query.fetch_objects(
                    limit=100
                ).where(where_filter)
                
                response = query_builder
                
                for obj in response.objects:
                    result = self._format_social_result(obj, platform)
                    if result and result.get("engagement_rate", 0) > engagement_spike_threshold:
                        crisis_indicators.append({
                            **result,
                            "crisis_type": "negative_sentiment_spike",
                            "severity": self._calculate_crisis_severity(result)
                        })
                        
            except Exception as e:
                print(f"Error in crisis detection for {platform}: {e}")
                continue
        
        # Analyze crisis patterns
        crisis_analysis = self._analyze_crisis_patterns(crisis_indicators)
        
        return {
            "detection_criteria": {
                "sentiment_threshold": sentiment_threshold,
                "engagement_spike_threshold": engagement_spike_threshold,
                "time_window_hours": time_window_hours
            },
            "crisis_indicators_found": len(crisis_indicators),
            "severity_breakdown": crisis_analysis.get("severity_breakdown", {}),
            "recommended_actions": crisis_analysis.get("recommended_actions", []),
            "crisis_indicators": crisis_indicators,
            "generated_at": datetime.now().isoformat()
        }

    async def trend_analysis_search(
        self,
        emerging_topics: List[str],
        time_windows: List[int] = [7, 14, 30],  # days
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze trending topics across multiple time windows"""
        
        trend_data = {}
        
        for topic in emerging_topics:
            topic_trends = {}
            
            for window_days in time_windows:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=window_days)
                
                # Search for topic mentions in this time window
                topic_results = await self._hybrid_search(
                    topic,
                    SearchScope.SOCIAL_ONLY,
                    SearchFilters(
                        platforms=platforms,
                        date_range=(start_date.isoformat(), end_date.isoformat())
                    ),
                    200
                )
                
                # Calculate trend metrics
                total_mentions = len(topic_results)
                avg_engagement = np.mean([r.get("engagement_rate", 0) for r in topic_results]) if topic_results else 0
                total_impressions = sum([r.get("impressions", 0) for r in topic_results])
                
                # Analyze trend velocity (mentions per day)
                mentions_per_day = total_mentions / window_days if window_days > 0 else 0
                
                topic_trends[f"last_{window_days}_days"] = {
                    "total_mentions": total_mentions,
                    "mentions_per_day": round(mentions_per_day, 2),
                    "avg_engagement_rate": round(avg_engagement, 2),
                    "total_impressions": total_impressions,
                    "platform_breakdown": self._calculate_platform_breakdown(topic_results)
                }
            
            trend_data[topic] = topic_trends
        
        # Calculate trend momentum and recommendations
        trend_analysis = self._analyze_trend_momentum(trend_data)
        
        return {
            "topics_analyzed": emerging_topics,
            "time_windows": time_windows,
            "platforms": platforms or ["all"],
            "trend_data": trend_data,
            "trend_analysis": trend_analysis,
            "generated_at": datetime.now().isoformat()
        }

    # =================== BUSINESS INTELLIGENCE QUERIES ===================

    async def roi_analysis_search(
        self,
        campaign_keywords: List[str],
        business_metrics: Dict[str, Any],
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze ROI of specific campaigns through semantic content analysis"""
        
        campaign_results = {}
        
        for campaign in campaign_keywords:
            # Find all content related to this campaign
            campaign_content = await self._hybrid_search(
                campaign,
                SearchScope.SOCIAL_ONLY,
                SearchFilters(platforms=platforms),
                500
            )
            
            # Calculate campaign performance metrics
            if campaign_content:
                total_impressions = sum([r.get("impressions", 0) for r in campaign_content])
                total_engagements = sum([r.get("engagements", 0) for r in campaign_content])
                avg_engagement_rate = np.mean([r.get("engagement_rate", 0) for r in campaign_content])
                
                # Estimate ROI based on business metrics
                estimated_reach = total_impressions
                estimated_conversions = total_engagements * business_metrics.get("engagement_to_conversion_rate", 0.02)
                estimated_revenue = estimated_conversions * business_metrics.get("avg_order_value", 50)
                estimated_cost = len(campaign_content) * business_metrics.get("cost_per_post", 100)
                roi = ((estimated_revenue - estimated_cost) / estimated_cost * 100) if estimated_cost > 0 else 0
                
                campaign_results[campaign] = {
                    "content_pieces": len(campaign_content),
                    "total_impressions": total_impressions,
                    "total_engagements": total_engagements,
                    "avg_engagement_rate": round(avg_engagement_rate, 2),
                    "estimated_conversions": round(estimated_conversions, 0),
                    "estimated_revenue": round(estimated_revenue, 2),
                    "estimated_cost": round(estimated_cost, 2),
                    "estimated_roi_percent": round(roi, 2),
                    "top_performing_content": sorted(campaign_content, key=lambda x: x.get("engagement_rate", 0), reverse=True)[:5]
                }
        
        return {
            "campaigns_analyzed": campaign_keywords,
            "business_metrics_used": business_metrics,
            "campaign_results": campaign_results,
            "total_campaign_content": sum([r.get("content_pieces", 0) for r in campaign_results.values()]),
            "portfolio_roi": round(np.mean([r.get("estimated_roi_percent", 0) for r in campaign_results.values()]), 2),
            "generated_at": datetime.now().isoformat()
        }

    async def audience_behavior_search(
        self,
        behavior_patterns: List[str],
        engagement_types: List[str] = ["likes", "comments", "shares", "saves"],
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze audience behavior patterns through content engagement"""
        
        behavior_analysis = {}
        
        for pattern in behavior_patterns:
            # Search for content that exhibits this behavior pattern
            pattern_content = await self._semantic_search(
                f"content that drives {pattern} audience behavior engagement",
                SearchScope.SOCIAL_ONLY,
                SearchFilters(platforms=platforms),
                200
            )
            
            if pattern_content:
                # Analyze engagement type preferences
                engagement_breakdown = {}
                for eng_type in engagement_types:
                    if f"{eng_type}_rate" in pattern_content[0]:  # Check if metric exists
                        avg_rate = np.mean([r.get(f"{eng_type}_rate", 0) for r in pattern_content])
                        engagement_breakdown[eng_type] = round(avg_rate, 2)
                
                # Identify content characteristics
                content_characteristics = self._extract_content_characteristics(pattern_content)
                
                behavior_analysis[pattern] = {
                    "sample_size": len(pattern_content),
                    "engagement_breakdown": engagement_breakdown,
                    "content_characteristics": content_characteristics,
                    "top_examples": pattern_content[:5]
                }
        
        return {
            "behavior_patterns_analyzed": behavior_patterns,
            "engagement_types": engagement_types,
            "behavior_analysis": behavior_analysis,
            "cross_pattern_insights": self._generate_cross_pattern_insights(behavior_analysis),
            "generated_at": datetime.now().isoformat()
        }

    # =================== LANGUAGE DETECTION AND MULTILINGUAL SUPPORT ===================

    def _detect_language(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Detect language of text content with confidence score"""
        if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 10:
            return None, None
        
        try:
            # Get language probabilities
            lang_probs = detect_langs(text)
            if lang_probs:
                best_lang = lang_probs[0]
                return best_lang.lang, best_lang.prob
        except LangDetectException:
            pass
        
        return None, None

    def _enhance_results_with_language_info(self, results: List[Dict]) -> List[Dict]:
        """Add language detection and confidence to search results"""
        enhanced_results = []
        
        for result in results:
            content_text = result.get("content_text", "")
            labels = result.get("labels", "")
            
            # Detect language from content and labels
            full_text = f"{content_text} {labels}".strip()
            detected_lang, lang_confidence = self._detect_language(full_text)
            
            # Add language information
            result["detected_language"] = detected_lang
            result["language_confidence"] = lang_confidence
            result["language_name"] = self._language_map.get(detected_lang, "Unknown") if detected_lang else None
            
            enhanced_results.append(result)
        
        return enhanced_results

    def _cluster_similar_results(self, results: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
        """Cluster similar results to reduce redundancy"""
        if not results or not self._multilingual_model:
            return results
        
        try:
            # Extract text for clustering
            texts = []
            for result in results:
                content = result.get("content_text", "")
                labels = result.get("labels", "")
                combined_text = f"{content} {labels}".strip()
                texts.append(combined_text)
            
            # Generate embeddings
            embeddings = self._multilingual_model.encode(texts)
            
            # Simple clustering based on cosine similarity
            clusters = []
            used_indices = set()
            
            for i, embedding in enumerate(embeddings):
                if i in used_indices:
                    continue
                
                cluster = [i]
                used_indices.add(i)
                
                # Find similar results
                for j, other_embedding in enumerate(embeddings):
                    if j in used_indices or i == j:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, other_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
                    )
                    
                    if similarity >= similarity_threshold:
                        cluster.append(j)
                        used_indices.add(j)
                
                clusters.append(cluster)
            
            # Create clustered results with representatives
            clustered_results = []
            for cluster_id, cluster_indices in enumerate(clusters):
                # Select representative (highest engagement or semantic score)
                representative_idx = max(cluster_indices, key=lambda idx: 
                    results[idx].get("engagement_rate", 0) + results[idx].get("semantic_score", 0)
                )
                
                representative = results[representative_idx].copy()
                representative["result_cluster_id"] = f"cluster_{cluster_id}"
                representative["cluster_size"] = len(cluster_indices)
                
                if len(cluster_indices) > 1:
                    # Add similar results as metadata
                    similar_results = [results[idx] for idx in cluster_indices if idx != representative_idx]
                    representative["similar_results"] = similar_results[:3]  # Limit to top 3
                
                clustered_results.append(representative)
            
            return clustered_results
            
        except Exception as e:
            print(f"Error in result clustering: {e}")
            return results

    def _generate_match_explanation(self, query: str, result: Dict) -> Dict[str, Any]:
        """Generate explanation for why this result matches the query"""
        explanation = {
            "query_terms_matched": [],
            "semantic_similarity_reason": "",
            "content_highlights": [],
            "label_highlights": [],
            "engagement_boost": False,
            "language_match": True
        }
        
        try:
            # Extract query keywords
            query_keywords = self._extract_keywords(query)
            content_text = result.get("content_text", "").lower()
            labels = result.get("labels", "").lower()
            
            # Find matched terms
            for keyword in query_keywords:
                if keyword in content_text:
                    explanation["query_terms_matched"].append(keyword)
                    explanation["content_highlights"].append(keyword)
                elif keyword in labels:
                    explanation["query_terms_matched"].append(keyword)
                    explanation["label_highlights"].append(keyword)
            
            # Semantic similarity explanation
            semantic_score = result.get("semantic_score", 0)
            if semantic_score > 0.8:
                explanation["semantic_similarity_reason"] = "Very high semantic similarity"
            elif semantic_score > 0.6:
                explanation["semantic_similarity_reason"] = "Good semantic similarity"
            elif semantic_score > 0.4:
                explanation["semantic_similarity_reason"] = "Moderate semantic similarity"
            else:
                explanation["semantic_similarity_reason"] = "Low semantic similarity"
            
            # Engagement boost
            engagement_rate = result.get("engagement_rate", 0)
            if engagement_rate > 10.0:
                explanation["engagement_boost"] = True
            
            # Language match
            detected_lang = result.get("detected_language")
            if detected_lang and detected_lang not in ['en']:
                explanation["language_match"] = False
                explanation["cross_language_match"] = True
        
        except Exception as e:
            print(f"Error generating match explanation: {e}")
        
        return explanation

    def _generate_cross_lingual_queries(self, query: str, query_lang: str) -> List[str]:
        """Generate cross-lingual query variations for better multilingual search"""
        cross_lingual_queries = []
        
        try:
            # Simple translation dictionary for common search terms
            translation_dict = {
                'en': {
                    'beauty': ['belleza', 'beauté', 'beleza', 'piękno', 'frumusețe'],
                    'makeup': ['maquillaje', 'maquillage', 'maquiagem', 'makijaż', 'machiaj'],
                    'skincare': ['cuidado de la piel', 'soin de la peau', 'cuidados com a pele', 'pielęgnacja skóry', 'îngrijirea pielii'],
                    'product': ['producto', 'produit', 'produto', 'produkt', 'produs'],
                    'brand': ['marca', 'marque', 'marca', 'marka', 'marcă'],
                    'review': ['reseña', 'avis', 'avaliação', 'recenzja', 'recenzie'],
                    'quality': ['calidad', 'qualité', 'qualidade', 'jakość', 'calitate']
                },
                'es': {
                    'belleza': ['beauty', 'beauté', 'beleza', 'piękno'],
                    'maquillaje': ['makeup', 'maquillage', 'maquiagem', 'makijaż'],
                    'producto': ['product', 'produit', 'produto', 'produkt']
                },
                'pl': {
                    'piękno': ['beauty', 'belleza', 'beauté', 'beleza'],
                    'makijaż': ['makeup', 'maquillaje', 'maquillage', 'maquiagem'],
                    'produkt': ['product', 'producto', 'produit', 'produto']
                }
            }
            
            # Generate variations based on query language
            query_words = query.lower().split()
            translations = translation_dict.get(query_lang, {})
            
            for word in query_words:
                if word in translations:
                    # Create queries with translated terms
                    for translation in translations[word][:2]:  # Limit to 2 translations
                        translated_query = query.replace(word, translation)
                        cross_lingual_queries.append(translated_query)
            
            # If no direct translations found, add semantic variations
            if not cross_lingual_queries and self._multilingual_model:
                # Use the multilingual model to find semantically similar terms
                # This is a placeholder - in practice, you'd use a more sophisticated approach
                semantic_variations = [
                    f"content about {query}",
                    f"posts related to {query}",
                    f"{query} discussion"
                ]
                cross_lingual_queries.extend(semantic_variations[:2])
        
        except Exception as e:
            print(f"Error generating cross-lingual queries: {e}")
        
        return cross_lingual_queries[:3]  # Limit to 3 additional queries

    def _calculate_confidence_score(self, result: Dict, query: str) -> float:
        """Calculate overall confidence score for search result"""
        try:
            # Component scores (0-1 scale)
            semantic_score = min(result.get("semantic_score", 0), 1.0)
            
            # Keyword match score
            query_keywords = self._extract_keywords(query)
            content_text = result.get("content_text", "").lower()
            labels = result.get("labels", "").lower()
            
            keyword_matches = sum(1 for kw in query_keywords 
                                if kw in content_text or kw in labels)
            keyword_score = min(keyword_matches / max(len(query_keywords), 1), 1.0)
            
            # Language confidence
            lang_confidence = result.get("language_confidence", 1.0) or 1.0
            
            # Engagement normalization (0-1 scale, capped at 50% engagement rate)
            engagement_score = min(result.get("engagement_rate", 0) / 50.0, 1.0)
            
            # Weighted combination
            confidence = (
                semantic_score * 0.4 +
                keyword_score * 0.3 +
                lang_confidence * 0.2 +
                engagement_score * 0.1
            )
            
            return round(confidence, 3)
            
        except Exception as e:
            print(f"Error calculating confidence score: {e}")
            return 0.5

    # =================== HELPER METHODS ===================

    def _build_platform_where_filter(self, platform: str, filters: Optional[SearchFilters]) -> Optional[wvq.Filter]:
        """Build Weaviate where filter for platform-specific searches"""
        if not filters:
            return None
        
        conditions = []
        
        # Date range filter
        if filters.date_range:
            start_date, end_date = filters.date_range
            conditions.append(
                wvq.Filter.by_property("created_time").greater_or_equal(start_date)
            )
            conditions.append(
                wvq.Filter.by_property("created_time").less_or_equal(end_date)
            )
        
        # Engagement threshold
        if filters.engagement_threshold:
            conditions.append(
                wvq.Filter.by_property("engagement_rate").greater_or_equal(filters.engagement_threshold)
            )
        
        # Viral content only
        if filters.viral_only:
            # Define viral threshold as top 5% of content
            viral_threshold = 50.0  # This should be calculated from your data
            conditions.append(
                wvq.Filter.by_property("engagement_rate").greater_or_equal(viral_threshold)
            )
        
        # Brand filters (search in labels)
        if filters.brands:
            brand_conditions = []
            for brand in filters.brands:
                brand_conditions.append(
                    wvq.Filter.by_property(f"{platform}_post_labels_names").contains_any([brand])
                )
            if brand_conditions:
                conditions.append(wvq.Filter.any_of(brand_conditions))
        
        # Country filter
        if filters.countries:
            conditions.append(
                wvq.Filter.by_property("derived_country").contains_any(filters.countries)
            )
        
        # Language filter
        if filters.languages:
            conditions.append(
                wvq.Filter.by_property("detected_language").contains_any(filters.languages)
            )
        
        # Semantic topic filters
        if filters.topic_ids:
            # This would require topic assignment data in posts
            conditions.append(
                wvq.Filter.by_property("assigned_topic_id").contains_any(filters.topic_ids)
            )
        
        return wvq.Filter.all_of(conditions) if conditions else None

    def _build_care_where_filter(self, filters: Optional[SearchFilters]) -> Optional[wvq.Filter]:
        """Build Weaviate where filter for customer care searches"""
        if not filters:
            return None
        conditions = []
        if filters.date_range:
            start_date, end_date = filters.date_range
            conditions.append(wvq.Filter.by_property("created_date").greater_or_equal(start_date))
            conditions.append(wvq.Filter.by_property("created_date").less_or_equal(end_date))
        if filters.sentiment_range:
            min_s, max_s = filters.sentiment_range
            conditions.append(wvq.Filter.by_property("sentiment_score").greater_or_equal(min_s))
            conditions.append(wvq.Filter.by_property("sentiment_score").less_or_equal(max_s))
        
        # Country filter for customer care
        if filters.countries:
            conditions.append(
                wvq.Filter.by_property("derived_country").contains_any(filters.countries)
            )
        
        # Language filter for customer care
        if filters.languages:
            conditions.append(
                wvq.Filter.by_property("detected_language").contains_any(filters.languages)
            )
        
        return wvq.Filter.all_of(conditions) if conditions else None

    def _format_social_result(self, obj, platform: str) -> Optional[Dict]:
        """Format Weaviate object into standardized search result"""
        try:
            props = obj.properties
            
            # Platform-specific field mapping
            field_map = {
                "tiktok": {
                    "content_id": "tiktok_id",
                    "content_text": "tiktok_content",
                    "labels": "tiktok_post_labels_names", 
                    "url": "tiktok_link",
                    "impressions": "tiktok_insights_impressions",
                    "engagements": "tiktok_insights_engagements",
                    "sentiment": "tiktok_sentiment"
                },
                "facebook": {
                    "content_id": "facebook_id",
                    "content_text": "facebook_content",
                    "labels": "facebook_post_labels_names",
                    "url": "facebook_url", 
                    "impressions": "facebook_insights_impressions",
                    "engagements": "facebook_insights_engagements",
                    "sentiment": "facebook_sentiment"
                },
                "instagram": {
                    "content_id": "instagram_id",
                    "content_text": "instagram_content",
                    "labels": "instagram_post_labels_names",
                    "url": "instagram_url",
                    "impressions": "instagram_insights_impressions", 
                    "engagements": "instagram_insights_engagement",
                    "sentiment": "instagram_sentiment"
                }
            }
            
            fields = field_map.get(platform, {})
            
            return {
                "content_id": props.get(fields.get("content_id", ""), ""),
                "platform": platform,
                "content_text": props.get(fields.get("content_text", ""), "")[:200],  # Truncate
                "labels": props.get(fields.get("labels", ""), ""),
                "engagement_rate": props.get("engagement_rate", 0),
                "impressions": props.get(fields.get("impressions", ""), 0),
                "engagements": props.get(fields.get("engagements", ""), 0),
                "created_time": props.get("created_time", ""),
                "semantic_score": obj.metadata.score if hasattr(obj, 'metadata') and obj.metadata else 0,
                "url": props.get(fields.get("url", "")),
                "sentiment": props.get(fields.get("sentiment", "")),
                "additional_metrics": self._extract_additional_metrics(props, platform)
            }
            
        except Exception as e:
            print(f"Error formatting result for {platform}: {e}")
            return None

    def _format_care_result(self, obj) -> Optional[Dict]:
        """Format Customer Care object into standardized search result"""
        try:
            props = obj.properties
            content_text = (props.get("content_summary") or "")[:200]
            labels_text = ", ".join(filter(None, [props.get("keywords"), props.get("issue_category")]))
            return {
                "content_id": props.get("case_id", ""),
                "platform": "customer_care",
                "content_text": content_text,
                "labels": labels_text,
                "engagement_rate": 0.0,
                "impressions": 0,
                "engagements": 0,
                "created_time": props.get("created_date", ""),
                "semantic_score": obj.metadata.score if hasattr(obj, 'metadata') and obj.metadata else 0,
                "url": None,
                "sentiment": props.get("sentiment_score", 0),
                "additional_metrics": {
                    "urgency_score": props.get("urgency_score", 0),
                    "customer_satisfaction": props.get("customer_satisfaction", 0),
                    "resolution_time_hours": props.get("resolution_time_hours", 0),
                    "is_escalated": props.get("is_escalated", False),
                    "priority": props.get("priority", ""),
                    "origin": props.get("origin", "")
                }
            }
        except Exception as e:
            print(f"Error formatting customer care result: {e}")
            return None

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from natural language query"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "find", "show", "get", "search", "look", "for", "content", "posts", "about",
            "with", "that", "have", "are", "is", "was", "were", "the", "and", "or", "but"
        }
        
        # Simple tokenization and filtering
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords

    def _calculate_keyword_score(self, result: Dict, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches"""
        text_fields = [
            result.get("content_text", ""),
            result.get("labels", "")
        ]
        
        full_text = " ".join(text_fields).lower()
        matches = sum([1 for keyword in keywords if keyword in full_text])
        
        # Weight by engagement rate
        engagement_weight = min(result.get("engagement_rate", 0) / 100, 1.0)
        
        return (matches / len(keywords)) * 0.7 + engagement_weight * 0.3

    def _extract_additional_metrics(self, props: Dict, platform: str) -> Dict:
        """Extract platform-specific additional metrics"""
        additional = {}
        
        if platform == "tiktok":
            additional.update({
                "video_views": props.get("tiktok_insights_video_views", 0),
                "completion_rate": props.get("tiktok_insights_completion_rate", 0),
                "duration": props.get("tiktok_duration", 0),
                "likes": props.get("tiktok_insights_likes", 0),
                "shares": props.get("tiktok_insights_shares", 0),
                "comments": props.get("tiktok_insights_comments", 0)
            })
        elif platform == "facebook":
            additional.update({
                "video_views": props.get("facebook_insights_video_views", 0),
                "completion_rate": props.get("facebook_insights_video_views_average_completion", 0),
                "reactions_breakdown": {
                    "like": props.get("facebook_reaction_like", 0),
                    "love": props.get("facebook_reaction_love", 0),
                    "wow": props.get("facebook_reaction_wow", 0),
                    "haha": props.get("facebook_reaction_haha", 0),
                    "angry": props.get("facebook_reaction_anger", 0),
                    "sorry": props.get("facebook_reaction_sorry", 0)
                },
                "shares": props.get("facebook_shares", 0),
                "comments": props.get("facebook_comments", 0)
            })
        elif platform == "instagram":
            additional.update({
                "likes": props.get("instagram_likes", 0),
                "comments": props.get("instagram_comments", 0),
                "saves": props.get("instagram_insights_saves", 0),
                "media_type": props.get("instagram_media_type", "")
            })
        
        return additional

    # =================== ANALYTICS AND INSIGHTS ===================

    async def _generate_search_analytics(self, query: str, results: List[Dict], scope: SearchScope) -> List[Dict]:
        """Generate analytics insights from search results"""
        if not results:
            return []
        
        insights = []
        
        # Platform performance analysis
        platform_performance = {}
        for result in results:
            platform = result.get("platform", "unknown")
            if platform not in platform_performance:
                platform_performance[platform] = {
                    "count": 0,
                    "total_engagement": 0,
                    "total_impressions": 0
                }
            
            platform_performance[platform]["count"] += 1
            platform_performance[platform]["total_engagement"] += result.get("engagements", 0)
            platform_performance[platform]["total_impressions"] += result.get("impressions", 0)
        
        # Generate insights
        best_platform = max(platform_performance.items(), 
                           key=lambda x: x[1]["total_engagement"] / max(x[1]["count"], 1))[0]
        
        insights.append({
            "type": "platform_performance",
            "insight": f"Query '{query}' shows strongest engagement on {best_platform}",
            "data": platform_performance
        })
        
        # Content type analysis
        content_types = {}
        for result in results:
            labels = result.get("labels", "")
            content_type = self._extract_content_type_from_labels(labels)
            if content_type:
                if content_type not in content_types:
                    content_types[content_type] = []
                content_types[content_type].append(result.get("engagement_rate", 0))
        
        if content_types:
            best_content_type = max(content_types.items(),
                                  key=lambda x: np.mean(x[1]))[0]
            insights.append({
                "type": "content_optimization", 
                "insight": f"'{best_content_type}' content type performs best for this query",
                "avg_engagement_rate": round(np.mean(content_types[best_content_type]), 2)
            })
        
        return insights

    def _analyze_content_patterns(self, results: List[Dict]) -> Dict:
        """Analyze patterns across discovered content"""
        if not results:
            return {}
        
        # Engagement distribution
        engagement_rates = [r.get("engagement_rate", 0) for r in results]
        
        # Platform distribution  
        platform_dist = {}
        for result in results:
            platform = result.get("platform", "unknown")
            platform_dist[platform] = platform_dist.get(platform, 0) + 1
        
        # Time pattern analysis
        time_patterns = self._analyze_posting_time_patterns(results)
        
        return {
            "engagement_distribution": {
                "mean": round(np.mean(engagement_rates), 2),
                "median": round(np.median(engagement_rates), 2),
                "top_10_percent": round(np.percentile(engagement_rates, 90), 2)
            },
            "platform_distribution": platform_dist,
            "time_patterns": time_patterns,
            "common_themes": self._extract_common_themes(results)
        }

    def _analyze_competitive_landscape(self, results: List[Dict]) -> Dict:
        """Analyze competitive positioning from search results"""
        if not results:
            return {}
        
        # Sentiment analysis of competitive mentions
        competitive_sentiment = []
        brand_mentions = {}
        
        for result in results:
            sentiment = result.get("sentiment")
            if sentiment is not None:
                competitive_sentiment.append(sentiment)
            
            # Count brand co-mentions
            competitor = result.get("competitor_keyword", "")
            if competitor:
                brand_mentions[competitor] = brand_mentions.get(competitor, 0) + 1
        
        avg_sentiment = np.mean(competitive_sentiment) if competitive_sentiment else 0
        
        return {
            "avg_competitive_sentiment": round(avg_sentiment, 2),
            "total_competitive_mentions": len(results),
            "brand_mention_frequency": brand_mentions,
            "competitive_positioning": "positive" if avg_sentiment > 0.1 else "neutral" if avg_sentiment > -0.1 else "negative"
        }

    def _analyze_crisis_patterns(self, crisis_indicators: List[Dict]) -> Dict:
        """Analyze crisis patterns and generate recommendations"""
        if not crisis_indicators:
            return {"severity_breakdown": {}, "recommended_actions": []}
        
        # Severity breakdown
        severity_counts = {}
        for indicator in crisis_indicators:
            severity = indicator.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Platform analysis
        platform_breakdown = {}
        for indicator in crisis_indicators:
            platform = indicator.get("platform", "unknown")
            platform_breakdown[platform] = platform_breakdown.get(platform, 0) + 1
        
        # Generate recommendations
        recommendations = []
        if severity_counts.get("high", 0) > 0:
            recommendations.append("URGENT: Activate crisis response team immediately")
        if severity_counts.get("medium", 0) > 2:
            recommendations.append("Increase social monitoring frequency")
        if platform_breakdown:
            worst_platform = max(platform_breakdown.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Focus crisis mitigation efforts on {worst_platform}")
        
        return {
            "severity_breakdown": severity_counts,
            "platform_breakdown": platform_breakdown,
            "recommended_actions": recommendations
        }

    def _calculate_crisis_severity(self, result: Dict) -> str:
        """Calculate crisis severity based on engagement and sentiment"""
        engagement_rate = result.get("engagement_rate", 0)
        sentiment = result.get("sentiment", 0)
        
        if engagement_rate > 500 and sentiment < -0.7:
            return "high"
        elif engagement_rate > 200 and sentiment < -0.3:
            return "medium"
        else:
            return "low"

    def _analyze_trend_momentum(self, trend_data: Dict) -> Dict:
        """Analyze trend momentum and generate predictions"""
        momentum_analysis = {}
        
        for topic, windows in trend_data.items():
            # Calculate momentum by comparing different time windows
            mentions_7d = windows.get("last_7_days", {}).get("mentions_per_day", 0)
            mentions_30d = windows.get("last_30_days", {}).get("mentions_per_day", 0)
            
            momentum = "growing" if mentions_7d > mentions_30d * 1.2 else "declining" if mentions_7d < mentions_30d * 0.8 else "stable"
            
            momentum_analysis[topic] = {
                "momentum": momentum,
                "growth_rate": round((mentions_7d / max(mentions_30d, 0.1) - 1) * 100, 1),
                "recommendation": self._generate_trend_recommendation(momentum, mentions_7d)
            }
        
        return momentum_analysis

    def _generate_trend_recommendation(self, momentum: str, mentions_per_day: float) -> str:
        """Generate actionable recommendations based on trend analysis"""
        if momentum == "growing" and mentions_per_day > 5:
            return "CAPITALIZE: Create content around this trending topic immediately"
        elif momentum == "growing":
            return "MONITOR: Trend showing growth, prepare content strategy"
        elif momentum == "declining":
            return "ARCHIVE: Trend declining, focus resources elsewhere"
        else:
            return "MAINTAIN: Stable trend, continue current approach"

    def _calculate_platform_breakdown(self, results: List[Dict]) -> Dict:
        """Calculate platform distribution of results"""
        breakdown = {}
        for result in results:
            platform = result.get("platform", "unknown")
            breakdown[platform] = breakdown.get(platform, 0) + 1
        return breakdown

    def _extract_content_type_from_labels(self, labels: str) -> Optional[str]:
        """Extract primary content type from labels string"""
        if not labels:
            return None
        
        # Look for content type indicators
        for part in labels.split(','):
            part = part.strip()
            if '[Axis]' in part:
                return part.replace('[Axis]', '').strip()
            elif '[Asset]' in part:
                return part.replace('[Asset]', '').strip()
        
        return None

    def _extract_content_characteristics(self, results: List[Dict]) -> Dict:
        """Extract common characteristics from content results"""
        characteristics = {
            "avg_engagement_rate": round(np.mean([r.get("engagement_rate", 0) for r in results]), 2),
            "common_platforms": self._calculate_platform_breakdown(results),
            "posting_time_patterns": self._analyze_posting_time_patterns(results)
        }
        
        return characteristics

    def _analyze_posting_time_patterns(self, results: List[Dict]) -> Dict:
        """Analyze when high-performing content was posted"""
        posting_times = []
        for result in results:
            created_time = result.get("created_time")
            if created_time:
                try:
                    dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                    posting_times.append({
                        "hour": dt.hour,
                        "day_of_week": dt.strftime("%A"),
                        "engagement_rate": result.get("engagement_rate", 0)
                    })
                except:
                    continue
        
        if not posting_times:
            return {}
        
        # Find optimal posting hours
        hourly_performance = {}
        for pt in posting_times:
            hour = pt["hour"]
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(pt["engagement_rate"])
        
        optimal_hours = sorted(
            [(hour, np.mean(rates)) for hour, rates in hourly_performance.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "optimal_posting_hours": [{"hour": h, "avg_engagement": round(e, 2)} for h, e in optimal_hours],
            "total_samples": len(posting_times)
        }

    def _generate_cross_pattern_insights(self, behavior_analysis: Dict) -> List[str]:
        """Generate cross-pattern insights from behavior analysis"""
        insights = []
        
        if len(behavior_analysis) < 2:
            return insights
        
        # Compare engagement patterns across behaviors
        engagement_data = {}
        for pattern, data in behavior_analysis.items():
            breakdown = data.get("engagement_breakdown", {})
            if breakdown:
                engagement_data[pattern] = breakdown
        
        # Find patterns with highest overall engagement
        if engagement_data:
            pattern_scores = {}
            for pattern, breakdown in engagement_data.items():
                avg_score = np.mean(list(breakdown.values()))
                pattern_scores[pattern] = avg_score
            
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            insights.append(f"'{best_pattern[0]}' behavior pattern shows highest average engagement across all types")
        
        return insights

    # =================== SEARCH ANALYTICS DOCS ===================

    async def _search_analytics_docs(self, query: str, filters: Optional[SearchFilters], limit: int) -> List[Dict]:
        """Search through analytics insights and reports in SocialAnalyticsDoc collection"""
        try:
            collection = self.client.collections.get(self._analytics_collection)
            
            query_builder = collection.query.near_text(
                query=query,
                limit=limit
            )
            
            # Apply filters
            where_conditions = []
            if filters:
                if filters.platforms:
                    where_conditions.append(
                        wvq.Filter.by_property("platform").contains_any(filters.platforms)
                    )
                if filters.dataset_ids:
                    where_conditions.append(
                        wvq.Filter.by_property("dataset_id").contains_any(filters.dataset_ids)
                    )
            
            if where_conditions:
                query_builder = query_builder.where(wvq.Filter.all_of(where_conditions))
            
            response = query_builder.return_metadata(wvq.MetadataQuery(score=True))
            
            analytics_results = []
            for obj in response.objects:
                analytics_results.append({
                    "result_type": "analytics_insight",
                    "platform": obj.properties.get("platform"),
                    "doc_type": obj.properties.get("doc_type"),
                    "title": obj.properties.get("title"),
                    "content_preview": obj.properties.get("text", "")[:300],
                    "file_path": obj.properties.get("file_path"),
                    "semantic_score": obj.metadata.score if obj.metadata else 0,
                    "created_at": obj.properties.get("created_at"),
                    "tags": obj.properties.get("tags", [])
                })
            
            return analytics_results
            
        except Exception as e:
            print(f"Error searching analytics docs: {e}")
            return []

    def _extract_common_themes(self, results: List[Dict]) -> List[str]:
        """Extract common themes from search results"""
        themes = []
        
        # Simple theme extraction from labels
        all_labels = " ".join([r.get("labels", "") for r in results])
        
        # Extract bracketed content types
        import re
        bracket_matches = re.findall(r'\[([^\]]+)\]', all_labels)
        
        # Count frequency and return top themes
        theme_counts = {}
        for match in bracket_matches:
            theme_counts[match] = theme_counts.get(match, 0) + 1
        
        # Return top 5 themes
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [theme[0] for theme in top_themes]


# =================== QUERY INTERFACE ===================

class SemanticQueryInterface:
    """Natural language interface for complex semantic queries"""
    
    def __init__(self, search_engine: EnterpriseSemanticSearchEngine):
        self.search_engine = search_engine
    
    async def process_natural_query(self, natural_query: str) -> Dict[str, Any]:
        """Process natural language queries and route to appropriate search methods"""
        
        # Parse query intent
        intent = self._parse_query_intent(natural_query)
        
        if intent["type"] == "content_discovery":
            return await self.search_engine.content_discovery_search(
                content_themes=intent["themes"],
                performance_criteria=intent["criteria"],
                platforms=intent.get("platforms"),
                limit=intent.get("limit", 50)
            )
        elif intent["type"] == "competitive_analysis":
            return await self.search_engine.competitive_intelligence_search(
                competitor_keywords=intent["competitors"],
                time_window_days=intent.get("time_window", 30),
                platforms=intent.get("platforms")
            )
        elif intent["type"] == "crisis_detection":
            return await self.search_engine.crisis_detection_search(
                sentiment_threshold=intent.get("sentiment_threshold", -0.5),
                engagement_spike_threshold=intent.get("engagement_threshold", 200),
                time_window_hours=intent.get("time_window", 24)
            )
        elif intent["type"] == "trend_analysis":
            return await self.search_engine.trend_analysis_search(
                emerging_topics=intent["topics"],
                time_windows=intent.get("time_windows", [7, 14, 30]),
                platforms=intent.get("platforms")
            )
        else:
            # Default to hybrid search
            return await self.search_engine.search(
                query=natural_query,
                search_type=SearchType.HYBRID,
                scope=SearchScope.ALL_DATA,
                limit=intent.get("limit", 50)
            )
    
    def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Parse natural language query to extract intent and parameters"""
        query_lower = query.lower()
        
        # Intent detection patterns
        if any(word in query_lower for word in ["crisis", "negative", "complaints", "angry", "problem"]):
            return {
                "type": "crisis_detection",
                "sentiment_threshold": -0.3,
                "engagement_threshold": 150
            }
        
        elif any(word in query_lower for word in ["competitor", "versus", "vs", "against", "competition"]):
            competitors = self._extract_competitors_from_query(query)
            return {
                "type": "competitive_analysis", 
                "competitors": competitors,
                "time_window": 30
            }
        
        elif any(word in query_lower for word in ["trending", "viral", "popular", "trend", "momentum"]):
            topics = self._extract_topics_from_query(query)
            return {
                "type": "trend_analysis",
                "topics": topics,
                "time_windows": [7, 14, 30]
            }
        
        elif any(word in query_lower for word in ["discover", "find content", "content that", "posts about"]):
            themes = self._extract_themes_from_query(query)
            criteria = self._extract_performance_criteria_from_query(query)
            return {
                "type": "content_discovery",
                "themes": themes,
                "criteria": criteria
            }
        
        else:
            return {
                "type": "general_search",
                "limit": 50
            }
    
    def _extract_competitors_from_query(self, query: str) -> List[str]:
        """Extract competitor names from query"""
        # This would be customized based on your industry
        common_competitors = ["ulta", "target", "cvs", "walgreens", "amazon beauty", "macy's"]
        found_competitors = []
        
        query_lower = query.lower()
        for competitor in common_competitors:
            if competitor in query_lower:
                found_competitors.append(competitor)
        
        return found_competitors or ["competitors"]
    
    def _extract_topics_from_query(self, query: str) -> List[str]:
        """Extract topic keywords from trending queries"""
        # Extract potential topics after trend-related words
        words = query.lower().split()
        topics = []
        
        trend_words = ["trending", "viral", "popular", "trend"]
        for i, word in enumerate(words):
            if word in trend_words and i + 1 < len(words):
                # Take next 2-3 words as potential topic
                topic_words = words[i+1:i+4]
                topics.extend(topic_words)
        
        return topics or ["trending topics"]
    
    def _extract_themes_from_query(self, query: str) -> List[str]:
        """Extract content themes from discovery queries"""
        # Remove discovery-related words and extract themes
        discovery_words = ["discover", "find", "content", "posts", "about", "that", "show", "me"]
        words = [w for w in query.lower().split() if w not in discovery_words]
        
        return words[:5] if words else ["content themes"]
    
    def _extract_performance_criteria_from_query(self, query: str) -> Dict[str, float]:
        """Extract performance criteria from query"""
        criteria = {}
        
        # Look for engagement mentions
        if "high engagement" in query.lower():
            criteria["min_engagement_rate"] = 5.0
        elif "viral" in query.lower():
            criteria["min_engagement_rate"] = 10.0
        
        # Look for view count mentions
        if "million views" in query.lower():
            criteria["min_impressions"] = 1000000
        elif "thousand views" in query.lower():
            criteria["min_impressions"] = 1000
        
        return criteria


# =================== COMMAND LINE INTERFACE ===================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enterprise Semantic Search for Social Analytics')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--type', choices=['semantic', 'hybrid', 'keyword', 'analytics'], 
                       default='hybrid', help='Search type')
    parser.add_argument('--scope', choices=['social', 'care', 'all', 'cross'], 
                       default='all', help='Search scope')
    parser.add_argument('--platforms', nargs='+', choices=['tiktok', 'facebook', 'instagram'],
                       help='Limit to specific platforms')
    parser.add_argument('--brands', nargs='+', help='Filter by specific brands')
    parser.add_argument('--limit', type=int, default=50, help='Maximum results')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--natural', action='store_true', help='Use natural language query processing')
    
    args = parser.parse_args()
    
    # Build filters
    filters = SearchFilters(
        platforms=args.platforms,
        brands=args.brands
    )
    
    with EnterpriseSemanticSearchEngine() as search_engine:
        if args.natural:
            # Use natural language interface
            query_interface = SemanticQueryInterface(search_engine)
            results = await query_interface.process_natural_query(args.query)
        else:
            # Direct search
            results = await search_engine.search(
                query=args.query,
                search_type=SearchType(args.type),
                scope=SearchScope(args.scope),
                filters=filters,
                limit=args.limit
            )
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {output_path}")
        else:
            print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())