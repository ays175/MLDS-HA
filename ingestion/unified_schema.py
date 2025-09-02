#!/usr/bin/env python3
"""
Unified Schema Manager - Single source of truth for all platform schemas
Replaces all platform-specific schema files
"""
import weaviate
import weaviate.classes.config as wvcc
from typing import Dict, List, Any
import yaml
from pathlib import Path


class UnifiedSchemaManager:
    """Manages all platform schemas from configuration"""
    
    def __init__(self, platform: str):
        self.platform = platform
        self.config = self._load_schema_config(platform)
    
    def _load_schema_config(self, platform: str) -> Dict[str, Any]:
        """Load complete schema configuration"""
        config_path = Path(f"ingestion/configs/{platform}_schema.yaml")
        if not config_path.exists():
            # For backward compatibility, use minimal config
            config_path = Path(f"ingestion/configs/{platform}.yaml")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_schema(self, client: weaviate.Client):
        """Create all collections for the platform"""
        print(f"🏗️ Creating {self.platform} knowledge graph schema...")
        
        # Delete existing collections if they exist
        for collection_name in self.config['collections'].values():
            try:
                client.collections.delete(collection_name)
                print(f"  🗑️ Deleted existing collection: {collection_name}")
            except Exception:
                pass
        
        # Create collections based on configuration
        schema_config = self.config.get('schema', {})
        
        # Platform collection
        if 'platform' in self.config['collections']:
            self._create_platform_collection(client)
        
        # Brand collection
        if 'brand' in self.config['collections']:
            self._create_brand_collection(client)
        
        # Content type collection
        if 'content_type' in self.config['collections']:
            self._create_content_type_collection(client)
        
        # Platform-specific collections
        if self.platform == 'tiktok' and 'duration_range' in self.config['collections']:
            self._create_duration_range_collection(client)
        
        if self.platform == 'customer_care':
            self._create_customer_care_entity_collections(client)
        
        # Main content collection (posts/cases)
        self._create_main_collection(client)
        
        print(f"✅ {self.platform} schema created successfully!")
    
    def _create_platform_collection(self, client: weaviate.Client):
        """Create platform collection"""
        collection_name = self.config['collections']['platform']
        client.collections.create(
            name=collection_name,
            description=f"{self.platform} platform entity",
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
            properties=[
                wvcc.Property(name="name", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="type", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="description", data_type=wvcc.DataType.TEXT),
            ]
        )
    
    def _create_brand_collection(self, client: weaviate.Client):
        """Create brand collection"""
        collection_name = self.config['collections']['brand']
        client.collections.create(
            name=collection_name,
            description=f"Brands in {self.platform} content",
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
            properties=[
                wvcc.Property(name="name", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, 
                            description="brand" if self.platform != 'customer_care' else None),
                wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT),
            ]
        )
    
    def _create_content_type_collection(self, client: weaviate.Client):
        """Create content type collection"""
        collection_name = self.config['collections']['content_type']
        client.collections.create(
            name=collection_name,
            description=f"Content types for {self.platform}",
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
            properties=[
                wvcc.Property(name="name", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="type", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT),
            ]
        )
    
    def _create_duration_range_collection(self, client: weaviate.Client):
        """Create TikTok duration range collection"""
        collection_name = self.config['collections']['duration_range']
        client.collections.create(
            name=collection_name,
            description="Video duration ranges",
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
            properties=[
                wvcc.Property(name="name", data_type=wvcc.DataType.TEXT),
                wvcc.Property(name="min_seconds", data_type=wvcc.DataType.NUMBER),
                wvcc.Property(name="max_seconds", data_type=wvcc.DataType.NUMBER),
                wvcc.Property(name="description", data_type=wvcc.DataType.TEXT),
            ]
        )
    
    def _create_customer_care_entity_collections(self, client: weaviate.Client):
        """Create customer care specific entity collections"""
        # Issue Type collection
        if 'issue_type' in self.config['collections']:
            client.collections.create(
                name=self.config['collections']['issue_type'],
                description="Customer care issue types",
                vectorizer_config=wvcc.Configure.Vectorizer.none(),
                properties=[
                    wvcc.Property(name="name", data_type=wvcc.DataType.TEXT),
                    wvcc.Property(name="type", data_type=wvcc.DataType.TEXT),
                    wvcc.Property(name="description", data_type=wvcc.DataType.TEXT),
                ]
            )
        
        # Channel collection
        if 'channel' in self.config['collections']:
            client.collections.create(
                name=self.config['collections']['channel'],
                description="Communication channels",
                vectorizer_config=wvcc.Configure.Vectorizer.none(),
                properties=[
                    wvcc.Property(name="name", data_type=wvcc.DataType.TEXT),
                    wvcc.Property(name="type", data_type=wvcc.DataType.TEXT),
                    wvcc.Property(name="description", data_type=wvcc.DataType.TEXT),
                ]
            )
        
        # Priority collection
        if 'priority' in self.config['collections']:
            client.collections.create(
                name=self.config['collections']['priority'],
                description="Case priority levels",
                vectorizer_config=wvcc.Configure.Vectorizer.none(),
                properties=[
                    wvcc.Property(name="name", data_type=wvcc.DataType.TEXT),
                    wvcc.Property(name="type", data_type=wvcc.DataType.TEXT),
                    wvcc.Property(name="level", data_type=wvcc.DataType.TEXT),
                    wvcc.Property(name="description", data_type=wvcc.DataType.TEXT),
                ]
            )
    
    def _create_main_collection(self, client: weaviate.Client):
        """Create main content collection (posts/cases)"""
        # Determine collection name
        if self.platform == 'customer_care':
            collection_name = self.config['collections']['case']
            description = "Customer care cases with full details"
        else:
            collection_name = self.config['collections']['post']
            description = f"{self.platform} posts with metrics and relationships"
        
        # Build properties from schema configuration
        properties = []
        
        # Try multiple field locations in config
        schema_fields = (
            self.config.get('schema', {}).get('fields', []) or
            self.config.get('schema', {}).get('post_fields', []) or
            self.config.get('schema', {}).get('case_fields', [])
        )
        
        # If using new comprehensive schema format
        if schema_fields:
            for field in schema_fields:
                # Handle both 'weaviate_name' and 'name'
                field_name = field.get('weaviate_name', field.get('name'))
                prop_config = {
                    'name': field_name,
                    'data_type': self._get_weaviate_datatype(field['type']),
                    'description': field.get('description', '')
                }
                
                # Add vectorization for specified fields
                if field.get('vectorize', False):
                    prop_config['vectorize_property_name'] = True
                
                properties.append(wvcc.Property(**prop_config))
        else:
            # Fallback: generate properties from old config format
            properties = self._generate_properties_from_legacy_config()
        
        # Create the collection
        client.collections.create(
            name=collection_name,
            description=description,
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
            properties=properties
        )
    
    def _get_weaviate_datatype(self, type_str: str) -> wvcc.DataType:
        """Convert string type to Weaviate DataType"""
        type_mapping = {
            'text': wvcc.DataType.TEXT,
            'number': wvcc.DataType.NUMBER,
            'boolean': wvcc.DataType.BOOL,
            'date': wvcc.DataType.DATE,
            'text_array': wvcc.DataType.TEXT_ARRAY,
            'number_array': wvcc.DataType.NUMBER_ARRAY,
            'boolean_array': wvcc.DataType.BOOL_ARRAY,
        }
        return type_mapping.get(type_str, wvcc.DataType.TEXT)
    
    def _generate_properties_from_legacy_config(self) -> List[wvcc.Property]:
        """Generate properties for platforms still using old config format"""
        # This provides backward compatibility
        # In a real implementation, we'd convert all platforms to the new format
        
        if self.platform == 'facebook':
            return self._get_facebook_properties()
        elif self.platform == 'instagram':
            return self._get_instagram_properties()
        elif self.platform == 'tiktok':
            return self._get_tiktok_properties()
        elif self.platform == 'customer_care':
            return self._get_customer_care_properties()
        else:
            return []
    
    def _get_facebook_properties(self) -> List[wvcc.Property]:
        """Get Facebook post properties"""
        return [
            # Identity
            wvcc.Property(name="profile_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="facebook_profileId", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="facebook_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="facebook_url", data_type=wvcc.DataType.TEXT),
            
            # Timing & type
            wvcc.Property(name="created_time", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="content_type", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="network", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="facebook_published", data_type=wvcc.DataType.TEXT),
            
            # Content (vectorized)
            wvcc.Property(name="facebook_content", data_type=wvcc.DataType.TEXT, 
                         vectorize_property_name=True),
            wvcc.Property(name="facebook_post_labels_names", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="labels_text", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="content_summary", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            
            # Non-vectorized content
            wvcc.Property(name="facebook_post_labels", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="facebook_attachments", data_type=wvcc.DataType.TEXT),
            
            # Metrics
            wvcc.Property(name="facebook_comments", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_comments_sentiment", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="facebook_sentiment", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_interactions", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_media_type", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="facebook_reactions", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_shares", data_type=wvcc.DataType.NUMBER),
            
            # Insights
            wvcc.Property(name="facebook_insights_engagements", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_insights_impressions", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_insights_interactions", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_insights_post_clicks", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_insights_reach", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_insights_reactions", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_insights_video_views", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_insights_video_views_average_completion", data_type=wvcc.DataType.NUMBER),
            
            # Reaction breakdown
            wvcc.Property(name="facebook_reaction_anger", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_reaction_haha", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_reaction_like", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_reaction_love", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_reaction_sorry", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="facebook_reaction_wow", data_type=wvcc.DataType.NUMBER),
            
            # Calculated metrics
            wvcc.Property(name="engagement_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="view_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="like_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="share_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="comment_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="click_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="reaction_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="completion_rate", data_type=wvcc.DataType.NUMBER),
            
            # Relationships
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="brands", data_type=wvcc.DataType.TEXT_ARRAY),
            wvcc.Property(name="content_types", data_type=wvcc.DataType.TEXT_ARRAY),
            
            # Processing status tracking
            wvcc.Property(name="processing_status", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="ingested_at", data_type=wvcc.DataType.DATE),
            wvcc.Property(name="sentiment_processed_at", data_type=wvcc.DataType.DATE),
            wvcc.Property(name="semantic_processed_at", data_type=wvcc.DataType.DATE),
        ]
    
    def _get_instagram_properties(self) -> List[wvcc.Property]:
        """Get Instagram post properties"""
        # Similar structure to Facebook but with Instagram-specific fields
        return [
            # Identity
            wvcc.Property(name="profile_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="instagram_profileId", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="instagram_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="instagram_link", data_type=wvcc.DataType.TEXT),
            
            # Timing & type
            wvcc.Property(name="created_time", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="content_type", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="network", data_type=wvcc.DataType.TEXT),
            
            # Content (vectorized)
            wvcc.Property(name="instagram_content", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="instagram_post_labels_names", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="content_summary", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            
            # Metrics
            wvcc.Property(name="instagram_comments", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="instagram_sentiment", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="instagram_interactions", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="instagram_likes", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="instagram_saved", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="instagram_video_views", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="instagram_reach", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="instagram_impressions", data_type=wvcc.DataType.NUMBER),
            
            # Relationships
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="brands", data_type=wvcc.DataType.TEXT_ARRAY),
            wvcc.Property(name="content_types", data_type=wvcc.DataType.TEXT_ARRAY),
            
            # Processing status tracking
            wvcc.Property(name="processing_status", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="ingested_at", data_type=wvcc.DataType.DATE),
            wvcc.Property(name="sentiment_processed_at", data_type=wvcc.DataType.DATE),
            wvcc.Property(name="semantic_processed_at", data_type=wvcc.DataType.DATE),
        ]
    
    def _get_tiktok_properties(self) -> List[wvcc.Property]:
        """Get TikTok post properties"""
        return [
            # Identity
            wvcc.Property(name="profile_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="tiktok_profileId", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="tiktok_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="tiktok_url", data_type=wvcc.DataType.TEXT),
            
            # Timing & type
            wvcc.Property(name="created_time", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="content_type", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="network", data_type=wvcc.DataType.TEXT),
            
            # TikTok specific
            wvcc.Property(name="duration_seconds", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="video_duration_range", data_type=wvcc.DataType.TEXT),
            
            # Content (vectorized)
            wvcc.Property(name="tiktok_content", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="tiktok_post_labels_names", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="content_themes", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="content_summary", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            
            # Metrics
            wvcc.Property(name="comments", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="impressions", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="likes", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="shares", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="engagements", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="reach", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="video_views", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="completion_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="tiktok_sentiment", data_type=wvcc.DataType.NUMBER),
            
            # Calculated metrics
            wvcc.Property(name="engagement_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="view_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="like_rate", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="share_rate", data_type=wvcc.DataType.NUMBER),
            
            # Relationships
            wvcc.Property(name="platform_ref", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="brand_refs", data_type=wvcc.DataType.TEXT_ARRAY),
            wvcc.Property(name="content_type_refs", data_type=wvcc.DataType.TEXT_ARRAY),
            wvcc.Property(name="duration_range_ref", data_type=wvcc.DataType.TEXT),
            
            # Processing status tracking
            wvcc.Property(name="processing_status", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="ingested_at", data_type=wvcc.DataType.DATE),
            wvcc.Property(name="sentiment_processed_at", data_type=wvcc.DataType.DATE),
            wvcc.Property(name="semantic_processed_at", data_type=wvcc.DataType.DATE),
        ]
    
    def _get_customer_care_properties(self) -> List[wvcc.Property]:
        """Get Customer Care case properties"""
        return [
            # Identity
            wvcc.Property(name="case_number", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="customer_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="agent_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="dataset_id", data_type=wvcc.DataType.TEXT),
            
            # Case details
            wvcc.Property(name="issue_type", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="channel", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="priority", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="status", data_type=wvcc.DataType.TEXT),
            
            # Content (vectorized)
            wvcc.Property(name="subject", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="description", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="resolution", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            wvcc.Property(name="content_summary", data_type=wvcc.DataType.TEXT,
                         vectorize_property_name=True),
            
            # Timing
            wvcc.Property(name="created_date", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="resolved_date", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="response_time_hours", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="resolution_time_hours", data_type=wvcc.DataType.NUMBER),
            
            # Metrics
            wvcc.Property(name="satisfaction_score", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="is_escalated", data_type=wvcc.DataType.BOOL),
            wvcc.Property(name="interaction_count", data_type=wvcc.DataType.NUMBER),
            wvcc.Property(name="sentiment_score", data_type=wvcc.DataType.NUMBER),
            
            # Enhanced fields
            wvcc.Property(name="detected_language", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="brands_mentioned", data_type=wvcc.DataType.TEXT_ARRAY),
            wvcc.Property(name="products_mentioned", data_type=wvcc.DataType.TEXT_ARRAY),
            wvcc.Property(name="origin", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="derived_country", data_type=wvcc.DataType.TEXT),
            
            # Relationships
            wvcc.Property(name="issue_type_ref", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="channel_ref", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="priority_ref", data_type=wvcc.DataType.TEXT),
            
            # Processing status tracking
            wvcc.Property(name="processing_status", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="ingested_at", data_type=wvcc.DataType.DATE),
            wvcc.Property(name="sentiment_processed_at", data_type=wvcc.DataType.DATE),
            wvcc.Property(name="semantic_processed_at", data_type=wvcc.DataType.DATE),
        ]


def create_unified_schema(client: weaviate.Client, platform: str):
    """Create schema for specified platform"""
    manager = UnifiedSchemaManager(platform)
    manager.create_schema(client)


def _create_semantic_collections(client: weaviate.Client) -> None:
    """Create cross-platform semantic topic collections"""
    
    def _collection_exists(name: str) -> bool:
        try:
            client.collections.get(name)
            return True
        except:
            return False
    
    # 1. SemanticTopic collection
    if not _collection_exists("SemanticTopic"):
        client.collections.create(
            name="SemanticTopic",
            description="Discovered semantic topics from content clustering",
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
            properties=[
                # Core topic properties
                wvcc.Property(name="topic_id", data_type=wvcc.DataType.TEXT, description="Unique topic identifier"),
                wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Platform where topic was discovered"),
                wvcc.Property(name="label", data_type=wvcc.DataType.TEXT, description="Human-readable topic label", vectorize_property_name=True),
                wvcc.Property(name="keywords", data_type=wvcc.DataType.TEXT, description="Top keywords defining the topic", vectorize_property_name=True),
                wvcc.Property(name="description", data_type=wvcc.DataType.TEXT, description="AI-generated topic description", vectorize_property_name=True),
                
                # Topic metrics
                wvcc.Property(name="size", data_type=wvcc.DataType.INT, description="Number of posts in this topic"),
                wvcc.Property(name="avg_sentiment", data_type=wvcc.DataType.NUMBER, description="Average sentiment score"),
                wvcc.Property(name="avg_engagement_rate", data_type=wvcc.DataType.NUMBER, description="Average engagement rate"),
                wvcc.Property(name="avg_urgency", data_type=wvcc.DataType.NUMBER, description="Average urgency (customer care)"),
                
                # Temporal properties
                wvcc.Property(name="discovered_at", data_type=wvcc.DataType.DATE, description="When topic was discovered"),
                wvcc.Property(name="first_seen", data_type=wvcc.DataType.DATE, description="Earliest post in topic"),
                wvcc.Property(name="last_seen", data_type=wvcc.DataType.DATE, description="Latest post in topic"),
                wvcc.Property(name="trend", data_type=wvcc.DataType.TEXT, description="Trend direction: rising, stable, declining"),
                wvcc.Property(name="trend_confidence", data_type=wvcc.DataType.NUMBER, description="Confidence in trend assessment"),
                
                # Quality metrics
                wvcc.Property(name="coherence_score", data_type=wvcc.DataType.NUMBER, description="Topic coherence (c_v metric)"),
                wvcc.Property(name="silhouette_score", data_type=wvcc.DataType.NUMBER, description="Cluster quality metric"),
                wvcc.Property(name="risk_score", data_type=wvcc.DataType.NUMBER, description="Composite risk score"),
                
                # Content examples
                wvcc.Property(name="example_posts", data_type=wvcc.DataType.TEXT_ARRAY, description="Example post IDs in this topic"),
                wvcc.Property(name="top_unigrams", data_type=wvcc.DataType.TEXT, description="Top single words (JSON)"),
                wvcc.Property(name="top_bigrams", data_type=wvcc.DataType.TEXT, description="Top word pairs with PMI (JSON)"),
                
                # Cross-references
                wvcc.Property(name="dataset_id", data_type=wvcc.DataType.TEXT, description="Dataset/analysis run identifier"),
                wvcc.Property(name="centroid_vector", data_type=wvcc.DataType.TEXT, description="Serialized centroid vector"),
            ],
        )
        print("✅ Created SemanticTopic collection")
    
    # 2. SemanticTopicAlignment collection
    if not _collection_exists("SemanticTopicAlignment"):
        client.collections.create(
            name="SemanticTopicAlignment",
            description="Cross-platform topic alignments and relationships",
            vectorizer_config=wvcc.Configure.Vectorizer.none(),
            properties=[
                # Alignment identifiers
                wvcc.Property(name="alignment_id", data_type=wvcc.DataType.TEXT, description="Unique alignment identifier"),
                wvcc.Property(name="platform_a", data_type=wvcc.DataType.TEXT, description="First platform"),
                wvcc.Property(name="platform_b", data_type=wvcc.DataType.TEXT, description="Second platform"),
                wvcc.Property(name="topic_a_id", data_type=wvcc.DataType.TEXT, description="Topic ID from platform A"),
                wvcc.Property(name="topic_b_id", data_type=wvcc.DataType.TEXT, description="Topic ID from platform B"),
                
                # Alignment metrics
                wvcc.Property(name="similarity", data_type=wvcc.DataType.NUMBER, description="Cosine similarity between topic centroids"),
                wvcc.Property(name="label_a", data_type=wvcc.DataType.TEXT, description="Topic A label"),
                wvcc.Property(name="label_b", data_type=wvcc.DataType.TEXT, description="Topic B label"),
                wvcc.Property(name="engagement_lift", data_type=wvcc.DataType.NUMBER, description="Engagement rate A/B ratio"),
                wvcc.Property(name="sentiment_divergence", data_type=wvcc.DataType.NUMBER, description="Sentiment difference A-B"),
                
                # Discovery metadata
                wvcc.Property(name="discovered_at", data_type=wvcc.DataType.DATE, description="When alignment was discovered"),
                wvcc.Property(name="dataset_id", data_type=wvcc.DataType.TEXT, description="Analysis run identifier"),
                
                # Relationship to topics (UUIDs)
                wvcc.Property(name="topic_a_uuid", data_type=wvcc.DataType.TEXT, description="UUID of topic A in SemanticTopic"),
                wvcc.Property(name="topic_b_uuid", data_type=wvcc.DataType.TEXT, description="UUID of topic B in SemanticTopic"),
            ],
        )
        print("✅ Created SemanticTopicAlignment collection")
    
    # 3. PostTopicMapping collection (maps posts to topics)
    if not _collection_exists("PostTopicMapping"):
        client.collections.create(
            name="PostTopicMapping",
            description="Maps posts/cases to their semantic topics",
            properties=[
                wvcc.Property(name="post_id", data_type=wvcc.DataType.TEXT, description="Post/Case ID"),
                wvcc.Property(name="post_uuid", data_type=wvcc.DataType.TEXT, description="Post/Case UUID in Weaviate"),
                wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Platform"),
                wvcc.Property(name="topic_id", data_type=wvcc.DataType.TEXT, description="Semantic topic ID"),
                wvcc.Property(name="topic_uuid", data_type=wvcc.DataType.TEXT, description="Topic UUID in SemanticTopic"),
                wvcc.Property(name="topic_label", data_type=wvcc.DataType.TEXT, description="Topic label"),
                wvcc.Property(name="assignment_confidence", data_type=wvcc.DataType.NUMBER, description="Confidence of assignment"),
                wvcc.Property(name="assigned_at", data_type=wvcc.DataType.DATE, description="When assignment was made"),
            ],
        )
        print("✅ Created PostTopicMapping collection")


def create_unified_schema(client: weaviate.Client, platform: str) -> None:
    """Create unified schema for any platform"""
    config_path = Path(f"ingestion/configs/{platform}.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    schema_manager = UnifiedSchemaManager(platform)
    schema_manager.create_schema(client)
    
    # Also create cross-platform semantic collections
    _create_semantic_collections(client)


def validate_unified_schema(client: weaviate.Client, platform: str) -> bool:
    """Validate that all collections exist for platform"""
    config_path = Path(f"ingestion/configs/{platform}.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_collections = list(config['collections'].values())
    
    # Also check semantic collections
    required_collections.extend(["SemanticTopic", "SemanticTopicAlignment", "PostTopicMapping"])
    
    for collection_name in required_collections:
        try:
            client.collections.get(collection_name)
            print(f"✅ {collection_name} collection exists")
        except Exception as e:
            print(f"❌ {collection_name} collection missing: {e}")
            return False
    
    return True
