#!/usr/bin/env python3
"""
TikTok-specific Weaviate schema definition
"""
import weaviate
import weaviate.classes.config as wvcc

def create_tiktok_knowledge_graph_schema(client):
    """Create TikTok-specific knowledge graph schema"""
    
    print("🏗️ Creating TikTok knowledge graph schema...")
    
    # Delete existing collections if they exist
    collections_to_delete = ["TikTokPlatform", "TikTokBrand", "TikTokContentType", "TikTokDurationRange", "TikTokPost"]
    
    for collection_name in collections_to_delete:
        try:
            client.collections.delete(collection_name)
            print(f"🗑️ Deleted existing collection: {collection_name}")
        except Exception as e:
            # Collection doesn't exist, that's fine
            pass
    
    # 1. TikTok Platform Collection
    client.collections.create(
        name="TikTokPlatform",
        description="Social media platforms for TikTok content",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Platform name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Platform type"),
            wvcc.Property(name="description", data_type=wvcc.DataType.TEXT, description="Platform description"),
        ]
    )
    
    # 2. TikTok Brand Collection
    client.collections.create(
        name="TikTokBrand",
        description="Brands featured in TikTok content",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Brand name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Brand type"),
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Associated platform"),
        ]
    )
    
    # 3. TikTok Content Type Collection (Axes, Assets, etc.)
    client.collections.create(
        name="TikTokContentType",
        description="Content categorization types (Axis, Asset, etc.)",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Content type name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Category type (axis, asset, etc.)"),
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Platform"),
        ]
    )
    
    # 4. TikTok Duration Range Collection
    client.collections.create(
        name="TikTokDurationRange",
        description="Video duration categories",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Duration range name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Range type"),
            wvcc.Property(name="min_duration", data_type=wvcc.DataType.NUMBER, description="Minimum duration in seconds"),
            wvcc.Property(name="max_duration", data_type=wvcc.DataType.NUMBER, description="Maximum duration in seconds"),
            wvcc.Property(name="post_count", data_type=wvcc.DataType.NUMBER, description="Number of posts in this range"),
        ]
    )
    
    # 5. TikTok Post Collection (main content)
    client.collections.create(
        name="TikTokPost",
        description="TikTok posts with performance metrics and relationships",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            # Identity
            wvcc.Property(name="post_id", data_type=wvcc.DataType.TEXT, description="TikTok post ID"),
            wvcc.Property(name="profile_id", data_type=wvcc.DataType.TEXT, description="Profile identifier"),
            wvcc.Property(name="tiktok_link", data_type=wvcc.DataType.TEXT, description="Direct link to TikTok post"),
            
            # Timing
            wvcc.Property(name="posted_date", data_type=wvcc.DataType.TEXT, description="Date posted (YYYY-MM-DD)"),
            wvcc.Property(name="posted_time", data_type=wvcc.DataType.TEXT, description="Time posted (HH:MM:SS)"),
            wvcc.Property(name="weekday", data_type=wvcc.DataType.TEXT, description="Day of week"),
            wvcc.Property(name="hour", data_type=wvcc.DataType.NUMBER, description="Hour of day (0-23)"),
            
            # Content - ENHANCED FOR VECTORIZATION
            wvcc.Property(name="duration", data_type=wvcc.DataType.NUMBER, description="Video duration in seconds"),
            wvcc.Property(name="media_type", data_type=wvcc.DataType.TEXT, description="Media type from attachments"),
            wvcc.Property(name="media_count", data_type=wvcc.DataType.NUMBER, description="Number of media items"),
            
            # VECTORIZED TEXT FIELDS
            wvcc.Property(
                name="labels_text", 
                data_type=wvcc.DataType.TEXT, 
                description="Content labels as searchable text",
                vectorize_property_name=True  # Enable vectorization for this field
            ),
            wvcc.Property(
                name="brands_text", 
                data_type=wvcc.DataType.TEXT, 
                description="Extracted brand names for semantic search",
                vectorize_property_name=True  # Enable vectorization
            ),
            wvcc.Property(
                name="content_themes", 
                data_type=wvcc.DataType.TEXT, 
                description="Content themes and axes for discovery",
                vectorize_property_name=True  # Enable vectorization
            ),
            wvcc.Property(
                name="content_summary", 
                data_type=wvcc.DataType.TEXT, 
                description="AI-generated content summary for semantic search",
                vectorize_property_name=True  # Enable vectorization
            ),
            
            # Performance Metrics
            wvcc.Property(name="comments", data_type=wvcc.DataType.NUMBER, description="Number of comments"),
            wvcc.Property(name="impressions", data_type=wvcc.DataType.NUMBER, description="Total impressions"),
            wvcc.Property(name="likes", data_type=wvcc.DataType.NUMBER, description="Number of likes"),
            wvcc.Property(name="shares", data_type=wvcc.DataType.NUMBER, description="Number of shares"),
            wvcc.Property(name="engagements", data_type=wvcc.DataType.NUMBER, description="Total engagements"),
            wvcc.Property(name="reach", data_type=wvcc.DataType.NUMBER, description="Unique users reached"),
            wvcc.Property(name="video_views", data_type=wvcc.DataType.NUMBER, description="Total video views"),
            wvcc.Property(name="completion_rate", data_type=wvcc.DataType.NUMBER, description="Video completion percentage"),
            
            # Calculated Metrics
            wvcc.Property(name="engagement_rate", data_type=wvcc.DataType.NUMBER, description="Engagement rate (%)"),
            wvcc.Property(name="view_rate", data_type=wvcc.DataType.NUMBER, description="View rate from impressions (%)"),
            wvcc.Property(name="like_rate", data_type=wvcc.DataType.NUMBER, description="Like rate from views (%)"),
            wvcc.Property(name="share_rate", data_type=wvcc.DataType.NUMBER, description="Share rate from views (%)"),
            
            # Relationships (stored as UUIDs)
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Platform UUID"),
            wvcc.Property(name="brands", data_type=wvcc.DataType.TEXT_ARRAY, description="Brand UUIDs"),
            wvcc.Property(name="content_types", data_type=wvcc.DataType.TEXT_ARRAY, description="Content type UUIDs"),
            wvcc.Property(name="duration_range", data_type=wvcc.DataType.TEXT, description="Duration range UUID"),
        ]
    )
    
    print("✅ TikTok schema created successfully!")

def validate_tiktok_schema(client):
    """Validate that all TikTok collections exist"""
    
    required_collections = ["TikTokPlatform", "TikTokBrand", "TikTokContentType", "TikTokDurationRange", "TikTokPost"]
    
    for collection_name in required_collections:
        try:
            collection = client.collections.get(collection_name)
            print(f"✅ {collection_name} collection exists")
        except Exception as e:
            print(f"❌ {collection_name} collection missing: {e}")
            return False
    
    return True
