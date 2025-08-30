#!/usr/bin/env python3
"""
Facebook-specific Weaviate schema definition
"""
import weaviate
import weaviate.classes.config as wvcc


def create_facebook_knowledge_graph_schema(client):
    """Create Facebook-specific knowledge graph schema"""

    print("🏗️ Creating Facebook knowledge graph schema...")

    # Delete existing collections if they exist
    collections_to_delete = [
        "FacebookPlatform",
        "FacebookBrand",
        "FacebookContentType",
        "FacebookPost",
    ]

    for collection_name in collections_to_delete:
        try:
            client.collections.delete(collection_name)
            print(f"🗑️ Deleted existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, that's fine
            pass

    # 1. Facebook Platform Collection
    client.collections.create(
        name="FacebookPlatform",
        description="Social media platforms for Facebook content",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Platform name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Platform type"),
            wvcc.Property(name="description", data_type=wvcc.DataType.TEXT, description="Platform description"),
        ],
    )

    # 2. Facebook Brand Collection
    client.collections.create(
        name="FacebookBrand",
        description="Brands featured in Facebook content",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Brand name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Brand type"),
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Associated platform"),
        ],
    )

    # 3. Facebook Content Type Collection (Axes, Assets, etc.)
    client.collections.create(
        name="FacebookContentType",
        description="Content categorization types (Axis, Asset, etc.)",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Content type name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Category type (axis, asset, etc.)"),
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Platform"),
        ],
    )

    # 4. Facebook Post Collection (main content)
    client.collections.create(
        name="FacebookPost",
        description="Facebook posts with performance metrics and relationships",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            # Identity
            wvcc.Property(name="profile_id", data_type=wvcc.DataType.TEXT, description="Profile identifier"),
            wvcc.Property(name="facebook_profileId", data_type=wvcc.DataType.TEXT, description="Facebook profile ID"),
            wvcc.Property(name="facebook_id", data_type=wvcc.DataType.TEXT, description="Facebook post ID"),
            wvcc.Property(name="facebook_url", data_type=wvcc.DataType.TEXT, description="Direct link to Facebook post"),

            # Timing & type
            wvcc.Property(name="created_time", data_type=wvcc.DataType.TEXT, description="ISO timestamp when created"),
            wvcc.Property(name="content_type", data_type=wvcc.DataType.TEXT, description="Post content type (post, reel, album, etc.)"),
            wvcc.Property(name="network", data_type=wvcc.DataType.TEXT, description="Network name (facebook)"),
            wvcc.Property(name="facebook_published", data_type=wvcc.DataType.TEXT, description="Published flag (TRUE/FALSE)"),

            # Content & labels (vectorized text fields)
            wvcc.Property(
                name="facebook_content",
                data_type=wvcc.DataType.TEXT,
                description="Post text content",
                vectorize_property_name=True,
            ),
            wvcc.Property(
                name="facebook_post_labels_names",
                data_type=wvcc.DataType.TEXT,
                description="Flattened label names",
                vectorize_property_name=True,
            ),
            wvcc.Property(name="facebook_post_labels", data_type=wvcc.DataType.TEXT, description="Raw labels JSON"),
            wvcc.Property(name="facebook_attachments", data_type=wvcc.DataType.TEXT, description="Attachments JSON (title, url, image_url, type)"),

            # Additional vector fields to aid semantic search
            wvcc.Property(
                name="labels_text",
                data_type=wvcc.DataType.TEXT,
                description="Content labels as searchable text",
                vectorize_property_name=True,
            ),
            wvcc.Property(
                name="content_summary",
                data_type=wvcc.DataType.TEXT,
                description="AI-generated content summary for semantic search",
                vectorize_property_name=True,
            ),

            # Sentiment and comments
            wvcc.Property(name="facebook_comments", data_type=wvcc.DataType.NUMBER, description="Number of comments"),
            wvcc.Property(name="facebook_comments_sentiment", data_type=wvcc.DataType.TEXT, description="Comments sentiment JSON {positive,neutral,negative}"),
            wvcc.Property(name="facebook_sentiment", data_type=wvcc.DataType.NUMBER, description="Aggregate sentiment score if available"),

            # Interaction breakdown
            wvcc.Property(name="facebook_interactions", data_type=wvcc.DataType.NUMBER, description="Total interactions"),
            wvcc.Property(name="facebook_media_type", data_type=wvcc.DataType.TEXT, description="Media type (photo, video, reel, album)"),
            wvcc.Property(name="facebook_reactions", data_type=wvcc.DataType.NUMBER, description="Total reactions"),
            wvcc.Property(name="facebook_shares", data_type=wvcc.DataType.NUMBER, description="Share count"),

            # Insights metrics
            wvcc.Property(name="facebook_insights_engagements", data_type=wvcc.DataType.NUMBER, description="Insights engagements"),
            wvcc.Property(name="facebook_insights_impressions", data_type=wvcc.DataType.NUMBER, description="Insights impressions"),
            wvcc.Property(name="facebook_insights_interactions", data_type=wvcc.DataType.NUMBER, description="Insights interactions"),
            wvcc.Property(name="facebook_insights_post_clicks", data_type=wvcc.DataType.NUMBER, description="Insights post clicks"),
            wvcc.Property(name="facebook_insights_reach", data_type=wvcc.DataType.NUMBER, description="Insights reach"),
            wvcc.Property(name="facebook_insights_reactions", data_type=wvcc.DataType.NUMBER, description="Insights reactions"),
            wvcc.Property(name="facebook_insights_video_views", data_type=wvcc.DataType.NUMBER, description="Insights video views"),
            wvcc.Property(name="facebook_insights_video_views_average_completion", data_type=wvcc.DataType.NUMBER, description="Insights video views average completion"),

            # Reaction breakdown
            wvcc.Property(name="facebook_reaction_anger", data_type=wvcc.DataType.NUMBER, description="Anger reactions"),
            wvcc.Property(name="facebook_reaction_haha", data_type=wvcc.DataType.NUMBER, description="Haha reactions"),
            wvcc.Property(name="facebook_reaction_like", data_type=wvcc.DataType.NUMBER, description="Like reactions"),
            wvcc.Property(name="facebook_reaction_love", data_type=wvcc.DataType.NUMBER, description="Love reactions"),
            wvcc.Property(name="facebook_reaction_sorry", data_type=wvcc.DataType.NUMBER, description="Sorry reactions"),
            wvcc.Property(name="facebook_reaction_wow", data_type=wvcc.DataType.NUMBER, description="Wow reactions"),

            # Calculated Metrics
            wvcc.Property(name="engagement_rate", data_type=wvcc.DataType.NUMBER, description="Engagement rate (%)"),
            wvcc.Property(name="view_rate", data_type=wvcc.DataType.NUMBER, description="View rate from impressions (%)"),
            wvcc.Property(name="like_rate", data_type=wvcc.DataType.NUMBER, description="Like rate from impressions (%)"),
            wvcc.Property(name="share_rate", data_type=wvcc.DataType.NUMBER, description="Share rate from impressions (%)"),
            wvcc.Property(name="comment_rate", data_type=wvcc.DataType.NUMBER, description="Comment rate from impressions (%)"),
            wvcc.Property(name="click_rate", data_type=wvcc.DataType.NUMBER, description="Post click rate from impressions (%)"),
            wvcc.Property(name="reaction_rate", data_type=wvcc.DataType.NUMBER, description="Reaction rate from impressions (%)"),
            wvcc.Property(name="completion_rate", data_type=wvcc.DataType.NUMBER, description="Average video completion (%)"),

            # Relationships (stored as UUIDs)
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Platform UUID"),
            wvcc.Property(name="brands", data_type=wvcc.DataType.TEXT_ARRAY, description="Brand UUIDs"),
            wvcc.Property(name="content_types", data_type=wvcc.DataType.TEXT_ARRAY, description="Content type UUIDs"),
        ],
    )

    print("✅ Facebook schema created successfully!")


def validate_facebook_schema(client):
    """Validate that all Facebook collections exist"""

    required_collections = [
        "FacebookPlatform",
        "FacebookBrand",
        "FacebookContentType",
        "FacebookPost",
    ]

    for collection_name in required_collections:
        try:
            client.collections.get(collection_name)
            print(f"✅ {collection_name} collection exists")
        except Exception as e:
            print(f"❌ {collection_name} collection missing: {e}")
            return False

    return True


