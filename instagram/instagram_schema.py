#!/usr/bin/env python3
"""
Instagram-specific Weaviate schema definition, adapted to the Instagram sample columns.

Collections:
- InstagramPlatform
- InstagramBrand
- InstagramContentType
- InstagramPost
"""

import weaviate
import weaviate.classes.config as wvcc


def create_instagram_knowledge_graph_schema(client):
    """Create Instagram-specific knowledge graph schema (drops existing Instagram collections)."""

    print("🏗️ Creating Instagram knowledge graph schema...")

    # Delete existing collections if they exist (Instagram only)
    collections_to_delete = [
        "InstagramPlatform",
        "InstagramBrand",
        "InstagramContentType",
        "InstagramPost",
    ]

    for collection_name in collections_to_delete:
        try:
            client.collections.delete(collection_name)
            print(f"🗑️ Deleted existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, that's fine
            pass

    # 1) Instagram Platform
    client.collections.create(
        name="InstagramPlatform",
        description="Social media platforms for Instagram content",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Platform name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Platform type"),
            wvcc.Property(name="description", data_type=wvcc.DataType.TEXT, description="Platform description"),
        ],
    )

    # 2) Instagram Brand
    client.collections.create(
        name="InstagramBrand",
        description="Brands featured in Instagram content",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            wvcc.Property(name="name", data_type=wvcc.DataType.TEXT, description="Brand name"),
            wvcc.Property(name="type", data_type=wvcc.DataType.TEXT, description="Brand type"),
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Associated platform"),
        ],
    )

    # 3) Instagram Content Type (Axes, Assets, etc.)
    client.collections.create(
        name="InstagramContentType",
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

    # 4) Instagram Post (main content)
    client.collections.create(
        name="InstagramPost",
        description="Instagram posts with performance metrics and relationships",
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_contextionary(
            vectorize_collection_name=True
        ),
        properties=[
            # Identity
            wvcc.Property(name="profile_id", data_type=wvcc.DataType.TEXT, description="Profile identifier"),
            wvcc.Property(name="instagram_profileId", data_type=wvcc.DataType.TEXT, description="Instagram profile ID"),
            wvcc.Property(name="instagram_id", data_type=wvcc.DataType.TEXT, description="Instagram post ID"),
            wvcc.Property(name="instagram_url", data_type=wvcc.DataType.TEXT, description="Direct link to Instagram post"),

            # Timing & type
            wvcc.Property(name="created_time", data_type=wvcc.DataType.TEXT, description="ISO timestamp when created"),
            wvcc.Property(name="content_type", data_type=wvcc.DataType.TEXT, description="Post content type (post, reel, story, carousel, etc.)"),
            wvcc.Property(name="network", data_type=wvcc.DataType.TEXT, description="Network name (instagram)"),

            # Content & labels (vectorized text fields)
            wvcc.Property(
                name="instagram_content",
                data_type=wvcc.DataType.TEXT,
                description="Post text/content",
                vectorize_property_name=True,
            ),
            wvcc.Property(
                name="instagram_post_labels_names",
                data_type=wvcc.DataType.TEXT,
                description="Flattened label names",
                vectorize_property_name=True,
            ),
            wvcc.Property(name="instagram_post_labels", data_type=wvcc.DataType.TEXT, description="Raw labels JSON"),
            wvcc.Property(name="instagram_attachments", data_type=wvcc.DataType.TEXT, description="Attachments JSON (title, url, image_url, type)"),

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
            wvcc.Property(name="instagram_comments", data_type=wvcc.DataType.NUMBER, description="Number of comments"),
            wvcc.Property(name="instagram_comments_sentiment", data_type=wvcc.DataType.TEXT, description="Comments sentiment JSON {positive,neutral,negative}"),
            wvcc.Property(name="instagram_sentiment", data_type=wvcc.DataType.NUMBER, description="Aggregate sentiment score if available"),

            # Interaction breakdown
            wvcc.Property(name="instagram_interactions", data_type=wvcc.DataType.NUMBER, description="Total interactions"),
            wvcc.Property(name="instagram_media_type", data_type=wvcc.DataType.TEXT, description="Media type (photo, video, reel, story, carousel)"),
            wvcc.Property(name="instagram_likes", data_type=wvcc.DataType.NUMBER, description="Like count"),
            wvcc.Property(name="instagram_insights_saves", data_type=wvcc.DataType.NUMBER, description="Saves (from insights)"),
            wvcc.Property(name="instagram_shares", data_type=wvcc.DataType.NUMBER, description="Share count (if available)"),
            wvcc.Property(name="instagram_insights_post_clicks", data_type=wvcc.DataType.NUMBER, description="Post clicks (if available)"),
            wvcc.Property(name="instagram_reactions", data_type=wvcc.DataType.NUMBER, description="Total reactions (if available)"),

            # Insights metrics (from sample columns)
            wvcc.Property(name="instagram_insights_engagement", data_type=wvcc.DataType.NUMBER, description="Insights engagement"),
            wvcc.Property(name="instagram_insights_impressions", data_type=wvcc.DataType.NUMBER, description="Insights impressions"),
            wvcc.Property(name="instagram_insights_reach", data_type=wvcc.DataType.NUMBER, description="Insights reach"),
            wvcc.Property(name="instagram_insights_video_views", data_type=wvcc.DataType.NUMBER, description="Insights video views"),
            wvcc.Property(name="instagram_insights_story_completion_rate", data_type=wvcc.DataType.NUMBER, description="Story completion rate (%)"),

            # Calculated Metrics (percentages derived from impressions/reach etc.)
            wvcc.Property(name="engagement_rate", data_type=wvcc.DataType.NUMBER, description="Engagement rate (%)"),
            wvcc.Property(name="view_rate", data_type=wvcc.DataType.NUMBER, description="View rate from impressions (%)"),
            wvcc.Property(name="like_rate", data_type=wvcc.DataType.NUMBER, description="Like rate from impressions (%)"),
            wvcc.Property(name="comment_rate", data_type=wvcc.DataType.NUMBER, description="Comment rate from impressions (%)"),
            wvcc.Property(name="save_rate", data_type=wvcc.DataType.NUMBER, description="Save rate from impressions (%)"),
            wvcc.Property(name="share_rate", data_type=wvcc.DataType.NUMBER, description="Share rate from impressions (%)"),
            wvcc.Property(name="click_rate", data_type=wvcc.DataType.NUMBER, description="Click rate from impressions (%)"),
            wvcc.Property(name="reaction_rate", data_type=wvcc.DataType.NUMBER, description="Reaction rate from impressions (%)"),

            # Relationships (stored as UUIDs)
            wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT, description="Platform UUID"),
            wvcc.Property(name="brands", data_type=wvcc.DataType.TEXT_ARRAY, description="Brand UUIDs"),
            wvcc.Property(name="content_types", data_type=wvcc.DataType.TEXT_ARRAY, description="Content type UUIDs"),
        ],
    )

    print("✅ Instagram schema created successfully!")


def validate_instagram_schema(client) -> bool:
    """Validate that all Instagram collections exist."""

    required_collections = [
        "InstagramPlatform",
        "InstagramBrand",
        "InstagramContentType",
        "InstagramPost",
    ]

    for collection_name in required_collections:
        try:
            client.collections.get(collection_name)
            print(f"✅ {collection_name} collection exists")
        except Exception as e:
            print(f"❌ {collection_name} collection missing: {e}")
            return False

    return True


