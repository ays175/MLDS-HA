# Facebook Ingestion Example

## Input CSV Row:
```csv
facebook_id,created_time,text,facebook_post_labels_names,facebook_insights_impressions,facebook_insights_engagements
FB_12345,2024-03-15T10:30:00,Check out our new Nike Air Max collection! #newrelease #nike,"Nike,Footwear,Product Launch",50000,2500
```

## Step-by-Step Processing:

### 1. Data Loading
```python
df = pd.read_csv('facebook_posts.csv')
# Row loaded as DataFrame
```

### 2. Text Sanitization
```python
# Cleans any encoding issues, removes invalid characters
text = "Check out our new Nike Air Max collection! #newrelease #nike"
```

### 3. Sentiment Analysis
```python
# HybridSentimentAnalyzer processes the text
sentiment_score = analyzer.analyze_text(text)
# Result: 0.82 (positive sentiment)
df['facebook_sentiment'] = 0.82
```

### 4. Entity Extraction
```python
# From facebook_post_labels_names = "Nike,Footwear,Product Launch"
entities = {
    'platforms': [{'name': 'Facebook', 'type': 'social', 'description': '...'}],
    'brands': [{'name': 'Nike', 'type': 'brand', 'platform': 'Facebook'}],
    'content_types': [
        {'name': 'Footwear', 'type': 'axis', 'platform': 'Facebook'},
        {'name': 'Product Launch', 'type': 'asset', 'platform': 'Facebook'}
    ]
}
```

### 5. UUID Generation (Deterministic)
```python
# Each entity gets a consistent UUID
entity_uuids = {
    'platforms': {'Facebook': 'uuid-facebook-platform'},
    'brands': {'Nike': 'uuid-nike-brand-fb'},
    'content_types': {
        'axis::Footwear': 'uuid-footwear-axis-fb',
        'asset::Product Launch': 'uuid-product-launch-asset-fb'
    }
}
```

### 6. Content Enrichment
```python
# Additional fields computed
row['word_count'] = 9
row['char_count'] = 52
row['hashtag_count'] = 2
row['mention_count'] = 0
row['contains_question'] = False
row['contains_url'] = False
row['language'] = 'en'
row['keywords'] = 'nike air max collection newrelease'
```

### 7. Weaviate Ingestion

#### 7a. Entity Ingestion
```python
# First, ingest entities (if not already present)
# Platform
client.collections.get('FacebookPlatform').data.insert({
    'name': 'Facebook',
    'type': 'social',
    'description': 'Facebook social media platform'
}, uuid='uuid-facebook-platform')

# Brand
client.collections.get('FacebookBrand').data.insert({
    'name': 'Nike',
    'type': 'brand',
    'platform': 'Facebook'
}, uuid='uuid-nike-brand-fb')

# Content Types
client.collections.get('FacebookContentType').data.insert({
    'name': 'Footwear',
    'type': 'axis',
    'platform': 'Facebook'
}, uuid='uuid-footwear-axis-fb')
```

#### 7b. Post Ingestion
```python
# Then, ingest the post with relationships
post_data = {
    'facebook_id': 'FB_12345',
    'created_time': '2024-03-15T10:30:00',
    'text': 'Check out our new Nike Air Max collection! #newrelease #nike',
    'facebook_insights_impressions': 50000,
    'facebook_insights_engagements': 2500,
    'facebook_sentiment': 0.82,
    'word_count': 9,
    'hashtag_count': 2,
    # ... other enriched fields
    
    # Relationships (as UUIDs)
    'belongsToPlatform': 'uuid-facebook-platform',
    'hasBrand': ['uuid-nike-brand-fb'],
    'hasContentType': ['uuid-footwear-axis-fb', 'uuid-product-launch-asset-fb']
}

client.collections.get('FacebookPost').data.insert(post_data)
```

### 8. Vector Generation
Weaviate automatically generates vectors for:
- The post text (for semantic search)
- Entity names (for similarity matching)

### 9. Semantic Analysis (During Ingestion)

The system automatically:

```python
# a) Samples posts from Weaviate
content_df = sample_weaviate_content(client, 'facebook', 'FacebookPost')

# b) Clusters posts by semantic similarity
labels, embeddings = cluster_content(content_df)

# c) Discovers topics
topics = [
    {
        'topic_id': 0,
        'label': 'product launches sneakers nike',
        'size': 234,
        'avg_sentiment': 0.78,
        'avg_engagement_rate': 4.5
    },
    # ... more topics
]

# d) Stores topics in Weaviate
client.collections.get('SemanticTopic').data.insert({
    'topic_id': 'facebook_0',
    'platform': 'facebook',
    'label': 'product launches sneakers nike',
    'avg_sentiment': 0.78,
    # ...
})

# e) Maps our post to its topic
client.collections.get('PostTopicMapping').data.insert({
    'post_id': 'FB_12345',
    'topic_id': 'facebook_0',
    'topic_label': 'product launches sneakers nike'
})
```

### 10. Metrics Generation (Post-Ingestion)

After all data is ingested with semantic topics:

```python
# Run unified metrics export
python metrics/unified_metrics_export.py --platform facebook

# Generates metrics including:
- Engagement rate: 2500/50000 = 5%
- Sentiment: 0.82 (positive)
- Brand performance: Nike posts avg engagement
- Content type analysis: Product Launch effectiveness
- Temporal patterns: 10:30 AM posting time analysis
- Semantic topics: 15 topics discovered
- Topic health: 'product launches' topic has excellent health
```

## Output Files Generated:

1. **Weaviate Storage**: 
   - Entities and posts stored with vectors
   - Searchable by semantic similarity

2. **Metrics Files**:
   - `metrics/facebook/facebook_unified_metrics_20240315_143022.json`
   - `metrics/facebook/facebook_hourly_metrics_20240315_143022.csv`
   - `metrics/facebook/facebook_brand_metrics_20240315_143022.csv`
   - `metrics/facebook/facebook_semantic_topics_20240315_143022.json`

3. **Semantic Collections in Weaviate**:
   - `SemanticTopic`: Discovered topics with metrics
   - `PostTopicMapping`: Links posts to their topics
   - `SemanticTopicAlignment`: Cross-platform topic relationships

## Key Benefits:

1. **Deterministic UUIDs**: Same data always gets same IDs
2. **Relationship Preservation**: Brands/content types linked to posts
3. **Enriched Data**: Sentiment, keywords, metadata added
4. **Semantic Topics**: Automatically discovered and mapped
5. **Vector Search**: Semantic similarity enabled
6. **Comprehensive Metrics**: Multi-dimensional analysis ready

## Complete Flow Summary:

```
CSV Input → Unified Ingest → Processing Pipeline → Weaviate Storage
                ↓                    ↓                    ↓
          Configuration      Sentiment Analysis    Entities + Posts
                            Content Enrichment     Semantic Topics
                            Entity Extraction      Topic Mappings
                                                  Vector Embeddings
```

Now semantic analysis is **integrated** into the ingestion process!
