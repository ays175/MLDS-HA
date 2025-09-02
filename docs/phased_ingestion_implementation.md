# Phased Ingestion Implementation Plan

## Current vs Proposed Architecture

### Current Unified Ingestion
```python
def ingest(input_file):
    df = load_data(input_file)                    # 2-5 min
    df = process_text(df)                         # 5-10 min  
    df = compute_sentiment(df)                    # 60-120 min ❌ EXPENSIVE
    df = extract_entities(df)                     # 10-15 min
    ingest_to_weaviate(df)                       # 15-30 min
    run_semantic_analysis(df)                    # 30-60 min ❌ EXPENSIVE
    export_metrics(df)                           # 5-10 min
    # Total: 127-250 minutes (2-4 hours)
```

### Proposed Phased Approach
```python
# Phase 1: Fast Ingestion (40-65 minutes)
def fast_ingest(input_file):
    df = load_and_validate(input_file)           # 2-5 min
    df = basic_text_processing(df)               # 5-10 min
    entities = conservative_entity_extraction(df) # 10-15 min ✅ FAST
    ingest_to_weaviate(df, skip_vectorization=True) # 15-30 min ✅ FAST
    export_basic_metrics(df)                     # 5 min
    # Total: 37-65 minutes

# Phase 2: Batch Processing (scheduled)
def batch_processing():
    advanced_sentiment_analysis()                # 2-4 hours, scheduled
    enhanced_entity_extraction()                 # 1-2 hours, scheduled  
    semantic_topic_discovery()                   # 3-5 hours, scheduled
    cross_platform_resolution()                 # 1-2 hours, scheduled

# Phase 3: Analytics (daily)
def generate_analytics():
    platform_metrics()                          # 30 min
    cross_platform_analytics()                  # 30 min
    export_final_reports()                      # 15 min
```

## Implementation Steps

### Step 1: Add Processing Modes to Current System

```python
# unified_ingest.py modifications
def ingest(input_file, mode="fast"):
    # ... existing code ...
    
    if mode == "fast":
        # Skip expensive operations
        skip_sentiment = True
        skip_semantic = True
    else:
        skip_sentiment = False
        skip_semantic = False
    
    # Process with flags
    if not skip_sentiment:
        df = self._compute_sentiment(df)
    else:
        df['sentiment_score'] = None  # Placeholder
        df['processing_status'] = 'pending_sentiment'
    
    if not skip_semantic:
        semantic_results = self._run_semantic_analysis(client, df, dataset_id)
    else:
        df['processing_status'] = 'pending_semantic'
```

### Step 2: Create Batch Processor

```python
# batch_processor.py (new file)
class BatchProcessor:
    def __init__(self):
        self.client = weaviate.connect_to_local()
    
    def process_pending_sentiment(self, platform: str):
        """Process records missing sentiment analysis"""
        # Query records with processing_status = 'pending_sentiment'
        collection = self.client.collections.get(f"{platform}Post")
        
        pending_query = collection.query.fetch_objects().where(
            wvq.Filter.by_property("processing_status").equal("pending_sentiment")
        )
        
        for batch in chunk(pending_query.objects, 5000):
            texts = [obj.properties.get('text', '') for obj in batch]
            scores = self.sentiment_analyzer.analyze_batch(texts)
            
            # Update records
            for obj, score in zip(batch, scores):
                collection.data.update(
                    uuid=obj.uuid,
                    properties={
                        "sentiment_score": score,
                        "processing_status": "sentiment_complete"
                    }
                )
    
    def process_semantic_analysis(self, platform: str):
        """Run semantic analysis on platform data"""
        # Only run if enough new data
        new_records = self.count_new_records(platform)
        if new_records < 1000:
            print(f"Only {new_records} new records, skipping semantic analysis")
            return
        
        # Run semantic pipeline
        from cross_platform.semantic_pipeline import run_semantic_pipeline
        run_semantic_pipeline(platform_filter=platform)
```

### Step 3: Add Scheduling System

```python
# scheduler.py (new file)
import schedule
import time
from batch_processor import BatchProcessor

def setup_batch_jobs():
    processor = BatchProcessor()
    
    # Daily sentiment processing
    schedule.every().day.at("02:00").do(
        processor.process_pending_sentiment, "facebook"
    )
    schedule.every().day.at("03:00").do(
        processor.process_pending_sentiment, "instagram"
    )
    
    # Weekly semantic analysis
    schedule.every().sunday.at("01:00").do(
        processor.process_semantic_analysis, "all"
    )
    
    # Daily metrics generation
    schedule.every().day.at("06:00").do(
        generate_daily_metrics
    )

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    setup_batch_jobs()
    run_scheduler()
```

### Step 4: Modify Configuration Files

```yaml
# Add to facebook.yaml, instagram.yaml, etc.
processing:
  mode: fast  # or full
  
  # Fast mode settings
  fast_mode:
    skip_sentiment: true
    skip_semantic: true
    skip_vectorization: true
    conservative_entities_only: true
  
  # Batch processing settings
  batch_processing:
    sentiment_analysis:
      enabled: true
      schedule: "daily"
      batch_size: 5000
    
    semantic_analysis:
      enabled: true
      schedule: "weekly"
      min_new_records: 1000
    
    entity_enhancement:
      enabled: true
      schedule: "weekly"
```

## Benefits Analysis

### Time to Value
- **Before**: 2-4 hours until any insights available
- **After**: 40-65 minutes until basic insights available
- **Improvement**: 3-4x faster initial results

### Resource Utilization
- **Before**: High CPU/GPU usage during ingestion, idle otherwise
- **After**: Distributed processing, better resource utilization
- **Improvement**: More predictable resource usage

### Reliability
- **Before**: Single point of failure - if sentiment fails, everything fails
- **After**: Core data available even if advanced processing fails
- **Improvement**: Higher availability and reliability

### Scalability
- **Before**: Processing time grows linearly with data size
- **After**: Fast ingestion + scheduled batch processing
- **Improvement**: Can handle larger datasets without blocking

## Migration Strategy

### Week 1: Preparation
- Add processing mode flags to current system
- Test fast mode with existing data
- Validate that basic metrics work without sentiment/semantic

### Week 2: Batch Processor
- Implement batch_processor.py
- Test sentiment backfill on sample data
- Implement checkpointing and error recovery

### Week 3: Scheduling & Integration
- Implement scheduler.py
- Test end-to-end workflow
- Performance testing and optimization

### Week 4: Production Deployment
- Deploy phased system
- Monitor performance and reliability
- Fine-tune batch sizes and schedules

## Quality Gates

### Phase 1 (Fast Ingestion)
- ✅ All required fields present
- ✅ Entity extraction > 80% confidence
- ✅ Data integrity checks pass
- ✅ Basic metrics generated

### Phase 2 (Batch Processing)
- ✅ Sentiment analysis coverage > 95%
- ✅ Topic coherence score > 0.3
- ✅ Entity resolution accuracy > 90%

### Phase 3 (Analytics)
- ✅ All platforms have current metrics
- ✅ Cross-platform analysis complete
- ✅ API endpoints responding < 2s

This phased approach transforms the current "all-or-nothing" ingestion into a progressive enhancement system that delivers value quickly while building comprehensive analysis over time.

---

## 🔧 Changes Needed to Current System

### **🎯 Core Philosophy Change**

**Current**: "All-or-nothing" - everything must complete or nothing is available
**Proposed**: "Progressive enhancement" - get basic value fast, add sophistication over time

### **⚡ Key Advantages**

1. **Time to Value**: 40-65 minutes vs 2-4 hours (3-4x improvement)
2. **Reliability**: Core data available even if advanced processing fails
3. **Resource Efficiency**: Spread expensive operations across time
4. **Scalability**: Can handle larger datasets without blocking

### **🔍 Detailed Stage Analysis**

#### **Phase 1: Fast Ingestion (Critical Path)**
- **Stage 1-2**: Data loading + basic text processing (10-15 min)
  - ✅ **Good**: Essential operations, can't be optimized further
  - ⚠️ **Consider**: Parallel processing for multiple files

- **Stage 3**: Conservative entity extraction (10-15 min)
  - ✅ **Excellent**: High-confidence only, defer ambiguous cases
  - 💡 **Enhancement**: Use pre-built entity dictionaries for speed

- **Stage 4-5**: Schema + Weaviate ingestion (20-35 min)
  - ✅ **Smart**: Skip expensive vectorization initially
  - 💡 **Optimization**: Use larger batch sizes, minimal validation

#### **Phase 2: Batch Processing (Scheduled)**
- **Stage 6**: Advanced sentiment (2-4 hours)
  - ✅ **Appropriate**: Most expensive operation, good candidate for batching
  - 💡 **Optimization**: Process only records missing sentiment scores

- **Stage 7**: Enhanced entity extraction (1-2 hours)
  - ✅ **Good**: ML-based entity discovery is expensive
  - ⚠️ **Risk**: May discover entities that conflict with conservative extraction

- **Stage 8**: Semantic analysis (3-5 hours)
  - ✅ **Perfect**: Requires full dataset, benefits from batch processing
  - 💡 **Optimization**: Only run when sufficient new data (>1K records)

### **🚨 Potential Issues & Solutions**

#### **1. Data Consistency**
**Problem**: Users might query data before batch processing completes
**Solution**: 
```python
# Add processing status fields
record = {
    'text': 'content...',
    'sentiment_score': None,  # Will be filled later
    'processing_status': 'pending_sentiment',
    'ingested_at': datetime.now(),
    'sentiment_processed_at': None
}
```

#### **2. Dependency Management**
**Problem**: Semantic analysis depends on sentiment scores
**Solution**: 
```python
# Make semantic analysis sentiment-aware but not dependent
def semantic_analysis(df):
    if 'sentiment_score' in df.columns and df['sentiment_score'].notna().any():
        # Use sentiment in topic analysis
        pass
    else:
        # Run without sentiment, update later
        pass
```

#### **3. Metrics Accuracy**
**Problem**: Early metrics might be incomplete
**Solution**:
```python
# Version metrics with processing status
metrics = {
    'processing_level': 'basic',  # basic, sentiment_complete, full
    'sentiment_coverage': 0.0,    # % of records with sentiment
    'semantic_coverage': 0.0,     # % of records with topics
    'last_updated': datetime.now()
}
```

### **🔧 Implementation Recommendations**

#### **1. Modify Current System Gradually**
```python
# Add to unified_ingest.py
def ingest(self, input_file, processing_mode="fast"):
    if processing_mode == "fast":
        return self._fast_ingest(input_file)
    else:
        return self._full_ingest(input_file)  # Current behavior

def _fast_ingest(self, input_file):
    # Skip expensive operations
    df = self.data_processor.process_dataframe(df, skip_sentiment=True)
    # Skip semantic analysis
    # Generate basic metrics only
```

#### **2. Create Batch Processing System**
```python
# New file: batch_processor.py
class BatchProcessor:
    def process_pending_operations(self, platform: str):
        # Find records needing processing
        # Apply expensive operations in batches
        # Update processing status
```

#### **3. Add Quality Monitoring**
```python
# Monitor processing pipeline health
def check_processing_health():
    for platform in ['facebook', 'instagram', 'tiktok', 'customer_care']:
        pending_sentiment = count_pending_sentiment(platform)
        pending_semantic = count_pending_semantic(platform)
        
        if pending_sentiment > 10000:
            alert(f"{platform} has {pending_sentiment} records pending sentiment")
```

### **📊 Expected Performance Impact**

| Metric | Current | Proposed | Improvement |
|--------|---------|----------|-------------|
| Time to basic insights | 2-4 hours | 40-65 min | **3-4x faster** |
| System availability | 0% during ingestion | 90%+ always | **Massive improvement** |
| Resource utilization | Bursty | Smooth | **Better efficiency** |
| Failure recovery | Start over | Continue from checkpoint | **Much better** |

## 📋 Implementation Changes Required

### **🔧 Configuration Updates** (All platform YAML files)
- Add `processing_mode` settings
- Add batch processing schedules
- Add processing status tracking

### **🗄️ Schema Changes** (`unified_schema.py`)
- Add processing status fields to all collections
- Add timestamp fields for tracking processing stages

### **⚙️ Core Ingestion Changes** (`unified_ingest.py`)
- Add fast/full/batch processing modes
- Implement `_fast_process_dataframe()` method
- Skip expensive operations in fast mode

### **🆕 New Components**
- `batch_processor.py` - Handle expensive batch operations
- `scheduler.py` - Manage scheduled batch jobs
- `global_config.py` - Centralized configuration management
- `entity_resolver.py` - Handle entity conflicts

## 🚨 Hardcoded Values Found & Fixed

### **Critical Issues Identified:**
1. **Collection Names** → Made configurable via global config
2. **Processing Thresholds** → Moved to YAML configuration
3. **File Paths** → Environment variables + config
4. **Sentiment Thresholds** → Configuration-driven
5. **Model Parameters** → Model configuration section

### **Global Configuration System:**
```python
# config/global_config.py (NEW FILE)
class GlobalConfig:
    """Centralized configuration management with environment variable support"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.getenv('MLDS_CONFIG', 'config/global.yaml')
        self.config = self._load_config()
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'processing.batch_size')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
```

## 🤝 Entity Conflict Resolution Strategy

### **Problem**: 
Conservative extraction (fast, high-confidence) vs Enhanced extraction (slow, ML-based) may find conflicting entities.

### **Solution**: 
1. **Entity Versioning** - Track extraction method and confidence
2. **Conflict Resolution Rules** - Automatic merging with fallback to manual review
3. **Manual Review Interface** - For complex conflicts requiring human judgment

### **Example Conflict Resolution:**
- **Conservative**: "Nike" (95% confidence, from labels)
- **Enhanced**: "Nike Inc" (85% confidence, from NER model)
- **Resolution**: Merge as same entity, keep "Nike" as primary name, "Nike Inc" as alternative

### **Entity Conflict Resolver:**
```python
# entity_resolver.py (NEW FILE)
class EntityConflictResolver:
    """Resolves conflicts between different entity extraction methods"""
    
    def resolve_entity_conflicts(self, conservative_entities: List[Dict], 
                                enhanced_entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Resolve conflicts between conservative and enhanced extraction"""
        
        resolved = []
        conflicts = []
        
        for enhanced in enhanced_entities:
            matches = self._find_potential_matches(enhanced, conservative_entities)
            
            if not matches:
                # New entity discovered by enhanced extraction
                if enhanced['confidence'] > 0.8:  # High confidence threshold
                    resolved.append(self._mark_as_new_discovery(enhanced))
                else:
                    conflicts.append(enhanced)  # Flag for manual review
            
            elif len(matches) == 1:
                # Single match - merge information
                resolved.append(self._merge_entities(matches[0], enhanced))
            
            else:
                # Multiple matches - flag for manual resolution
                conflicts.append({
                    'enhanced_entity': enhanced,
                    'potential_matches': matches,
                    'resolution_needed': 'multiple_matches'
                })
        
        return resolved, conflicts
```

## ⚡ Benefits of This Approach

1. **3-4x Faster Initial Results** (40-65 min vs 2-4 hours)
2. **No Hardcoded Values** - Fully configurable system
3. **Graceful Entity Evolution** - Conservative → Enhanced without conflicts
4. **Production Ready** - Proper error handling, logging, monitoring
5. **Better UX** - System remains responsive during processing
6. **Scalability** - Can handle much larger datasets
7. **Reliability** - Graceful degradation vs complete failure

## 🎯 Strategic Implementation Priority

### **Week 1**: Foundation
- Add fast mode to current system
- Create global configuration system
- Test fast processing with existing data

### **Week 2**: Batch Processing
- Build batch processor for sentiment
- Implement entity conflict resolution
- Add processing status tracking

### **Week 3**: Advanced Features
- Add semantic batch processing
- Create scheduler system
- Implement quality monitoring

### **Week 4**: Production Deployment
- Full scheduling and monitoring
- Performance testing and optimization
- Documentation and training

**The phased approach transforms your system from "batch ETL" to "streaming + batch" architecture while maintaining data quality and handling entity conflicts intelligently!**
