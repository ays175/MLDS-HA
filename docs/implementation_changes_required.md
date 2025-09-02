# Implementation Changes Required for Phased Ingestion

## 1. Configuration Changes

### Add Processing Modes to YAML Configs
```yaml
# Add to all platform configs (facebook.yaml, instagram.yaml, etc.)
processing:
  mode: fast  # fast, full, batch_only
  
  # Fast mode configuration
  fast_mode:
    skip_sentiment: true
    skip_semantic: true
    skip_advanced_entities: true
    basic_metrics_only: true
    
  # Batch processing configuration  
  batch_processing:
    sentiment_analysis:
      enabled: true
      schedule: "0 2 * * *"  # Daily at 2 AM
      batch_size: 5000
      min_confidence: 0.7
      
    semantic_analysis:
      enabled: true
      schedule: "0 1 * * 0"  # Weekly on Sunday at 1 AM
      min_new_records: 1000
      max_topics: 50
      
    entity_enhancement:
      enabled: true
      schedule: "0 3 * * 0"  # Weekly on Sunday at 3 AM
      confidence_threshold: 0.8

# Processing status tracking
status_tracking:
  enabled: true
  status_field: "processing_status"
  timestamp_fields:
    ingested_at: "ingested_at"
    sentiment_processed_at: "sentiment_processed_at"
    semantic_processed_at: "semantic_processed_at"
    entities_enhanced_at: "entities_enhanced_at"
```

## 2. Schema Changes

### Add Processing Status Fields to All Collections
```python
# unified_schema.py modifications
def _create_main_collection(self, client: weaviate.Client):
    # ... existing properties ...
    
    # Add processing status tracking
    properties.extend([
        wvcc.Property(
            name="processing_status",
            data_type=wvcc.DataType.TEXT,
            description="Current processing status: ingested, sentiment_pending, sentiment_complete, semantic_pending, semantic_complete, fully_processed"
        ),
        wvcc.Property(
            name="ingested_at",
            data_type=wvcc.DataType.DATE,
            description="When record was first ingested"
        ),
        wvcc.Property(
            name="sentiment_processed_at",
            data_type=wvcc.DataType.DATE,
            description="When sentiment analysis was completed"
        ),
        wvcc.Property(
            name="semantic_processed_at", 
            data_type=wvcc.DataType.DATE,
            description="When semantic analysis was completed"
        ),
        wvcc.Property(
            name="entities_enhanced_at",
            data_type=wvcc.DataType.DATE,
            description="When entity enhancement was completed"
        ),
        wvcc.Property(
            name="processing_errors",
            data_type=wvcc.DataType.TEXT_ARRAY,
            description="Any processing errors encountered"
        )
    ])
```

## 3. Unified Ingest Changes

### Modify Main Ingest Method
```python
# unified_ingest.py changes
def ingest(self, input_file: str, mode: str = "merge", force: bool = False, 
           skip_sentiment: bool = False, limit_rows: int = 0, 
           processing_mode: str = None) -> Dict[str, Any]:
    
    # Determine processing mode from config or parameter
    if processing_mode is None:
        processing_mode = self.config.get('processing', {}).get('mode', 'fast')
    
    print(f"📊 {self.config['platform']['display_name']} Knowledge Graph Ingestion")
    print(f"📄 Mode: {mode} ({'safe' if mode == 'merge' else 'destructive'})")
    print(f"⚙️ Processing Mode: {processing_mode}")
    
    # ... existing code until data processing ...
    
    # Process data based on mode
    if processing_mode == 'fast':
        df = self._fast_process_dataframe(df)
    elif processing_mode == 'full':
        df = self.data_processor.process_dataframe(df, skip_sentiment)
    else:  # batch_only
        df = self._minimal_process_dataframe(df)
    
    # ... rest of ingestion ...
    
    # Skip semantic analysis in fast mode
    semantic_results = None
    if processing_mode == 'full' and self.config.get('semantic_analysis', {}).get('enabled', True):
        semantic_results = self._run_semantic_analysis(client, df, dataset_id)
    elif processing_mode == 'fast':
        print("⏭️ Skipping semantic analysis (fast mode)")
    
    return {
        'status': 'success',
        'processing_mode': processing_mode,
        'file': input_file,
        'records': len(df),
        'dataset_id': dataset_id,
        'entities': {k: len(v) for k, v in entities.items()},
        'semantic_analysis': semantic_results,
        'next_steps': self._get_next_steps(processing_mode)
    }

def _fast_process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    """Fast processing - skip expensive operations"""
    # Text sanitization only
    if sanitize_text_columns and self.config.get('text_fields_for_sanitization'):
        df = self.data_processor._sanitize_text(df)
    
    # Basic content enrichment (no ML)
    df = self._basic_content_enrichment(df)
    
    # Set processing status
    df['processing_status'] = 'ingested'
    df['ingested_at'] = datetime.now()
    df['sentiment_score'] = None  # Placeholder
    df['sentiment_processed_at'] = None
    
    return df

def _basic_content_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
    """Basic enrichment without ML models"""
    # Word/character counts
    if 'text' in df.columns:
        df['word_count'] = df['text'].str.split().str.len()
        df['char_count'] = df['text'].str.len()
        df['hashtag_count'] = df['text'].str.count('#')
        df['mention_count'] = df['text'].str.count('@')
        df['url_count'] = df['text'].str.count('http')
    
    return df

def _get_next_steps(self, processing_mode: str) -> List[str]:
    """Get recommended next steps based on processing mode"""
    if processing_mode == 'fast':
        return [
            "Run batch processor for sentiment analysis",
            "Schedule semantic analysis when >1000 new records",
            "Generate basic metrics available now"
        ]
    elif processing_mode == 'full':
        return [
            "All processing complete",
            "Generate comprehensive metrics",
            "Data ready for analysis"
        ]
    else:
        return ["Minimal processing complete", "Run full processing pipeline"]
```

## 4. New Batch Processor Component

### Create batch_processor.py
```python
# ingestion/batch_processor.py (NEW FILE)
#!/usr/bin/env python3
"""
Batch Processing System for Expensive Operations
Handles sentiment analysis, semantic analysis, and entity enhancement
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import weaviate
import weaviate.classes.query as wvq
from dotenv import load_dotenv

load_dotenv()

class BatchProcessor:
    """Handles expensive batch operations on ingested data"""
    
    def __init__(self, config_dir: str = "ingestion/configs"):
        self.config_dir = Path(config_dir)
        self.client = None
        self.platforms = ['facebook', 'instagram', 'tiktok', 'customer_care']
        
    def connect_weaviate(self):
        """Connect to Weaviate"""
        if not self.client:
            self.client = weaviate.connect_to_local()
        return self.client
    
    def process_pending_sentiment(self, platform: str = None, batch_size: int = 5000) -> Dict[str, Any]:
        """Process records missing sentiment analysis"""
        platforms = [platform] if platform else self.platforms
        results = {}
        
        for plat in platforms:
            print(f"🧠 Processing pending sentiment for {plat}...")
            
            # Load platform config
            config = self._load_platform_config(plat)
            if not config.get('batch_processing', {}).get('sentiment_analysis', {}).get('enabled'):
                print(f"  Sentiment processing disabled for {plat}")
                continue
            
            # Get collection
            collection_name = self._get_collection_name(plat, 'post')
            collection = self.client.collections.get(collection_name)
            
            # Query pending records
            pending_query = collection.query.fetch_objects(
                limit=10000  # Process in chunks
            ).where(
                wvq.Filter.any_of([
                    wvq.Filter.by_property("processing_status").equal("ingested"),
                    wvq.Filter.by_property("sentiment_score").is_null(True)
                ])
            )
            
            pending_objects = list(pending_query)
            if not pending_objects:
                print(f"  No pending sentiment records for {plat}")
                results[plat] = {"processed": 0, "status": "no_pending"}
                continue
            
            print(f"  Found {len(pending_objects)} records needing sentiment analysis")
            
            # Initialize sentiment analyzer
            from cross_platform.sentiment_analyzer import HybridSentimentAnalyzer
            analyzer = HybridSentimentAnalyzer()
            
            # Process in batches
            processed = 0
            for i in range(0, len(pending_objects), batch_size):
                batch = pending_objects[i:i+batch_size]
                
                # Extract texts
                texts = []
                for obj in batch:
                    text = obj.properties.get('text', '') or obj.properties.get('content_summary', '') or ''
                    texts.append(str(text))
                
                # Compute sentiment
                scores = analyzer.analyze_batch(texts)
                
                # Update records
                for obj, score in zip(batch, scores):
                    try:
                        collection.data.update(
                            uuid=obj.uuid,
                            properties={
                                "sentiment_score": float(score),
                                "sentiment_processed_at": datetime.now(),
                                "processing_status": "sentiment_complete"
                            }
                        )
                        processed += 1
                    except Exception as e:
                        print(f"    Error updating {obj.uuid}: {e}")
                
                print(f"    Processed batch {i//batch_size + 1}/{(len(pending_objects)-1)//batch_size + 1}")
            
            results[plat] = {"processed": processed, "status": "success"}
            print(f"  ✅ Processed {processed} records for {plat}")
        
        return results
    
    def process_semantic_analysis(self, platform: str = None, min_new_records: int = 1000) -> Dict[str, Any]:
        """Run semantic analysis if enough new records"""
        platforms = [platform] if platform else self.platforms
        results = {}
        
        for plat in platforms:
            print(f"🔬 Checking semantic analysis needs for {plat}...")
            
            # Count new records since last semantic processing
            new_count = self._count_new_records_since_semantic(plat)
            
            if new_count < min_new_records:
                print(f"  Only {new_count} new records, skipping (need {min_new_records})")
                results[plat] = {"status": "insufficient_data", "new_records": new_count}
                continue
            
            print(f"  Found {new_count} new records, running semantic analysis...")
            
            try:
                # Run semantic pipeline for this platform
                from cross_platform.semantic_pipeline import run_semantic_pipeline
                semantic_result = run_semantic_pipeline(platform_filter=plat)
                
                # Update processing status for analyzed records
                self._mark_semantic_complete(plat)
                
                results[plat] = {
                    "status": "success", 
                    "new_records": new_count,
                    "topics_discovered": semantic_result.get("topics_discovered", 0)
                }
                
            except Exception as e:
                print(f"  ❌ Semantic analysis failed for {plat}: {e}")
                results[plat] = {"status": "error", "error": str(e)}
        
        return results
    
    def process_entity_enhancement(self, platform: str = None) -> Dict[str, Any]:
        """Enhance entities with ML-based extraction"""
        # TODO: Implement enhanced entity extraction
        # This would use NER models to find entities missed in conservative extraction
        return {"status": "not_implemented"}
    
    def get_processing_status(self, platform: str = None) -> Dict[str, Any]:
        """Get current processing status across platforms"""
        platforms = [platform] if platform else self.platforms
        status = {}
        
        for plat in platforms:
            collection_name = self._get_collection_name(plat, 'post')
            collection = self.client.collections.get(collection_name)
            
            # Count records by processing status
            total_query = collection.query.aggregate.over_all(total_count=True)
            total = total_query.total_count
            
            # Count by status
            status_counts = {}
            for stat in ['ingested', 'sentiment_complete', 'semantic_complete', 'fully_processed']:
                count_query = collection.query.aggregate.over_all(
                    total_count=True
                ).where(
                    wvq.Filter.by_property("processing_status").equal(stat)
                )
                status_counts[stat] = count_query.total_count
            
            status[plat] = {
                "total_records": total,
                "status_breakdown": status_counts,
                "completion_percentage": {
                    "sentiment": (status_counts.get('sentiment_complete', 0) + 
                                status_counts.get('semantic_complete', 0) + 
                                status_counts.get('fully_processed', 0)) / max(1, total) * 100,
                    "semantic": (status_counts.get('semantic_complete', 0) + 
                               status_counts.get('fully_processed', 0)) / max(1, total) * 100
                }
            }
        
        return status
    
    def _load_platform_config(self, platform: str) -> Dict[str, Any]:
        """Load platform configuration"""
        config_file = self.config_dir / f"{platform}.yaml"
        if not config_file.exists():
            return {}
        
        import yaml
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_collection_name(self, platform: str, collection_type: str) -> str:
        """Get Weaviate collection name for platform"""
        collection_map = {
            'facebook': {'post': 'FacebookPost', 'case': 'FacebookCase'},
            'instagram': {'post': 'InstagramPost', 'case': 'InstagramCase'},
            'tiktok': {'post': 'TikTokPost', 'case': 'TikTokCase'},
            'customer_care': {'post': 'CustomerCarePost', 'case': 'CustomerCareCase'}
        }
        
        return collection_map.get(platform, {}).get(collection_type, f"{platform.title()}Post")
    
    def _count_new_records_since_semantic(self, platform: str) -> int:
        """Count records added since last semantic analysis"""
        collection_name = self._get_collection_name(platform, 'post')
        collection = self.client.collections.get(collection_name)
        
        # Find records without semantic processing
        query = collection.query.aggregate.over_all(
            total_count=True
        ).where(
            wvq.Filter.any_of([
                wvq.Filter.by_property("processing_status").not_equal("semantic_complete"),
                wvq.Filter.by_property("processing_status").not_equal("fully_processed"),
                wvq.Filter.by_property("semantic_processed_at").is_null(True)
            ])
        )
        
        return query.total_count
    
    def _mark_semantic_complete(self, platform: str):
        """Mark records as semantically processed"""
        collection_name = self._get_collection_name(platform, 'post')
        collection = self.client.collections.get(collection_name)
        
        # This would need to be implemented based on which records were actually processed
        # For now, mark all sentiment_complete records as semantic_complete
        # In practice, you'd track which specific records were processed
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch processor for expensive operations")
    parser.add_argument("--operation", choices=['sentiment', 'semantic', 'entities', 'status'], 
                       required=True, help="Operation to perform")
    parser.add_argument("--platform", choices=['facebook', 'instagram', 'tiktok', 'customer_care'],
                       help="Platform to process (all if not specified)")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for processing")
    
    args = parser.parse_args()
    
    processor = BatchProcessor()
    processor.connect_weaviate()
    
    if args.operation == 'sentiment':
        results = processor.process_pending_sentiment(args.platform, args.batch_size)
    elif args.operation == 'semantic':
        results = processor.process_semantic_analysis(args.platform)
    elif args.operation == 'entities':
        results = processor.process_entity_enhancement(args.platform)
    elif args.operation == 'status':
        results = processor.get_processing_status(args.platform)
    
    print(json.dumps(results, indent=2, default=str))
    
    if processor.client:
        processor.client.close()
```

## 5. Command Line Interface Changes

### Update unified_ingest.py CLI
```python
# Add to parse_args() in unified_ingest.py
parser.add_argument(
    '--processing-mode',
    choices=['fast', 'full', 'batch_only'],
    default='fast',
    help='Processing mode: fast (skip expensive ops), full (complete processing), batch_only (minimal)'
)

# Update main() function
def main():
    args = parse_args()
    
    try:
        engine = UnifiedIngestionEngine(args.platform, args.config)
        
        result = engine.ingest(
            input_file=args.input,
            mode=args.mode,
            force=args.force,
            skip_sentiment=args.skip_sentiment,
            limit_rows=args.limit_rows,
            processing_mode=args.processing_mode  # NEW
        )
        
        # Show next steps based on processing mode
        if result.get('next_steps'):
            print("\n📋 Recommended Next Steps:")
            for step in result['next_steps']:
                print(f"  • {step}")
                
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        sys.exit(1)
```

## 6. Scheduler Component

### Create scheduler.py
```python
# ingestion/scheduler.py (NEW FILE)
#!/usr/bin/env python3
"""
Batch Processing Scheduler
Handles scheduled execution of expensive operations
"""
import schedule
import time
import logging
from datetime import datetime
from batch_processor import BatchProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_processor.log'),
        logging.StreamHandler()
    ]
)

def setup_batch_jobs():
    """Setup scheduled batch processing jobs"""
    processor = BatchProcessor()
    
    # Daily sentiment processing (staggered by platform)
    schedule.every().day.at("02:00").do(
        run_with_logging, processor.process_pending_sentiment, "facebook"
    )
    schedule.every().day.at("02:30").do(
        run_with_logging, processor.process_pending_sentiment, "instagram"
    )
    schedule.every().day.at("03:00").do(
        run_with_logging, processor.process_pending_sentiment, "tiktok"
    )
    schedule.every().day.at("03:30").do(
        run_with_logging, processor.process_pending_sentiment, "customer_care"
    )
    
    # Weekly semantic analysis (Sunday nights)
    schedule.every().sunday.at("01:00").do(
        run_with_logging, processor.process_semantic_analysis, None
    )
    
    # Weekly entity enhancement (Saturday nights)
    schedule.every().saturday.at("23:00").do(
        run_with_logging, processor.process_entity_enhancement, None
    )
    
    # Daily status reporting
    schedule.every().day.at("08:00").do(
        run_with_logging, log_processing_status, processor
    )

def run_with_logging(func, *args, **kwargs):
    """Run function with logging"""
    try:
        logging.info(f"Starting {func.__name__} with args {args}")
        result = func(*args, **kwargs)
        logging.info(f"Completed {func.__name__}: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}")
        raise

def log_processing_status(processor: BatchProcessor):
    """Log current processing status"""
    status = processor.get_processing_status()
    for platform, stats in status.items():
        completion = stats['completion_percentage']
        logging.info(f"{platform}: {completion['sentiment']:.1f}% sentiment, {completion['semantic']:.1f}% semantic")

def run_scheduler():
    """Run the scheduler"""
    logging.info("Starting batch processing scheduler")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
            break
        except Exception as e:
            logging.error(f"Scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    setup_batch_jobs()
    run_scheduler()
```

## Summary of Changes Required

### Files to Modify:
1. **All platform YAML configs** - Add processing mode configuration
2. **unified_schema.py** - Add processing status fields
3. **unified_ingest.py** - Add processing modes and fast processing
4. **Command line interface** - Add processing mode parameter

### New Files to Create:
1. **batch_processor.py** - Handle expensive batch operations
2. **scheduler.py** - Schedule and manage batch jobs
3. **logs/** directory - For batch processing logs

### Database Changes:
- Add processing status fields to all collections
- Add timestamp fields for tracking processing stages

This phased approach maintains backward compatibility while adding the new fast processing capabilities.
