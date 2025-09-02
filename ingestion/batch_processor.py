#!/usr/bin/env python3
"""
Simple Batch Processor for Expensive Operations
Handles sentiment analysis and semantic analysis on ingested data
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import weaviate
import weaviate.classes.query as wvq
from dotenv import load_dotenv

# Add parent directory to path for config import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.global_config import get_global_config

load_dotenv()


class BatchProcessor:
    """Simple batch processor for expensive operations"""
    
    def __init__(self):
        self.client = None
        self.config = get_global_config()
        self.platforms = ['facebook', 'instagram', 'tiktok', 'customer_care']
    
    def connect_weaviate(self):
        """Connect to Weaviate"""
        if not self.client:
            self.client = weaviate.connect_to_local()
        return self.client
    
    def process_pending_sentiment(self, platform: str, batch_size: int = None) -> Dict[str, Any]:
        """Process records missing sentiment analysis"""
        print(f"🧠 Processing pending sentiment for {platform}...")
        
        # Use config batch size if not provided
        if batch_size is None:
            batch_size = self.config.get('processing.batch_size', 1000)
        
        # Get collection name from config
        collection_name = self.config.get_collection_name(platform)
        if not collection_name:
            return {"error": f"Unknown platform: {platform}"}
        
        try:
            collection = self.client.collections.get(collection_name)
            
            # Find records with processing_status = 'ingested' (no sentiment yet)
            pending_query = collection.query.fetch_objects(
                limit=batch_size
            ).where(
                wvq.Filter.by_property("processing_status").equal("ingested")
            )
            
            pending_objects = list(pending_query)
            
            if not pending_objects:
                print(f"  No pending sentiment records for {platform}")
                return {"processed": 0, "status": "no_pending"}
            
            print(f"  Found {len(pending_objects)} records needing sentiment analysis")
            
            # Initialize sentiment analyzer
            from cross_platform.sentiment_analyzer import HybridSentimentAnalyzer
            analyzer = HybridSentimentAnalyzer()
            
            # Extract texts and compute sentiment
            texts = []
            for obj in pending_objects:
                # Get text field based on platform
                if platform == 'customer_care':
                    text = obj.properties.get('content_summary', '') or obj.properties.get('description', '')
                else:
                    text = obj.properties.get('text', '') or obj.properties.get(f'{platform}_content', '')
                texts.append(str(text) if text else '')
            
            # Compute sentiment scores
            scores = analyzer.analyze_batch(texts)
            
            # Update records
            processed = 0
            for obj, score in zip(pending_objects, scores):
                try:
                    # Update with sentiment and new status
                    update_props = {
                        "sentiment_score": float(score),
                        "sentiment_processed_at": datetime.now(),
                        "processing_status": "sentiment_complete"
                    }
                    
                    collection.data.update(uuid=obj.uuid, properties=update_props)
                    processed += 1
                    
                except Exception as e:
                    print(f"    Error updating {obj.uuid}: {e}")
            
            print(f"  ✅ Processed {processed} records for {platform}")
            return {"processed": processed, "status": "success"}
            
        except Exception as e:
            print(f"  ❌ Error processing {platform}: {e}")
            return {"error": str(e)}
    
    def process_semantic_analysis(self, platform: str, min_records: int = 100) -> Dict[str, Any]:
        """Run semantic analysis if enough records with sentiment"""
        print(f"🔬 Checking semantic analysis needs for {platform}...")
        
        collection_map = {
            'facebook': 'FacebookPost',
            'instagram': 'InstagramPost',
            'tiktok': 'TikTokPost', 
            'customer_care': 'CustomerCareCase'
        }
        
        collection_name = collection_map.get(platform)
        if not collection_name:
            return {"error": f"Unknown platform: {platform}"}
        
        try:
            collection = self.client.collections.get(collection_name)
            
            # Count records ready for semantic analysis (have sentiment)
            ready_query = collection.query.aggregate.over_all(
                total_count=True
            ).where(
                wvq.Filter.by_property("processing_status").equal("sentiment_complete")
            )
            
            ready_count = ready_query.total_count
            
            if ready_count < min_records:
                print(f"  Only {ready_count} records ready, need {min_records}")
                return {"status": "insufficient_data", "ready_records": ready_count}
            
            print(f"  Found {ready_count} records ready for semantic analysis")
            
            # Run semantic pipeline for this platform
            try:
                from cross_platform.semantic_pipeline import run_semantic_pipeline
                result = run_semantic_pipeline(platform_filter=platform)
                
                # Mark records as semantically processed
                self._mark_semantic_complete(platform)
                
                return {
                    "status": "success",
                    "ready_records": ready_count,
                    "topics_discovered": result.get("topics_discovered", 0)
                }
                
            except Exception as e:
                print(f"  ❌ Semantic analysis failed: {e}")
                return {"status": "error", "error": str(e)}
                
        except Exception as e:
            print(f"  ❌ Error checking {platform}: {e}")
            return {"error": str(e)}
    
    def get_processing_status(self, platform: str = None) -> Dict[str, Any]:
        """Get processing status for platforms"""
        platforms = [platform] if platform else self.platforms
        status = {}
        
        collection_map = {
            'facebook': 'FacebookPost',
            'instagram': 'InstagramPost',
            'tiktok': 'TikTokPost',
            'customer_care': 'CustomerCareCase'
        }
        
        for plat in platforms:
            collection_name = collection_map.get(plat)
            if not collection_name:
                continue
                
            try:
                collection = self.client.collections.get(collection_name)
                
                # Count total records
                total_query = collection.query.aggregate.over_all(total_count=True)
                total = total_query.total_count
                
                # Count by status
                status_counts = {}
                for stat in ['ingested', 'sentiment_complete', 'semantic_complete']:
                    count_query = collection.query.aggregate.over_all(
                        total_count=True
                    ).where(
                        wvq.Filter.by_property("processing_status").equal(stat)
                    )
                    status_counts[stat] = count_query.total_count
                
                # Calculate percentages
                sentiment_complete = status_counts.get('sentiment_complete', 0) + status_counts.get('semantic_complete', 0)
                semantic_complete = status_counts.get('semantic_complete', 0)
                
                status[plat] = {
                    "total_records": total,
                    "status_counts": status_counts,
                    "sentiment_percentage": round(sentiment_complete / max(1, total) * 100, 1),
                    "semantic_percentage": round(semantic_complete / max(1, total) * 100, 1)
                }
                
            except Exception as e:
                status[plat] = {"error": str(e)}
        
        return status
    
    def _mark_semantic_complete(self, platform: str):
        """Mark sentiment_complete records as semantic_complete"""
        collection_map = {
            'facebook': 'FacebookPost',
            'instagram': 'InstagramPost',
            'tiktok': 'TikTokPost',
            'customer_care': 'CustomerCareCase'
        }
        
        collection_name = collection_map.get(platform)
        if not collection_name:
            return
        
        try:
            collection = self.client.collections.get(collection_name)
            
            # Get records with sentiment_complete status
            query = collection.query.fetch_objects(
                limit=10000
            ).where(
                wvq.Filter.by_property("processing_status").equal("sentiment_complete")
            )
            
            # Update to semantic_complete
            for obj in query:
                try:
                    collection.data.update(
                        uuid=obj.uuid,
                        properties={
                            "processing_status": "semantic_complete",
                            "semantic_processed_at": datetime.now()
                        }
                    )
                except:
                    pass  # Silent fail for individual updates
                    
        except Exception as e:
            print(f"Warning: Could not mark semantic complete for {platform}: {e}")


def main():
    """CLI interface"""
    import argparse
    parser = argparse.ArgumentParser(description="Batch processor for expensive operations")
    parser.add_argument("--operation", choices=['sentiment', 'semantic', 'status'], 
                       required=True, help="Operation to perform")
    parser.add_argument("--platform", choices=['facebook', 'instagram', 'tiktok', 'customer_care'],
                       help="Platform to process (all if not specified)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    
    args = parser.parse_args()
    
    processor = BatchProcessor()
    processor.connect_weaviate()
    
    try:
        if args.operation == 'sentiment':
            if args.platform:
                results = processor.process_pending_sentiment(args.platform, args.batch_size)
            else:
                results = {}
                for platform in processor.platforms:
                    results[platform] = processor.process_pending_sentiment(platform, args.batch_size)
        
        elif args.operation == 'semantic':
            if args.platform:
                results = processor.process_semantic_analysis(args.platform)
            else:
                results = {}
                for platform in processor.platforms:
                    results[platform] = processor.process_semantic_analysis(platform)
        
        elif args.operation == 'status':
            results = processor.get_processing_status(args.platform)
        
        print(json.dumps(results, indent=2, default=str))
        
    finally:
        if processor.client:
            processor.client.close()


if __name__ == "__main__":
    main()
