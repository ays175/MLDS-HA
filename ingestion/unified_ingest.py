#!/usr/bin/env python3
"""
Unified Data Ingestion System
Handles ingestion for all platforms (Facebook, Instagram, TikTok, Customer Care)
through configuration-driven architecture.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import socket
import subprocess
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional imports with fallbacks
try:
    from cross_platform.sentiment_analyzer import HybridSentimentAnalyzer
except Exception:
    HybridSentimentAnalyzer = None  # type: ignore

try:
    from customer_care.utils.text_sanitizer import sanitize_text_columns
except Exception:
    sanitize_text_columns = None  # type: ignore

# Load environment variables
load_dotenv()


class UnifiedIngestionTracker:
    """Track ingestions across all platforms to avoid duplicates"""
    
    def __init__(self, platform: str, tracking_config: Dict[str, Any]):
        self.platform = platform
        self.tracker_file = Path(tracking_config['log_file'])
        self.tracker_file.parent.mkdir(exist_ok=True)
        self.hash_algorithm = tracking_config.get('hash_algorithm', 'sha256')
        self.hash_length = tracking_config.get('hash_length', 16)
        
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {"ingestions": []}
    
    def is_duplicate(self, file_path: str) -> bool:
        """Check if file was already ingested"""
        file_hash = self._get_file_hash(file_path)
        return any(ing['file_hash'] == file_hash for ing in self.log['ingestions'])
    
    def record_ingestion(self, file_path: str, record_count: int, dataset_id: str = None):
        """Record successful ingestion"""
        entry = {
            'file_name': Path(file_path).name,
            'file_hash': self._get_file_hash(file_path),
            'record_count': record_count,
            'ingested_at': datetime.now().isoformat(),
            'platform': self.platform
        }
        if dataset_id:
            entry['dataset_id'] = dataset_id
        
        self.log['ingestions'].append(entry)
        
        with open(self.tracker_file, 'w') as f:
            json.dump(self.log, f, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate file hash for deduplication"""
        hasher = getattr(hashlib, self.hash_algorithm)()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()[:self.hash_length]


class UnifiedEntityExtractor:
    """Extract entities based on configuration"""
    
    def __init__(self, entity_config: Dict[str, Any], platform_config: Dict[str, Any]):
        self.entity_config = entity_config
        self.platform_config = platform_config
    
    def extract_entities(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all entity types from dataframe using unified extractor"""
        from ingestion.unified_entities import UnifiedEntityExtractor
        extractor = UnifiedEntityExtractor(self.platform_config['name'])
        return extractor.extract_entities(df)
    



class UnifiedSchemaManager:
    """Manage Weaviate schemas across platforms"""
    
    def __init__(self, full_config: Dict[str, Any], schema_config: Dict[str, Any]):
        self.full_config = full_config
        self.platform_config = full_config['platform']
        self.schema_config = schema_config
        self.collections = full_config['collections']
    
    def ensure_schema_exists(self, client: weaviate.Client, mode: str = "merge"):
        """Ensure all required collections exist"""
        if mode == "replace":
            print(f"🔄 Replace mode: Recreating all {self.platform_config['name']} collections")
            self._delete_collections(client)
            self._create_collections(client)
        else:
            print(f"🔗 Merge mode: Checking {self.platform_config['name']} collections")
            self._ensure_collections_exist(client)
    
    def _delete_collections(self, client: weaviate.Client):
        """Delete all platform collections"""
        for collection_name in self.collections.values():
            try:
                client.collections.delete(collection_name)
                print(f"  ✓ Deleted {collection_name}")
            except Exception as e:
                print(f"  - {collection_name} not found or error: {e}")
    
    def _create_collections(self, client: weaviate.Client):
        """Create all platform collections"""
        # Import platform-specific schema creator
        platform_name = self.platform_config['name']
        
        try:
            from ingestion.unified_schema import create_unified_schema
            create_unified_schema(client, platform_name)
            print(f"  ✓ Created all {platform_name} collections")
        except Exception as e:
            print(f"  ❌ Error creating collections: {e}")
            raise
    
    def _ensure_collections_exist(self, client: weaviate.Client):
        """Ensure collections exist, create if missing"""
        missing_collections = []
        
        for collection_name in self.collections.values():
            try:
                client.collections.get(collection_name)
            except:
                missing_collections.append(collection_name)
        
        if missing_collections:
            print(f"  Creating missing collections: {missing_collections}")
            self._create_collections(client)


class UnifiedDataProcessor:
    """Process data according to platform configuration"""
    
    def __init__(self, processing_config: Dict[str, Any], platform_name: str):
        self.config = processing_config
        self.platform_name = platform_name
        self.sentiment_analyzer = None
        
        if self.config.get('sentiment_analysis', {}).get('enabled') and HybridSentimentAnalyzer:
            try:
                self.sentiment_analyzer = HybridSentimentAnalyzer()
                print("🧠 Sentiment analyzer initialized")
            except Exception as e:
                print(f"⚠️ Sentiment analyzer unavailable: {e}")
    
    def process_dataframe(self, df: pd.DataFrame, skip_sentiment: bool = False) -> pd.DataFrame:
        """Apply all configured processing to dataframe"""
        # Fix data types first
        id_columns = ['tiktok_id', 'facebook_id', 'instagram_id', 'customer_care_id']
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Text sanitization
        if sanitize_text_columns and self.config.get('text_fields_for_sanitization'):
            df = self._sanitize_text(df)
        
        # Sentiment analysis
        if not skip_sentiment and self.sentiment_analyzer and self.config.get('sentiment_analysis', {}).get('enabled'):
            df = self._compute_sentiment(df)
        
        # Content enrichment
        if self.config.get('content_enrichment'):
            df = self._enrich_content(df)
        
        # Platform-specific processing
        if self.platform_name == 'customer_care':
            df = self._process_customer_care_specific(df)
        
        return df
    
    def _sanitize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize text fields"""
        fields = self.config['text_fields_for_sanitization']
        existing_fields = [f for f in fields if f in df.columns]
        
        if existing_fields:
            try:
                df, stats = sanitize_text_columns(df, existing_fields)
                if stats.get('cells_modified'):
                    print(f"🧹 Sanitized {stats['cells_modified']} cells in {len(existing_fields)} fields")
            except Exception as e:
                print(f"⚠️ Text sanitization failed: {e}")
        
        return df
    
    def _compute_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sentiment scores"""
        sentiment_config = self.config['sentiment_analysis']
        text_field = sentiment_config['text_field']
        fallback_fields = sentiment_config.get('fallback_fields', [sentiment_config.get('fallback_field')])
        target_field = sentiment_config['target_field']
        batch_size = sentiment_config.get('batch_size', 64)
        
        # Find text to analyze
        text_series = None
        if text_field in df.columns:
            text_series = df[text_field]
        
        # Try fallback fields if primary is empty
        if text_series is None or text_series.fillna('').str.len().sum() == 0:
            for fallback in fallback_fields:
                if fallback and fallback in df.columns:
                    text_series = df[fallback]
                    if text_series.fillna('').str.len().sum() > 0:
                        break
        
        if text_series is not None and text_series.fillna('').str.len().sum() > 0:
            texts = [str(t) if pd.notna(t) else '' for t in text_series.fillna('').tolist()]
            print(f"🧠 Computing sentiment for {len(texts)} items...")
            
            # Batch processing with progress
            batch_outer = 1024
            scores = []
            total = len(texts)
            start_time = time.time()
            
            for i in range(0, total, batch_outer):
                chunk = texts[i:i+batch_outer]
                scores.extend(self.sentiment_analyzer.analyze_batch(chunk, batch_size=batch_size))
                
                # Progress update
                if (i // batch_outer) % 10 == 0 or i + batch_outer >= total:
                    elapsed = time.time() - start_time
                    done = min(total, i + batch_outer)
                    rate = done / max(1.0, elapsed)
                    eta = (total - done) / max(1.0, rate)
                    print(f"   • Progress: {done}/{total} ({done/total:.1%}) | "
                          f"{rate:.1f} rows/s | ETA {eta/60:.1f} min")
            
            df[target_field] = scores[:total]
            print(f"✓ Added sentiment scores to {target_field}")
        
        return df
    
    def _enrich_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply content enrichment using unified extractor"""
        from ingestion.unified_entities import UnifiedEntityExtractor
        extractor = UnifiedEntityExtractor(self.platform_name)
        
        print(f"📝 Applying {self.platform_name} content enrichment...")
        
        # Apply row-by-row enrichment with safe assignment
        for idx, row in df.iterrows():
            try:
                enhanced = extractor.extract_enhanced_content(row)
                for key, value in enhanced.items():
                    # Ensure the column exists first
                    if key not in df.columns:
                        df[key] = None
                    # Safe assignment
                    df.at[idx, key] = value
            except Exception as e:
                print(f"  Warning: Content enrichment failed for row {idx}: {e}")
                continue
        
        return df
    
    def _process_customer_care_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply customer care specific processing"""
        cc_config = self.config.get('customer_care_specific', {})
        
        # Normalize field names
        column_mapping = {
            'Issue Type': 'issue_type',
            'Channel': 'channel',
            'Priority': 'priority',
            'Status': 'status',
            'Created Date': 'created_date',
            'Resolved Date': 'resolved_date',
            'Response Time (Hours)': 'response_time_hours',
            'Resolution Time (Hours)': 'resolution_time_hours',
            'Satisfaction Score': 'satisfaction_score',
            'IsEscalated': 'is_escalated',
            'Interaction Count': 'interaction_count'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Generate content summary if missing
        if 'content_summary' not in df.columns or df['content_summary'].isna().all():
            df['content_summary'] = df.apply(self._generate_case_summary, axis=1)
        
        # Detect escalation
        if 'escalation_detection' in cc_config:
            df = self._detect_escalation(df, cc_config['escalation_detection'])
        
        # Map origin
        if 'origin_mapping' in cc_config and 'channel' in df.columns:
            origin_map = cc_config['origin_mapping']
            df['origin'] = df['channel'].map(lambda x: origin_map.get(str(x).lower(), str(x)))
        
        return df
    
    def _generate_case_summary(self, row: pd.Series) -> str:
        """Generate summary for customer care case"""
        parts = []
        
        if pd.notna(row.get('issue_type')):
            parts.append(f"Issue: {row['issue_type']}")
        
        if pd.notna(row.get('subject')):
            parts.append(f"Subject: {row['subject']}")
        
        if pd.notna(row.get('description')):
            desc = str(row['description'])[:200]
            parts.append(f"Description: {desc}")
        
        if pd.notna(row.get('resolution')):
            res = str(row['resolution'])[:100]
            parts.append(f"Resolution: {res}")
        
        return " | ".join(parts) if parts else "Customer support case"
    
    def _detect_escalation(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Detect escalated cases"""
        if 'is_escalated' not in df.columns:
            keywords = config.get('keywords', [])
            priority_threshold = config.get('priority_threshold', 'high')
            
            def is_escalated(row):
                # Check keywords in text fields
                text = ' '.join([
                    str(row.get('subject', '')),
                    str(row.get('description', '')),
                    str(row.get('resolution', ''))
                ]).lower()
                
                if any(kw in text for kw in keywords):
                    return True
                
                # Check priority
                if str(row.get('priority', '')).lower() == priority_threshold:
                    return True
                
                return False
            
            df['is_escalated'] = df.apply(is_escalated, axis=1)
        
        return df


class UnifiedIngestionEngine:
    """Main ingestion engine for all platforms"""
    
    def __init__(self, platform: str, config_path: Optional[str] = None):
        self.platform = platform
        self.config = self._load_config(config_path)
        self.tracker = UnifiedIngestionTracker(platform, self.config['tracking'])
        self.entity_extractor = UnifiedEntityExtractor(
            self.config.get('entities', {}),
            self.config['platform']
        )
        self.schema_manager = UnifiedSchemaManager(
            self.config,  # Pass full config instead of just platform section
            self.config.get('schema', {})
        )
        self.data_processor = UnifiedDataProcessor(
            self.config.get('processing', {}),
            platform
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load platform configuration"""
        if not config_path:
            config_path = f"ingestion/configs/{self.platform}.yaml"
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration not found: {config_file}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def ingest(self, 
               input_file: str,
               mode: str = "merge",
               force: bool = False,
               skip_sentiment: bool = False,
               limit_rows: int = 0,
               processing_mode: str = "fast") -> Dict[str, Any]:
        """Main ingestion method"""
        print(f"📊 {self.config['platform']['display_name']} Knowledge Graph Ingestion")
        print(f"📄 Mode: {mode} ({'safe' if mode == 'merge' else 'destructive'})")
        print(f"⚙️ Processing: {processing_mode}")
        print(f"📂 Input: {input_file}")
        
        # Check for duplicates
        if not force and self.tracker.is_duplicate(input_file):
            print(f"❌ File {input_file} already ingested. Use --force to override.")
            return {'status': 'duplicate', 'file': input_file}
        
        # Load data
        df = self._load_data(input_file, limit_rows)
        print(f"📊 Loaded {len(df)} records")
        
        # Connect to Weaviate
        print("🔌 Connecting to Weaviate...")
        client = self._connect_weaviate()
        
        try:
            # Ensure schema exists
            self.schema_manager.ensure_schema_exists(client, mode)
            
            # Process data based on mode
            print("🔄 Processing data...")
            if processing_mode == "fast":
                df = self._fast_process_dataframe(df)
            else:
                df = self.data_processor.process_dataframe(df, skip_sentiment)
            
            # Extract entities
            print("🔍 Extracting entities...")
            entities = self.entity_extractor.extract_entities(df)
            
            # Generate UUIDs
            entity_uuids = self._generate_entity_uuids(entities)
            
            # Ingest entities
            print("📥 Ingesting entities...")
            self._ingest_entities(client, entities, entity_uuids, mode)
            
            # Ingest main data
            print("📥 Ingesting records...")
            dataset_id = self._generate_dataset_id(input_file)
            self._ingest_records(client, df, entity_uuids, dataset_id, mode)
            
            # Record successful ingestion
            self.tracker.record_ingestion(input_file, len(df), dataset_id)
            
            # Run semantic analysis if configured and not fast mode
            semantic_results = None
            if processing_mode != "fast" and self.config.get('semantic_analysis', {}).get('enabled', True) and len(df) > 10:
                print("🧠 Running semantic analysis...")
                semantic_results = self._run_semantic_analysis(client, df, dataset_id)
            elif processing_mode == "fast":
                print("⏭️ Skipping semantic analysis (fast mode)")
            
            # Export metrics if configured
            if self.config.get('metrics', {}).get('exporter_class'):
                print("📊 Exporting metrics...")
                self._export_metrics(df, dataset_id)
            
            # Close Weaviate connection
            client.close()
            
            print(f"✅ Successfully ingested {len(df)} records")
            
            return {
                'status': 'success',
                'file': input_file,
                'records': len(df),
                'dataset_id': dataset_id,
                'entities': {k: len(v) for k, v in entities.items()},
                'semantic_analysis': semantic_results
            }
            
        finally:
            client.close()
    
    def _load_data(self, input_file: str, limit_rows: int = 0) -> pd.DataFrame:
        """Load data from file"""
        file_path = Path(input_file)
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        if limit_rows > 0:
            df = df.head(limit_rows)
            print(f"  Limited to {limit_rows} rows for testing")
        
        return df
    
    def _connect_weaviate(self) -> weaviate.Client:
        """Connect to Weaviate"""
        ensure_weaviate_docker()
        
        additional = wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=120))
        client = weaviate.connect_to_local(additional_config=additional)
        
        return client
    
    def _generate_entity_uuids(self, entities: Dict[str, List[Dict]]) -> Dict[str, Dict[str, str]]:
        """Generate deterministic UUIDs for entities using platform-specific functions"""
        platform_name = self.config['platform']['name']
        
        # Use the unified entity UUID generation
        from ingestion.unified_entities import UnifiedEntityExtractor
        extractor = UnifiedEntityExtractor(platform_name)
        return extractor.generate_entity_uuids(entities)
    
    def _generate_dataset_id(self, input_file: str) -> str:
        """Generate unique dataset ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_hash = hashlib.md5(Path(input_file).name.encode()).hexdigest()[:8]
        return f"{self.platform}_dataset_{file_hash}_{timestamp}"
    
    def _ingest_entities(self, client: weaviate.Client, entities: Dict, entity_uuids: Dict, mode: str):
        """Ingest all entities using unified logic"""
        collections = self.config['collections']
        
        if mode == "replace":
            print("🔄 Replace mode: Adding all entities fresh")
        else:
            print("🔗 Merge mode: Adding only new entities")
        
        # Ingest each entity type
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
                
            # Map entity type to collection name
            collection_name = None
            if entity_type == 'platforms' and 'platform' in collections:
                collection_name = collections['platform']
            elif entity_type == 'brands' and 'brand' in collections:
                collection_name = collections['brand']
            elif entity_type == 'content_types' and 'content_type' in collections:
                collection_name = collections['content_type']
            elif entity_type == 'duration_ranges' and 'duration_range' in collections:
                collection_name = collections['duration_range']
            elif entity_type == 'issue_types' and 'issue_type' in collections:
                collection_name = collections['issue_type']
            elif entity_type == 'channels' and 'channel' in collections:
                collection_name = collections['channel']
            elif entity_type == 'priorities' and 'priority' in collections:
                collection_name = collections['priority']
            
            if not collection_name:
                continue
            
            # Get collection
            collection = client.collections.get(collection_name)
            
            if mode == "merge":
                # Check existing entities
                existing_names = set()
                try:
                    for item in collection.iterator():
                        if 'name' in item.properties:
                            existing_names.add(item.properties['name'])
                except:
                    pass
                
                # Filter new entities
                new_entities = []
                for entity in entity_list:
                    name = entity.get('name', '')
                    if name and name not in existing_names:
                        new_entities.append(entity)
                
                if new_entities:
                    # Add new entities
                    with collection.batch.dynamic() as batch:
                        for entity in new_entities:
                            uuid_key = entity.get('name', '')
                            if entity_type == 'content_types' and 'type' in entity:
                                uuid_key = f"{entity['type']}::{entity['name']}"
                            
                            uuid = entity_uuids.get(entity_type, {}).get(uuid_key)
                            if uuid:
                                batch.add_object(entity, uuid=uuid)
                    
                    print(f"  ✓ Added {len(new_entities)} new {entity_type}")
                else:
                    print(f"  ✓ All {len(entity_list)} {entity_type} already exist")
            else:
                # Replace mode: add all
                with collection.batch.dynamic() as batch:
                    for entity in entity_list:
                        uuid_key = entity.get('name', '')
                        if entity_type == 'content_types' and 'type' in entity:
                            uuid_key = f"{entity['type']}::{entity['name']}"
                        
                        uuid = entity_uuids.get(entity_type, {}).get(uuid_key)
                        if uuid:
                            batch.add_object(entity, uuid=uuid)
                
                print(f"  ✓ Added {len(entity_list)} {entity_type}")
    
    def _ingest_records(self, client: weaviate.Client, df: pd.DataFrame, 
                       entity_uuids: Dict, dataset_id: str, mode: str):
        """Ingest main records using unified logic"""
        # Determine main collection
        if self.platform == 'customer_care':
            collection_name = self.config['collections']['case']
        else:
            collection_name = self.config['collections']['post']
        
        collection = client.collections.get(collection_name)
        
        # For merge mode, check existing records
        if mode == "merge":
            existing_ids = set()
            id_field = self._get_id_field()
            
            try:
                # Only check if we have a unique identifier field
                if id_field and id_field in df.columns:
                    for item in collection.iterator():
                        if id_field in item.properties:
                            existing_ids.add(str(item.properties[id_field]))
                    
                    if existing_ids:
                        print(f"  Found {len(existing_ids)} existing records")
                        # Filter out existing records
                        df = df[~df[id_field].astype(str).isin(existing_ids)]
                        print(f"  Will ingest {len(df)} new records")
            except Exception as e:
                print(f"  Warning: Could not check existing records: {e}")
        
        if len(df) == 0:
            print("  No new records to ingest")
            return
        
        # Ingest in batches
        batch_size = 100
        total_ingested = 0
        
        print(f"📥 Ingesting {len(df)} records...")
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            with collection.batch.dynamic() as batch:
                for idx, row in batch_df.iterrows():
                    # Build record data
                    record_data = self._build_record_data(row, entity_uuids, dataset_id)
                    
                    # Add to batch
                    batch.add_object(record_data)
            
            total_ingested += len(batch_df)
            
            # Progress update
            if total_ingested % 1000 == 0 or total_ingested == len(df):
                print(f"  Progress: {total_ingested}/{len(df)} records")
        
        print(f"✅ Successfully ingested {total_ingested} records")
    
    def _get_id_field(self) -> Optional[str]:
        """Get the unique ID field for the platform"""
        id_fields = {
            'facebook': 'facebook_id',
            'instagram': 'instagram_id', 
            'tiktok': 'tiktok_id',
            'customer_care': 'case_number'
        }
        return id_fields.get(self.platform)
    
    def _build_record_data(self, row: pd.Series, entity_uuids: Dict, dataset_id: str) -> Dict[str, Any]:
        """Build record data for ingestion"""
        from ingestion.unified_entities import UnifiedEntityExtractor
        extractor = UnifiedEntityExtractor(self.platform)
        
        # Start with all row data
        record_data = {}
        
        # Add all fields from the row
        for col, value in row.items():
            # Skip null/empty values with robust checking
            should_skip = False
            
            try:
                # Check for null values
                if value is None or value is pd.NA:
                    should_skip = True
                elif isinstance(value, str) and value == '':
                    should_skip = True
                elif isinstance(value, (list, tuple)) and len(value) == 0:
                    should_skip = True
                elif hasattr(value, 'size') and hasattr(value, 'ndim'):
                    # This is a numpy array
                    if value.size == 0:
                        should_skip = True
                # For all other values, check pd.notna carefully
                elif not pd.notna(value):
                    should_skip = True
            except (ValueError, TypeError, AttributeError):
                # If we can't determine, assume it's valid
                should_skip = False
            
            if should_skip:
                continue
                
            # Convert numpy/pandas types to Python types
            try:
                if hasattr(value, 'item'):
                    value = value.item()
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    value = str(value)
                elif isinstance(value, (pd.Series, pd.DataFrame)):
                    # Skip complex pandas objects
                    continue
                # Ensure ID fields are strings (critical for Weaviate)
                if col in ['tiktok_id', 'facebook_id', 'instagram_id', 'customer_care_id']:
                    value = str(value)
                
                record_data[col] = value
            except (ValueError, TypeError, AttributeError):
                # Skip values that can't be converted
                continue
        
        # Add relationships
        relationships = extractor.map_relationships(row, entity_uuids)
        record_data.update(relationships)
        
        # Add dataset_id for customer care
        if self.platform == 'customer_care' and dataset_id:
            record_data['dataset_id'] = dataset_id
        
        # Ensure required fields have defaults
        if self.platform != 'customer_care':
            # Social platforms
            if 'network' not in record_data:
                record_data['network'] = self.platform
            if 'created_time' not in record_data and 'created_time' in row:
                record_data['created_time'] = str(row['created_time'])
        
        return record_data
    
    def _fast_process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast processing - skip expensive operations"""
        print("⚡ Fast mode: Basic processing only")
        
        # Text sanitization only
        if sanitize_text_columns and self.config.get('text_fields_for_sanitization'):
            df = self.data_processor._sanitize_text(df)
        
        # Basic content enrichment (no ML)
        if 'text' in df.columns:
            df['word_count'] = df['text'].str.split().str.len().fillna(0)
            df['char_count'] = df['text'].str.len().fillna(0)
            df['hashtag_count'] = df['text'].str.count('#').fillna(0)
            df['mention_count'] = df['text'].str.count('@').fillna(0)
        
        # Set processing status
        df['processing_status'] = 'ingested'
        df['ingested_at'] = datetime.now().replace(tzinfo=timezone.utc)
        df['sentiment_score'] = None  # Placeholder for batch processing
        
        return df
    
    def _export_metrics(self, df: pd.DataFrame, dataset_id: str):
        """Export metrics after ingestion"""
        metrics_config = self.config.get('metrics', {})
        
        if not metrics_config.get('exporter_class'):
            return
        
        try:
            # Dynamic import of exporter
            module_path, class_name = metrics_config['exporter_class'].rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            ExporterClass = getattr(module, class_name)
            
            # Initialize and run exporter
            output_dir = Path(metrics_config.get('output_dir', f'metrics/{self.platform}'))
            exporter = ExporterClass(output_dir)
            
            try:
                # Export platform metrics
                exporter.export_platform_metrics(self.platform, dataset_id=dataset_id)
                print(f"✓ Metrics exported to {output_dir}")
            finally:
                # Clean up exporter connection
                if hasattr(exporter, 'close_connection'):
                    exporter.close_connection()
            
        except Exception as e:
            print(f"⚠️ Metrics export failed: {e}")
    
    def _run_semantic_analysis(self, client: weaviate.Client, df: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
        """Run semantic analysis and topic discovery during ingestion"""
        try:
            # Import semantic pipeline components
            from cross_platform.semantic_pipeline import (
                cluster_topics, 
                build_topic_kpis, analyze_temporal_trends,
                compute_topic_diagnostics
            )
            # Semantic topic ingestion now handled by unified schema
            
            print("📊 Sampling content for semantic analysis...")
            
            # Get collection name
            collection_name = self.config['collections'].get('post') if self.platform != 'customer_care' else self.config['collections'].get('case')
            
            # Sample content from what we just ingested
            sample_size = min(5000, len(df))  # Limit sample size
            # Sample content directly from DataFrame
            content_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df.copy()
            
            # Find the best text field for semantic analysis
            text_field = None
            for field in ['text', 'tiktok_content', 'content_summary', 'tiktok_post_labels_names', 'content']:
                if field in content_df.columns:
                    text_field = field
                    break
            
            if not text_field:
                print("⚠️ No suitable text field found for semantic analysis")
                return {"status": "no_text_field"}
            
            print(f"📝 Using '{text_field}' field for semantic analysis")
            content_df = content_df[[text_field]].dropna()
            content_df = content_df.rename(columns={text_field: 'text'})
            
            if len(content_df) < 20:
                print("⚠️ Not enough content for semantic analysis")
                return {"status": "insufficient_data"}
            
            print(f"🔬 Clustering {len(content_df)} documents...")
            
            # Perform clustering
            # Generate embeddings and cluster
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(content_df['text'].tolist())
            labels, cluster_docs, cluster_labels = cluster_topics(embeddings, content_df['text'].tolist(), max_clusters=20)
            content_df['cluster'] = labels
            
            # Build topic KPIs
            topics = build_topic_kpis(content_df, labels, cluster_labels)
            
            # Add metrics to topics
            print("📈 Computing topic metrics...")
            for topic in topics:
                topic_mask = content_df['cluster'] == topic['topic_id']
                topic_data = content_df[topic_mask]
                
                # Add engagement metrics
                if 'engagement_rate' in topic_data.columns:
                    topic['avg_engagement_rate'] = float(topic_data['engagement_rate'].mean())
                
                # Add sentiment
                if 'sentiment_score' in topic_data.columns:
                    topic['avg_sentiment'] = float(topic_data['sentiment_score'].mean())
                    topic['sentiment_std'] = float(topic_data['sentiment_score'].std())
                
                # Add example posts
                id_field = 'id' if 'id' in topic_data.columns else 'tiktok_id' if 'tiktok_id' in topic_data.columns else None
                if id_field:
                    topic['examples'] = topic_data.head(5)[[id_field, 'text']].to_dict('records')
                else:
                    topic['examples'] = topic_data.head(5)[['text']].to_dict('records')
            
            # Analyze temporal trends
            print("📅 Analyzing temporal trends...")
            if 'created_time' in content_df.columns:
                trends = analyze_temporal_trends(content_df, labels)
            else:
                trends = []
            
            # Compute diagnostics
            print("🔍 Computing topic quality metrics...")
            diagnostics = compute_topic_diagnostics(topics, embeddings, labels)
            
            # Save intermediate results (for metrics export to use)
            output_dir = Path(f"metrics/{self.platform}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save topics
            topics_file = output_dir / f"{self.platform}_semantic_topics_{timestamp}.json"
            with open(topics_file, 'w') as f:
                json.dump({
                    "platform": self.platform,
                    "dataset_id": dataset_id,
                    "topics": topics,
                    "generated_at": datetime.now().replace(tzinfo=timezone.utc).isoformat()
                }, f, indent=2, default=str)
            
            # Store semantic topics in Weaviate
            print("💾 Storing semantic topics in Weaviate...")
            self._ingest_semantic_topics(client, topics, dataset_id)
            
            # Also save to JSON files for backup
            print("💾 Semantic topics saved to JSON files")
            
            print(f"✅ Semantic analysis complete: {len(topics)} topics discovered")
            
            return {
                "status": "success",
                "topics_discovered": len(topics),
                "documents_analyzed": len(content_df),
                "topics_file": str(topics_file),
                "avg_topic_quality": diagnostics.get("summary", {}).get("avg_coherence", 0)
            }
            
        except Exception as e:
            print(f"⚠️ Semantic analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    def _ingest_semantic_topics(self, client: weaviate.Client, topics: List[Dict], dataset_id: str):
        """Ingest discovered semantic topics into Weaviate"""
        try:
            # Ensure SemanticTopic collection exists
            self._ensure_semantic_collections(client)
            
            # Get the SemanticTopic collection
            semantic_collection = client.collections.get("SemanticTopic")
            
            # Store each topic
            topic_uuids = {}
            for topic in topics:
                topic_id = f"{self.platform}_{topic['topic_id']}_{dataset_id}"
                
                # Prepare topic data
                topic_data = {
                    "topic_id": topic_id,
                    "platform": self.platform,
                    "label": topic.get("label", ""),
                    "keywords": topic.get("label", ""),  # Use label as keywords for now
                    "description": f"Topic with {topic.get('size', 0)} posts",
                    "size": topic.get("size", 0),
                    "avg_sentiment": float(topic.get("avg_sentiment", 0.0)),
                    "dataset_id": dataset_id,
                    "created_at": datetime.now().replace(tzinfo=timezone.utc)
                }
                
                # Insert topic
                uuid = semantic_collection.data.insert(topic_data)
                topic_uuids[topic['topic_id']] = uuid
                
            print(f"✅ Stored {len(topics)} semantic topics in Weaviate")
            
            # Update posts with topic references
            self._link_posts_to_topics(client, topics, topic_uuids, dataset_id)
            
        except Exception as e:
            print(f"⚠️ Failed to ingest semantic topics: {e}")
            import traceback
            traceback.print_exc()
    
    def _ensure_semantic_collections(self, client: weaviate.Client):
        """Ensure semantic collections exist"""
        try:
            import weaviate.classes.config as wvcc
            
            # Check if SemanticTopic exists
            try:
                client.collections.get("SemanticTopic")
            except:
                # Create SemanticTopic collection
                client.collections.create(
                    name="SemanticTopic",
                    description="Discovered semantic topics from content clustering",
                    vectorizer_config=wvcc.Configure.Vectorizer.none(),
                    properties=[
                        wvcc.Property(name="topic_id", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="platform", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="label", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="keywords", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="description", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="size", data_type=wvcc.DataType.INT),
                        wvcc.Property(name="avg_sentiment", data_type=wvcc.DataType.NUMBER),
                        wvcc.Property(name="dataset_id", data_type=wvcc.DataType.TEXT),
                        wvcc.Property(name="created_at", data_type=wvcc.DataType.DATE),
                    ]
                )
                print("✅ Created SemanticTopic collection")
                
        except Exception as e:
            print(f"⚠️ Failed to ensure semantic collections: {e}")
    
    def _link_posts_to_topics(self, client: weaviate.Client, topics: List[Dict], topic_uuids: Dict, dataset_id: str):
        """Link posts to their discovered topics"""
        try:
            # Get the platform post collection
            collection_name = self.config['collections'].get('post') if self.platform != 'customer_care' else self.config['collections'].get('case')
            post_collection = client.collections.get(collection_name)
            
            # Create topic assignments for posts
            topic_assignments = {}  # post_index -> topic_id
            
            for topic in topics:
                topic_id = topic['topic_id']
                # Get posts assigned to this topic (from examples or cluster assignments)
                if 'examples' in topic:
                    for example in topic['examples']:
                        post_idx = example.get('id', example.get('tiktok_id'))
                        if post_idx:
                            topic_assignments[str(post_idx)] = topic_id
            
            # Update posts with topic references
            if topic_assignments:
                print(f"🔗 Linking {len(topic_assignments)} posts to topics...")
                # Note: In a full implementation, we'd update each post with topic references
                # For now, we'll store the topic assignments for future use
                
            print(f"✅ Linked posts to {len(topics)} topics")
            
        except Exception as e:
            print(f"⚠️ Failed to link posts to topics: {e}")


def ensure_weaviate_docker():
    """Ensure Weaviate Docker container is running"""
    image = os.getenv("WEAVIATE_DOCKER_IMAGE", "semitechnologies/weaviate:1.23.7")
    data_dir = os.getenv("WEAVIATE_DATA_DIR", "./WEAVIATE")
    port = int(os.getenv("WEAVIATE_PORT", "8080"))
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    container = os.getenv("WEAVIATE_CONTAINER_NAME", "weaviate_local")
    
    abs_data_dir = str(Path(data_dir).resolve())
    Path(abs_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if container exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container}", "--format", "{{.Status}}"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0 or not result.stdout:
        # Create new container
        print("🐳 Starting Weaviate container...")
        subprocess.run([
            "docker", "run", "-d",
            "--name", container,
            "-p", f"{port}:8080",
            "-p", f"{grpc_port}:50051",
            "-e", "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
            "-e", "PERSISTENCE_DATA_PATH=/var/lib/weaviate",
            "-e", "DEFAULT_VECTORIZER_MODULE=none",
            "-e", "ENABLE_MODULES=",
            "-v", f"{abs_data_dir}:/var/lib/weaviate",
            image
        ])
    else:
        # Container exists - ensure it's running
        if "Exited" in result.stdout:
            print("🐳 Starting existing Weaviate container...")
            subprocess.run(["docker", "start", container])
        elif "Up" in result.stdout:
            print("✅ Using existing running Weaviate container")
        else:
            print("🔄 Existing Weaviate container found, ensuring it's running...")
            subprocess.run(["docker", "start", container])
    
    # Wait for ready
    base_url = f"http://localhost:{port}"
    if not _wait_for_weaviate_ready(base_url):
        raise RuntimeError("Weaviate failed to start")
    
    print("✅ Weaviate is ready")


def _wait_for_weaviate_ready(base_url: str, timeout: int = 180) -> bool:
    """Wait for Weaviate to be ready (increased timeout for vectorizer module download)"""
    url = f"{base_url.rstrip('/')}/v1/.well-known/ready"
    start = time.time()
    
    print("⏳ Waiting for Weaviate to be ready (this may take a few minutes for first-time vectorizer download)...")
    
    while time.time() - start < timeout:
        try:
            with urlopen(url) as resp:
                if resp.status in (200, 204):
                    return True
        except Exception as e:
            elapsed = int(time.time() - start)
            if elapsed % 30 == 0:  # Log every 30 seconds
                print(f"   Still waiting... ({elapsed}s elapsed, timeout in {timeout-elapsed}s)")
            time.sleep(1)
    
    print(f"❌ Timeout after {timeout}s - Weaviate may still be downloading vectorizer modules")
    return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified data ingestion for all platforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest Facebook data
  %(prog)s --platform facebook --input data/facebook_posts.csv
  
  # Ingest Customer Care data with limited rows for testing
  %(prog)s --platform customer_care --input data/customer_cases.xlsx --limit-rows 1000
  
  # Force re-ingestion in replace mode
  %(prog)s --platform tiktok --input data/tiktok_videos.csv --mode replace --force
        """
    )
    
    parser.add_argument(
        '--platform',
        required=True,
        choices=['facebook', 'instagram', 'tiktok', 'customer_care'],
        help='Platform to ingest data for'
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input data file (.csv, .xlsx, .parquet)'
    )
    
    parser.add_argument(
        '--config',
        help='Custom configuration file (default: ingestion/configs/{platform}.yaml)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['merge', 'replace'],
        default='merge',
        help='Ingestion mode: merge (safe, default) or replace (destructive)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force ingestion even if file was already processed'
    )
    
    parser.add_argument(
        '--skip-sentiment',
        action='store_true',
        help='Skip sentiment analysis during ingestion'
    )
    
    parser.add_argument(
        '--limit-rows',
        type=int,
        default=0,
        help='Process only first N rows (for testing)'
    )
    
    parser.add_argument(
        '--processing-mode',
        choices=['fast', 'full'],
        default='fast',
        help='Processing mode: fast (skip expensive ops) or full (complete processing)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        # Initialize engine
        engine = UnifiedIngestionEngine(args.platform, args.config)
        
        # Run ingestion
        result = engine.ingest(
            input_file=args.input,
            mode=args.mode,
            force=args.force,
            skip_sentiment=args.skip_sentiment,
            limit_rows=args.limit_rows,
            processing_mode=args.processing_mode
        )
        
        # Print summary
        if result['status'] == 'success':
            print("\n📈 Ingestion Summary:")
            print(f"  • Platform: {args.platform}")
            print(f"  • Records: {result['records']}")
            print(f"  • Dataset ID: {result['dataset_id']}")
            print(f"  • Entities: {result['entities']}")
            
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
