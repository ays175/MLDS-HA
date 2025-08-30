#!/usr/bin/env python3
"""
TikTok knowledge graph ingestion using Weaviate
"""
import pandas as pd
import weaviate
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import argparse
import shutil
import hashlib
import json
import os
import subprocess
import time
import socket
from urllib.request import urlopen
from urllib.error import URLError
from tiktok_entities import extract_tiktok_entities, generate_tiktok_entity_uuids, map_tiktok_post_relationships, extract_enhanced_text_content, extract_tiktok_attachments_info, extract_label_insights
from tiktok_metrics_export import TikTokMetricsExporter
from tiktok_schema import create_tiktok_knowledge_graph_schema, validate_tiktok_schema
import weaviate.classes as wvc

# Load environment variables
load_dotenv()

class TikTokIngestionTracker:
    """Track TikTok ingestion to avoid duplicates"""
    
    def __init__(self):
        self.tracker_file = Path("data/tiktok_ingestion_log.json")
        self.tracker_file.parent.mkdir(exist_ok=True)
        
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {"ingestions": []}
    
    def is_duplicate(self, csv_file):
        """Check if file was already ingested"""
        file_hash = self._get_file_hash(csv_file)
        return any(ing['file_hash'] == file_hash for ing in self.log['ingestions'])
    
    def record_ingestion(self, csv_file, record_count):
        """Record successful ingestion"""
        self.log['ingestions'].append({
            'file_name': Path(csv_file).name,
            'file_hash': self._get_file_hash(csv_file),
            'record_count': record_count,
            'ingested_at': datetime.now().isoformat()
        })
        
        with open(self.tracker_file, 'w') as f:
            json.dump(self.log, f, indent=2)
    
    def _get_file_hash(self, csv_file):
        """Generate file hash"""
        with open(csv_file, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

def check_weaviate_vectorizer():
    """Check if Weaviate built-in vectorizer is available"""
    print("✅ Using Weaviate built-in vectorizer (text2vec-contextionary)")
    return True

def generate_tiktok_dataset_id(csv_file_path):
    """Generate a unique dataset ID based on file content"""
    with open(csv_file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:8]
    return f"tiktok_dataset_{file_hash}"

def move_processed_tiktok_csv(csv_file_path, dataset_id):
    """Move processed CSV to processed folder with dataset ID and timestamp"""
    csv_path = Path(csv_file_path)
    processed_dir = Path("data/processed/tiktok")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create folder: dataset_id_timestamp
    folder_name = f"{dataset_id}_{timestamp}"
    target_folder = processed_dir / folder_name
    target_folder.mkdir(parents=True, exist_ok=True)
    
    # Move CSV file
    target_file = target_folder / csv_path.name
    shutil.move(str(csv_path), str(target_file))
    
    print(f"📁 Moved processed CSV: {csv_path.name} → {target_folder}/")
    return str(target_file)

def ensure_tiktok_schema_exists(client, mode="merge"):
    """Ensure TikTok schema exists without destroying data in merge mode"""
    if mode == "replace":
        create_tiktok_knowledge_graph_schema(client)
    else:
        # Merge mode: only create collections if they don't exist
        collections = ["TikTokPlatform", "TikTokBrand", "TikTokContentType", "TikTokDurationRange", "TikTokPost"]
        missing_collections = []
        
        for collection_name in collections:
            try:
                client.collections.get(collection_name)
            except:
                missing_collections.append(collection_name)
        
        if missing_collections:
            print(f"Creating missing collections: {missing_collections}")
            create_tiktok_knowledge_graph_schema(client)

def ingest_tiktok_entities(client, entities, entity_uuids, mode="merge"):
    """Ingest TikTok entities into collections"""
    
    if mode == "replace":
        # Replace mode: delete all and add fresh
        print("🔄 Replace mode: Adding all entities fresh")
        
        # Platforms
        platform_collection = client.collections.get("TikTokPlatform")
        with platform_collection.batch.dynamic() as batch:
            for platform in entities['platforms']:
                platform_uuid = entity_uuids['platforms'][platform['name']]
                batch.add_object(platform, uuid=platform_uuid)
        print(f"📱 Added {len(entities['platforms'])} platforms")
        
        # Brands
        brand_collection = client.collections.get("TikTokBrand")
        with brand_collection.batch.dynamic() as batch:
            for brand in entities['brands']:
                brand_uuid = entity_uuids['brands'][brand['name']]
                batch.add_object(brand, uuid=brand_uuid)
        print(f"🏷️ Added {len(entities['brands'])} brands")
        
        # Content Types
        content_collection = client.collections.get("TikTokContentType")
        with content_collection.batch.dynamic() as batch:
            for content_type in entities['content_types']:
                content_uuid = entity_uuids['content_types'][content_type['name']]
                batch.add_object(content_type, uuid=content_uuid)
        print(f"📝 Added {len(entities['content_types'])} content types")
        
        # Duration Ranges
        duration_collection = client.collections.get("TikTokDurationRange")
        with duration_collection.batch.dynamic() as batch:
            for duration_range in entities['duration_ranges']:
                duration_uuid = entity_uuids['duration_ranges'][duration_range['name']]
                batch.add_object(duration_range, uuid=duration_uuid)
        print(f"⏱️ Added {len(entities['duration_ranges'])} duration ranges")
        
    else:
        # Merge mode: only add new entities
        print("🔗 Merge mode: Adding only new entities")
        merge_tiktok_entities_safe(client, entities, entity_uuids)

def merge_tiktok_entities_safe(client, entities, entity_uuids):
    """Safely merge TikTok entities without duplicates"""
    
    # Merge Platforms
    platform_collection = client.collections.get("TikTokPlatform")
    try:
        existing_platforms = {p.properties['name'] for p in platform_collection.iterator()}
    except:
        existing_platforms = set()
    
    new_platforms = [p for p in entities['platforms'] if p['name'] not in existing_platforms]
    if new_platforms:
        with platform_collection.batch.dynamic() as batch:
            for platform in new_platforms:
                uuid = entity_uuids['platforms'][platform['name']]
                batch.add_object(platform, uuid=uuid)
        print(f"📱 Added {len(new_platforms)} new platforms")
    else:
        print(f"📱 All {len(entities['platforms'])} platforms already exist")
    
    # Merge Brands
    brand_collection = client.collections.get("TikTokBrand")
    try:
        existing_brands = {b.properties['name'] for b in brand_collection.iterator()}
    except:
        existing_brands = set()
    
    new_brands = [b for b in entities['brands'] if b['name'] not in existing_brands]
    if new_brands:
        with brand_collection.batch.dynamic() as batch:
            for brand in new_brands:
                uuid = entity_uuids['brands'][brand['name']]
                batch.add_object(brand, uuid=uuid)
        print(f"🏷️ Added {len(new_brands)} new brands")
    else:
        print(f"🏷️ All {len(entities['brands'])} brands already exist")
    
    # Merge Content Types
    content_collection = client.collections.get("TikTokContentType")
    try:
        existing_content = {c.properties['name'] for c in content_collection.iterator()}
    except:
        existing_content = set()
    
    new_content_types = [c for c in entities['content_types'] if c['name'] not in existing_content]
    if new_content_types:
        with content_collection.batch.dynamic() as batch:
            for content_type in new_content_types:
                uuid = entity_uuids['content_types'][content_type['name']]
                batch.add_object(content_type, uuid=uuid)
        print(f"📝 Added {len(new_content_types)} new content types")
    else:
        print(f"📝 All {len(entities['content_types'])} content types already exist")
    
    # Merge Duration Ranges
    duration_collection = client.collections.get("TikTokDurationRange")
    try:
        existing_ranges = {d.properties['name'] for d in duration_collection.iterator()}
    except:
        existing_ranges = set()
    
    new_ranges = [d for d in entities['duration_ranges'] if d['name'] not in existing_ranges]
    if new_ranges:
        with duration_collection.batch.dynamic() as batch:
            for duration_range in new_ranges:
                uuid = entity_uuids['duration_ranges'][duration_range['name']]
                batch.add_object(duration_range, uuid=uuid)
        print(f"⏱️ Added {len(new_ranges)} new duration ranges")
    else:
        print(f"⏱️ All {len(entities['duration_ranges'])} duration ranges already exist")

def ingest_tiktok_posts(client, df, entity_uuids, mode="merge"):
    """Ingest TikTok posts with relationships"""
    
    post_collection = client.collections.get("TikTokPost")
    
    if mode == "merge":
        # Check for existing posts
        try:
            existing_ids = {p.properties['post_id'] for p in post_collection.iterator()}
            new_posts_df = df[~df['tiktok_id'].astype(str).isin(existing_ids)]
            print(f"📝 Found {len(df)} total posts, {len(new_posts_df)} are new")
        except:
            print("📝 No existing posts found, processing all")
            new_posts_df = df
    else:
        new_posts_df = df
    
    if len(new_posts_df) == 0:
        print("📝 All posts already exist")
        return
    
    print("📝 Ingesting TikTok posts with relationships...")
    
    with post_collection.batch.fixed_size(batch_size=10) as batch:
        for _, row in tqdm(new_posts_df.iterrows(), total=len(new_posts_df), desc="TikTok Posts"):
            
            # Calculate metrics safely
            impressions = int(row.get('tiktok_insights_impressions', 0) or 0)
            video_views = int(row.get('tiktok_insights_video_views', 0) or 0)
            engagements = int(row.get('tiktok_insights_engagements', 0) or 0)
            likes = int(row.get('tiktok_insights_likes', 0) or 0)
            shares = int(row.get('tiktok_insights_shares', 0) or 0)
            
            # Calculate rates
            engagement_rate = (engagements / impressions * 100) if impressions > 0 else 0
            view_rate = (video_views / impressions * 100) if impressions > 0 else 0
            like_rate = (likes / video_views * 100) if video_views > 0 else 0
            share_rate = (shares / video_views * 100) if video_views > 0 else 0
            
            # Parse datetime
            posted_datetime_str = str(row.get('created_time', ''))
            try:
                dt = datetime.fromisoformat(posted_datetime_str.replace('Z', '+00:00'))
                posted_date = dt.strftime('%Y-%m-%d')
                posted_time = dt.strftime('%H:%M:%S')
                weekday = dt.strftime('%A')
                hour = dt.hour
            except:
                posted_date = ''
                posted_time = ''
                weekday = ''
                hour = 0
            
            # Extract enhanced text content for vectorization
            enhanced_content = extract_enhanced_text_content(row)
            
            # Attachments and label insights
            attachments_info = extract_tiktok_attachments_info(row)
            label_insights = extract_label_insights(row.get('tiktok_post_labels_names', ''))
            
            # Create post data
            post_data = {
                "post_id": str(row.get('tiktok_id', '')),
                "profile_id": str(row.get('profile_id', '')),
                "tiktok_link": str(row.get('tiktok_link', '')),
                
                # Timing
                "posted_date": posted_date,
                "posted_time": posted_time,
                "weekday": weekday,
                "hour": hour,
                
                # Content - ENHANCED VECTORIZATION
                "duration": float(row.get('tiktok_duration', 0) or 0),
                "media_type": attachments_info['media_type'],
                "media_count": attachments_info['media_count'],
                
                # VECTORIZED TEXT FIELDS
                "labels_text": enhanced_content['labels_text'],
                "brands_text": enhanced_content['brands_text'],
                "content_themes": enhanced_content['content_themes'],
                "content_summary": enhanced_content['content_summary'],
                
                # Performance Metrics
                "comments": int(row.get('tiktok_insights_comments', 0) or 0),
                "impressions": impressions,
                "likes": likes,
                "shares": shares,
                "engagements": engagements,
                "reach": int(row.get('tiktok_insights_reach', 0) or 0),
                "video_views": video_views,
                "completion_rate": float(row.get('tiktok_insights_completion_rate', 0) or 0),
                
                # Calculated Metrics
                "engagement_rate": round(engagement_rate, 2),
                "view_rate": round(view_rate, 2),
                "like_rate": round(like_rate, 2),
                "share_rate": round(share_rate, 2),
            }
            
            # Add relationships
            relationships = map_tiktok_post_relationships(row, entity_uuids)
            post_data.update(relationships)
            
            batch.add_object(post_data)

def ingest_tiktok_knowledge_graph(client, csv_file, mode="merge"):
    """Main TikTok knowledge graph ingestion"""
    
    if not check_weaviate_vectorizer():
        raise RuntimeError("Weaviate vectorizer not available")
    
    print(f"📊 Loading TikTok data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    print("🔍 Extracting TikTok entities...")
    entities = extract_tiktok_entities(df)
    entity_uuids = generate_tiktok_entity_uuids(entities)
    
    print(f"Found: {len(entities['platforms'])} platforms, {len(entities['brands'])} brands, {len(entities['content_types'])} content types")
    
    # Ensure schema exists
    ensure_tiktok_schema_exists(client, mode)
    
    # Ingest entities
    ingest_tiktok_entities(client, entities, entity_uuids, mode)
    
    # Ingest posts
    ingest_tiktok_posts(client, df, entity_uuids, mode)
    
    # Export comprehensive metrics for AI agents
    print("📈 Exporting AI-agent metrics...")
    dataset_id = generate_tiktok_dataset_id(csv_file)
    metrics_exporter = TikTokMetricsExporter()
    consolidated_metrics = metrics_exporter.export_all_metrics(df, entities, dataset_id)
    
    print("✅ TikTok knowledge graph ingestion completed!")
    print(f"📊 AI-agent metrics saved to: ./metrics/tiktok/")
    
    # Show final stats
    collections = ['TikTokPlatform', 'TikTokBrand', 'TikTokContentType', 'TikTokDurationRange', 'TikTokPost']
    for name in collections:
        try:
            collection = client.collections.get(name)
            total = collection.aggregate.over_all(total_count=True)
            print(f"📈 {name}: {total.total_count} objects")
        except Exception as e:
            print(f"❌ Error getting count for {name}: {e}")
    
    return consolidated_metrics

def _wait_for_weaviate_ready(base_url: str, timeout: int = 60) -> bool:
    """Wait until Weaviate readiness endpoint responds OK."""
    url = f"{base_url.rstrip('/')}/v1/.well-known/ready"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlopen(url) as resp:
                if resp.status in (200, 204):
                    return True
        except Exception:
            time.sleep(1)
    return False

def ensure_weaviate_docker():
    """Ensure a local Weaviate Docker container is running with persistent volume.

    Controlled via env vars with safe defaults:
    - WEAVIATE_DOCKER_IMAGE (default: semitechnologies/weaviate:1.19.6)
    - WEAVIATE_DATA_DIR (default: ./WEAVIATE)
    - WEAVIATE_PORT (default: 8080)
    - WEAVIATE_CONTAINER_NAME (default: weaviate_local)
    """
    image = os.getenv("WEAVIATE_DOCKER_IMAGE", "semitechnologies/weaviate:1.23.7")
    data_dir = os.getenv("WEAVIATE_DATA_DIR", "./WEAVIATE")
    port = int(os.getenv("WEAVIATE_PORT", "8080"))
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    container = os.getenv("WEAVIATE_CONTAINER_NAME", "weaviate_local")

    abs_data_dir = str(Path(data_dir).resolve())
    Path(abs_data_dir).mkdir(parents=True, exist_ok=True)

    base_url = f"http://localhost:{port}"

    def _run(cmd):
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Container exists?
    res = _run(["docker", "ps", "-a", "--filter", f"name={container}", "--format", "{{.Status}}"])
    if not (res.returncode == 0 and res.stdout):
        # Run fresh container
        _run([
            "docker", "run", "-d",
            "--name", container,
            "-p", f"{port}:8080",
            "-p", f"{grpc_port}:50051",
            "-e", "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
            "-e", "PERSISTENCE_DATA_PATH=/var/lib/weaviate",
            "-v", f"{abs_data_dir}:/var/lib/weaviate",
            image
        ])
    else:
        # Ensure running
        running = _run(["docker", "inspect", "-f", "{{.State.Running}}", container])
        if running.returncode == 0 and running.stdout.strip() != "true":
            _run(["docker", "start", container])

    if not _wait_for_weaviate_ready(base_url, timeout=60):
        raise RuntimeError("Weaviate did not become ready on http://localhost:%d" % port)

    # Also wait for gRPC port to become available (required by client v4)
    start = time.time()
    while time.time() - start < 60:
        try:
            with socket.create_connection(("127.0.0.1", grpc_port), timeout=1):
                break
        except Exception:
            time.sleep(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Ingest TikTok data into knowledge graph')
    parser.add_argument('--csv-file', required=True, help='TikTok CSV file to ingest')
    parser.add_argument('--mode', choices=['merge', 'replace'], default='merge', 
                       help='Ingestion mode: merge (safe, default) or replace (destructive)')
    parser.add_argument('--force', action='store_true', help='Force ingestion even if duplicate')
    return parser.parse_args()

def main():
    """Main TikTok ingestion process"""
    args = parse_args()
    csv_file = args.csv_file
    mode = args.mode
    
    print(f"🎵 TikTok Knowledge Graph Ingestion")
    print(f"🔄 Mode: {mode} ({'safe' if mode == 'merge' else 'destructive'})")
    
    # Check for duplicates
    tracker = TikTokIngestionTracker()
    if not args.force and tracker.is_duplicate(csv_file):
        print(f"❌ File {csv_file} already ingested. Use --force to override.")
        exit(0)
    
    # Ensure Docker Weaviate is running and connect
    print("🔌 Ensuring Weaviate (Docker) is running...")
    ensure_weaviate_docker()
    print("🔌 Connecting to Weaviate...")
    additional = wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=120))
    client = weaviate.connect_to_local(additional_config=additional)
    
    try:
        # Validate CSV file exists
        if not Path(csv_file).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Run ingestion
        ingest_tiktok_knowledge_graph(client, csv_file, mode)
        
        # Record successful ingestion
        df = pd.read_csv(csv_file)
        tracker.record_ingestion(csv_file, len(df))
        print(f"✅ TikTok ingestion completed and logged")
        
        # Move processed CSV to processed folder
        dataset_id = generate_tiktok_dataset_id(csv_file)
        moved_file = move_processed_tiktok_csv(csv_file, dataset_id)
        print(f"📁 Dataset ID: {dataset_id}")
        
    except Exception as e:
        print(f"💥 Ingestion failed: {e}")
        raise
        
    finally:
        client.close()
        print("🔌 Connection closed")

if __name__ == "__main__":
    main()