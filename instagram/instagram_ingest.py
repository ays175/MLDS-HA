#!/usr/bin/env python3
"""
Instagram knowledge graph ingestion using Weaviate
"""

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv

from instagram_schema import create_instagram_knowledge_graph_schema
from instagram_metrics_export import InstagramMetricsExporter
from instagram_entities import (
    extract_instagram_entities,
    generate_instagram_entity_uuids,
    map_instagram_post_relationships,
    extract_instagram_enhanced_text,
)


# Load environment variables
load_dotenv()


class InstagramIngestionTracker:
    """Track Instagram ingestion to avoid duplicates"""

    def __init__(self):
        self.tracker_file = Path("data/instagram_ingestion_log.json")
        self.tracker_file.parent.mkdir(exist_ok=True)

        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {"ingestions": []}

    def is_duplicate(self, file_path: str) -> bool:
        file_hash = self._get_file_hash(file_path)
        return any(ing['file_hash'] == file_hash for ing in self.log['ingestions'])

    def record_ingestion(self, file_path: str, record_count: int):
        self.log['ingestions'].append({
            'file_name': Path(file_path).name,
            'file_hash': self._get_file_hash(file_path),
            'record_count': record_count,
            'ingested_at': datetime.now().isoformat()
        })

        with open(self.tracker_file, 'w') as f:
            json.dump(self.log, f, indent=2)

    def _get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]


def check_weaviate_vectorizer() -> bool:
    print("✅ Using Weaviate built-in vectorizer (text2vec-contextionary)")
    return True


def _wait_for_weaviate_ready(base_url: str, timeout: int = 60) -> bool:
    url = f"{base_url.rstrip('/')}/v1/.well-known/ready"
    start = time.time()
    while time.time() - start < timeout:
        try:
            from urllib.request import urlopen
            with urlopen(url) as resp:
                if resp.status in (200, 204):
                    return True
        except Exception:
            time.sleep(1)
    return False


def ensure_weaviate_docker():
    """Ensure a local Weaviate Docker container is running with persistent volume."""
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

    res = _run(["docker", "ps", "-a", "--filter", f"name={container}", "--format", "{{.Status}}"])
    if not (res.returncode == 0 and res.stdout):
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
        running = _run(["docker", "inspect", "-f", "{{.State.Running}}", container])
        if running.returncode == 0 and running.stdout.strip() != "true":
            _run(["docker", "start", container])

    if not _wait_for_weaviate_ready(base_url, timeout=60):
        raise RuntimeError("Weaviate did not become ready on http://localhost:%d" % port)

    start = time.time()
    while time.time() - start < 60:
        try:
            with socket.create_connection(("127.0.0.1", grpc_port), timeout=1):
                break
        except Exception:
            time.sleep(1)


def generate_instagram_dataset_id(file_path: str) -> str:
    """Generate a unique dataset ID based on file content"""
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:8]
    return f"instagram_dataset_{file_hash}"


def move_processed_instagram_file(file_path: str, dataset_id: str) -> str:
    """Move processed source file to processed/instagram folder with dataset ID and timestamp"""
    src_path = Path(file_path)
    processed_dir = Path("data/processed/instagram")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{dataset_id}_{timestamp}"
    target_folder = processed_dir / folder_name
    target_folder.mkdir(parents=True, exist_ok=True)

    target_file = target_folder / src_path.name
    try:
        import shutil
        shutil.move(str(src_path), str(target_file))
        print(f"📁 Moved processed file: {src_path.name} → {target_folder}/")
    except Exception:
        try:
            import shutil
            shutil.copy2(str(src_path), str(target_file))
            print(f"📁 Copied processed file: {src_path.name} → {target_folder}/")
        except Exception:
            print("⚠️ Could not move/copy processed file")
            return str(src_path)
    return str(target_file)


def ensure_instagram_schema_exists(client, mode: str = "merge"):
    if mode == "replace":
        create_instagram_knowledge_graph_schema(client)
    else:
        collections = ["InstagramPlatform", "InstagramBrand", "InstagramContentType", "InstagramPost"]
        missing = []
        for name in collections:
            try:
                client.collections.get(name)
            except Exception:
                missing.append(name)
        if missing:
            print(f"Creating missing collections: {missing}")
            create_instagram_knowledge_graph_schema(client)


def _safe_int(value, default: int = 0) -> int:
    try:
        if value == '' or value is None:
            return default
        if isinstance(value, str) and value.strip().lower() in ('nan', 'na', 'none'):
            return default
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value == '' or value is None:
            return default
        if isinstance(value, str) and value.strip().lower() in ('nan', 'na', 'none'):
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _read_instagram_source(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    if p.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(p)
    # default to CSV
    return pd.read_csv(p)


def ingest_instagram_entities(client, entities: Dict, entity_uuids: Dict, mode: str = "merge"):
    if mode == "replace":
        print("🔄 Replace mode: Adding all entities fresh")

        platform_collection = client.collections.get("InstagramPlatform")
        with platform_collection.batch.dynamic() as batch:
            for platform in entities['platforms']:
                batch.add_object(platform, uuid=entity_uuids['platforms'][platform['name']])
        print(f"📱 Added {len(entities['platforms'])} platforms")

        brand_collection = client.collections.get("InstagramBrand")
        with brand_collection.batch.dynamic() as batch:
            for brand in entities['brands']:
                batch.add_object(brand, uuid=entity_uuids['brands'][brand['name']])
        print(f"🏷️ Added {len(entities['brands'])} brands")

        content_collection = client.collections.get("InstagramContentType")
        with content_collection.batch.dynamic() as batch:
            for ct in entities['content_types']:
                key = f"{ct['type']}::{ct['name']}"
                batch.add_object(ct, uuid=entity_uuids['content_types'][key])
        print(f"📝 Added {len(entities['content_types'])} content types")
        return

    # Merge mode
    print("🔗 Merge mode: Adding only new entities")

    # Platforms
    platform_collection = client.collections.get("InstagramPlatform")
    try:
        existing_platforms = {p.properties['name'] for p in platform_collection.iterator()}
    except Exception:
        existing_platforms = set()
    new_platforms = [p for p in entities['platforms'] if p['name'] not in existing_platforms]
    if new_platforms:
        with platform_collection.batch.dynamic() as batch:
            for platform in new_platforms:
                batch.add_object(platform, uuid=entity_uuids['platforms'][platform['name']])
        print(f"📱 Added {len(new_platforms)} new platforms")
    else:
        print(f"📱 All {len(entities['platforms'])} platforms already exist")

    # Brands
    brand_collection = client.collections.get("InstagramBrand")
    try:
        existing_brands = {b.properties['name'] for b in brand_collection.iterator()}
    except Exception:
        existing_brands = set()
    new_brands = [b for b in entities['brands'] if b['name'] not in existing_brands]
    if new_brands:
        with brand_collection.batch.dynamic() as batch:
            for brand in new_brands:
                batch.add_object(brand, uuid=entity_uuids['brands'][brand['name']])
        print(f"🏷️ Added {len(new_brands)} new brands")
    else:
        print(f"🏷️ All {len(entities['brands'])} brands already exist")

    # Content Types
    content_collection = client.collections.get("InstagramContentType")
    try:
        existing_content = {c.properties['name'] + '::' + c.properties['type'] for c in content_collection.iterator()}
    except Exception:
        existing_content = set()
    new_content = [c for c in entities['content_types'] if (c['name'] + '::' + c['type']) not in existing_content]
    if new_content:
        with content_collection.batch.dynamic() as batch:
            for ct in new_content:
                key = f"{ct['type']}::{ct['name']}"
                batch.add_object(ct, uuid=entity_uuids['content_types'][key])
        print(f"📝 Added {len(new_content)} new content types")
    else:
        print(f"📝 All {len(entities['content_types'])} content types already exist")


def ingest_instagram_posts(client, df: pd.DataFrame, entity_uuids: Dict, mode: str = "merge"):
    post_collection = client.collections.get("InstagramPost")

    new_posts_df = df
    if mode == "merge":
        try:
            existing_ids = {p.properties['instagram_id'] for p in post_collection.iterator()}
            new_posts_df = df[~df['instagram_id'].astype(str).isin(existing_ids)]
            print(f"📝 Found {len(df)} total posts, {len(new_posts_df)} are new")
        except Exception:
            print("📝 No existing posts found, processing all")

    if len(new_posts_df) == 0:
        print("📝 All posts already exist")
        return

    print("📝 Ingesting Instagram posts with relationships...")

    with post_collection.batch.fixed_size(batch_size=10) as batch:
        for _, row in new_posts_df.iterrows():
            impressions = _safe_int(row.get('instagram_insights_impressions'))
            video_views = _safe_int(row.get('instagram_insights_video_views'))
            engagements = _safe_int(row.get('instagram_insights_engagement'))
            interactions = _safe_int(row.get('instagram_interactions'))
            if engagements == 0 and interactions > 0:
                engagements = interactions

            likes = _safe_int(row.get('instagram_likes'))
            comments = _safe_int(row.get('instagram_comments'))
            saves = _safe_int(row.get('instagram_insights_saves'))
            shares = _safe_int(row.get('instagram_shares'))
            clicks = _safe_int(row.get('instagram_insights_post_clicks'))
            reactions_total = _safe_int(row.get('instagram_reactions'))

            engagement_rate = (engagements / impressions * 100) if impressions > 0 else 0
            view_rate = (video_views / impressions * 100) if impressions > 0 else 0
            like_rate = (likes / impressions * 100) if impressions > 0 else 0
            comment_rate = (comments / impressions * 100) if impressions > 0 else 0
            save_rate = (saves / impressions * 100) if impressions > 0 else 0
            share_rate = (shares / impressions * 100) if impressions > 0 else 0
            click_rate = (clicks / impressions * 100) if impressions > 0 else 0
            reaction_rate = (reactions_total / impressions * 100) if impressions > 0 else 0
            story_completion_rate = _safe_float(row.get('instagram_insights_story_completion_rate'))

            enhanced = extract_instagram_enhanced_text(row)

            post_data = {
                # Identity
                "profile_id": str(row.get('profile_id', '') or ''),
                "instagram_profileId": str(row.get('instagram_profileId', '') or ''),
                "instagram_id": str(row.get('instagram_id', '') or ''),
                "instagram_url": str(row.get('instagram_url', '') or ''),

                # Timing & type
                "created_time": str(row.get('created_time', '') or ''),
                "content_type": str(row.get('content_type', '') or ''),
                "network": str(row.get('network', '') or ''),

                # Content & labels
                "instagram_content": str(row.get('instagram_content', '') or ''),
                "instagram_post_labels_names": str(row.get('instagram_post_labels_names', '') or ''),
                "instagram_post_labels": str(row.get('instagram_post_labels', '') or ''),
                "instagram_attachments": str(row.get('instagram_attachments', '') or ''),

                # Vector fields
                "labels_text": enhanced['labels_text'],
                "content_summary": enhanced['content_summary'],

                # Interaction breakdown
                "instagram_comments": comments,
                "instagram_comments_sentiment": str(row.get('instagram_comments_sentiment', '') or ''),
                "instagram_sentiment": _safe_float(row.get('instagram_sentiment')),
                "instagram_interactions": interactions,
                "instagram_media_type": str(row.get('instagram_media_type', '') or ''),
                "instagram_likes": likes,
                "instagram_shares": shares,

                # Insights
                "instagram_insights_engagement": engagements,
                "instagram_insights_impressions": impressions,
                "instagram_insights_reach": _safe_int(row.get('instagram_insights_reach')),
                "instagram_insights_video_views": video_views,
                "instagram_insights_saves": saves,
                "instagram_insights_story_completion_rate": story_completion_rate,
                "instagram_insights_post_clicks": clicks,
                "instagram_reactions": reactions_total,

                # Calculated metrics
                "engagement_rate": round(engagement_rate, 2),
                "view_rate": round(view_rate, 2),
                "like_rate": round(like_rate, 2),
                "comment_rate": round(comment_rate, 2),
                "save_rate": round(save_rate, 2),
                "share_rate": round(share_rate, 2),
                "click_rate": round(click_rate, 2),
                "reaction_rate": round(reaction_rate, 2),
            }

            relationships = map_instagram_post_relationships(row, entity_uuids)
            post_data.update(relationships)

            batch.add_object(post_data)


def ingest_instagram_knowledge_graph(client, file_path: str, mode: str = "merge"):
    if not check_weaviate_vectorizer():
        raise RuntimeError("Weaviate vectorizer not available")

    print(f"📊 Loading Instagram data from {file_path}")
    df = _read_instagram_source(file_path)

    print("🔍 Extracting Instagram entities...")
    entities = extract_instagram_entities(df)
    entity_uuids = generate_instagram_entity_uuids(entities)
    print(f"Found: {len(entities['platforms'])} platforms, {len(entities['brands'])} brands, {len(entities['content_types'])} content types")

    ensure_instagram_schema_exists(client, mode)

    ingest_instagram_entities(client, entities, entity_uuids, mode)
    ingest_instagram_posts(client, df, entity_uuids, mode)

    # Export comprehensive metrics for AI agents
    print("📈 Exporting Instagram AI-agent metrics...")
    dataset_id = generate_instagram_dataset_id(file_path)
    metrics_exporter = InstagramMetricsExporter()
    consolidated_metrics = metrics_exporter.export_all_metrics(df, entities, dataset_id)

    print("✅ Instagram knowledge graph ingestion completed!")
    print(f"📊 AI-agent metrics saved to: ./metrics/instagram/")
    return consolidated_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Ingest Instagram data into knowledge graph')
    parser.add_argument('--file', required=True, help='Instagram CSV/XLSX file to ingest')
    parser.add_argument('--mode', choices=['merge', 'replace'], default='merge',
                        help='Ingestion mode: merge (safe, default) or replace (destructive)')
    parser.add_argument('--force', action='store_true', help='Force ingestion even if duplicate')
    return parser.parse_args()


def main():
    args = parse_args()
    src_file = args.file
    mode = args.mode

    print(f"📘 Instagram Knowledge Graph Ingestion")
    print(f"🔄 Mode: {mode} ({'safe' if mode == 'merge' else 'destructive'})")

    tracker = InstagramIngestionTracker()
    if not args.force and tracker.is_duplicate(src_file):
        print(f"❌ File {src_file} already ingested. Use --force to override.")
        exit(0)

    print("🔌 Ensuring Weaviate (Docker) is running...")
    ensure_weaviate_docker()
    print("🔌 Connecting to Weaviate...")
    additional = wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=120))
    client = weaviate.connect_to_local(additional_config=additional)

    try:
        if not Path(src_file).exists():
            raise FileNotFoundError(f"Source file not found: {src_file}")

        consolidated = ingest_instagram_knowledge_graph(client, src_file, mode)

        df = _read_instagram_source(src_file)
        tracker.record_ingestion(src_file, len(df))
        print(f"✅ Instagram ingestion completed and logged")

        # Move processed file to processed folder
        dataset_id = generate_instagram_dataset_id(src_file)
        moved_file = move_processed_instagram_file(src_file, dataset_id)
        print(f"📁 Dataset ID: {dataset_id}")

    except Exception as e:
        print(f"💥 Ingestion failed: {e}")
        raise
    finally:
        client.close()
        print("🔌 Connection closed")


if __name__ == "__main__":
    main()


