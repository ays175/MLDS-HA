#!/usr/bin/env python3
"""
Facebook knowledge graph ingestion using Weaviate
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

import pandas as pd
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv

from facebook_schema import create_facebook_knowledge_graph_schema
from facebook_metrics_export import FacebookMetricsExporter
from facebook_entities import (
    extract_facebook_entities,
    generate_facebook_entity_uuids,
    map_facebook_post_relationships,
    extract_enhanced_text_content,
)


# Load environment variables
load_dotenv()


class FacebookIngestionTracker:
    """Track Facebook ingestion to avoid duplicates"""

    def __init__(self):
        self.tracker_file = Path("data/facebook_ingestion_log.json")
        self.tracker_file.parent.mkdir(exist_ok=True)

        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = {"ingestions": []}

    def is_duplicate(self, csv_file: str) -> bool:
        file_hash = self._get_file_hash(csv_file)
        return any(ing['file_hash'] == file_hash for ing in self.log['ingestions'])

    def record_ingestion(self, csv_file: str, record_count: int):
        self.log['ingestions'].append({
            'file_name': Path(csv_file).name,
            'file_hash': self._get_file_hash(csv_file),
            'record_count': record_count,
            'ingested_at': datetime.now().isoformat()
        })

        with open(self.tracker_file, 'w') as f:
            json.dump(self.log, f, indent=2)

    def _get_file_hash(self, csv_file: str) -> str:
        with open(csv_file, 'rb') as f:
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


def generate_facebook_dataset_id(csv_file_path: str) -> str:
    """Generate a unique dataset ID based on file content"""
    with open(csv_file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:8]
    return f"facebook_dataset_{file_hash}"


def move_processed_facebook_csv(csv_file_path: str, dataset_id: str) -> str:
    """Move processed CSV to processed/facebook folder with dataset ID and timestamp"""
    csv_path = Path(csv_file_path)
    processed_dir = Path("data/processed/facebook")

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{dataset_id}_{timestamp}"
    target_folder = processed_dir / folder_name
    target_folder.mkdir(parents=True, exist_ok=True)

    # Move CSV file
    target_file = target_folder / csv_path.name
    try:
        import shutil
        shutil.move(str(csv_path), str(target_file))
        print(f"📁 Moved processed CSV: {csv_path.name} → {target_folder}/")
    except Exception:
        # If moving fails (e.g., same filesystem constraints), copy as fallback
        try:
            import shutil
            shutil.copy2(str(csv_path), str(target_file))
            print(f"📁 Copied processed CSV: {csv_path.name} → {target_folder}/")
        except Exception:
            print("⚠️ Could not move/copy processed CSV")
            return str(csv_path)
    return str(target_file)


def ensure_facebook_schema_exists(client, mode: str = "merge"):
    if mode == "replace":
        create_facebook_knowledge_graph_schema(client)
    else:
        collections = ["FacebookPlatform", "FacebookBrand", "FacebookContentType", "FacebookPost"]
        missing = []
        for name in collections:
            try:
                client.collections.get(name)
            except Exception:
                missing.append(name)
        if missing:
            print(f"Creating missing collections: {missing}")
            create_facebook_knowledge_graph_schema(client)


def _safe_int(value, default: int = 0) -> int:
    try:
        if value == '' or value is None:
            return default
        # Treat NaN-like values as default
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
        # Treat NaN-like values as default
        if isinstance(value, str) and value.strip().lower() in ('nan', 'na', 'none'):
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _parse_label_items(label_text: str):
    """Extract category and name pairs from bracketed labels.
    Example: "[Axis] Makeup, [Brand] K18" -> [{category:"Axis", name:"Makeup"}, {category:"Brand", name:"K18"}]"""
    if not isinstance(label_text, str):
        return []
    items = []
    for part in label_text.split(','):
        part = part.strip()
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", part)
        if m:
            category = m.group(1).strip()
            name = m.group(2).strip()
            if name:
                items.append({"category": category, "name": name})
    return items


def extract_facebook_entities(df: pd.DataFrame):
    platforms = [{"name": "Facebook", "type": "social", "description": "Facebook platform"}]

    brand_names = set()
    content_types = {}

    for _, row in df.iterrows():
        labels = _parse_label_items(row.get('facebook_post_labels_names', ''))
        for item in labels:
            if item['category'].lower().strip() == 'brand':
                brand_names.add(item['name'])
            else:
                key = (item['category'], item['name'])
                content_types[key] = {
                    "name": item['name'],
                    "type": item['category'],
                    "platform": "Facebook",
                }

    brands = [{"name": b, "type": "brand", "platform": "Facebook"} for b in sorted(brand_names)]
    content_types_list = list(content_types.values())
    return {"platforms": platforms, "brands": brands, "content_types": content_types_list}


def _uuid_for(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def generate_facebook_entity_uuids(entities):
    uuids = {"platforms": {}, "brands": {}, "content_types": {}}
    for p in entities['platforms']:
        uuids['platforms'][p['name']] = _uuid_for(f"facebook:platform:{p['name']}")
    for b in entities['brands']:
        uuids['brands'][b['name']] = _uuid_for(f"facebook:brand:{b['name']}")
    for c in entities['content_types']:
        key = f"{c['type']}::{c['name']}"
        uuids['content_types'][key] = _uuid_for(f"facebook:contenttype:{key}")
    return uuids


def ingest_facebook_entities(client, entities, entity_uuids, mode: str = "merge"):
    if mode == "replace":
        print("🔄 Replace mode: Adding all entities fresh")

        platform_collection = client.collections.get("FacebookPlatform")
        with platform_collection.batch.dynamic() as batch:
            for platform in entities['platforms']:
                batch.add_object(platform, uuid=entity_uuids['platforms'][platform['name']])
        print(f"📱 Added {len(entities['platforms'])} platforms")

        brand_collection = client.collections.get("FacebookBrand")
        with brand_collection.batch.dynamic() as batch:
            for brand in entities['brands']:
                batch.add_object(brand, uuid=entity_uuids['brands'][brand['name']])
        print(f"🏷️ Added {len(entities['brands'])} brands")

        content_collection = client.collections.get("FacebookContentType")
        with content_collection.batch.dynamic() as batch:
            for ct in entities['content_types']:
                key = f"{ct['type']}::{ct['name']}"
                batch.add_object(ct, uuid=entity_uuids['content_types'][key])
        print(f"📝 Added {len(entities['content_types'])} content types")
        return

    # Merge mode
    print("🔗 Merge mode: Adding only new entities")

    # Platforms
    platform_collection = client.collections.get("FacebookPlatform")
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
    brand_collection = client.collections.get("FacebookBrand")
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
    content_collection = client.collections.get("FacebookContentType")
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


def _map_facebook_post_relationships(row, entity_uuids):
    labels = _parse_label_items(row.get('facebook_post_labels_names', ''))

    platform_uuid = entity_uuids['platforms'].get('Facebook')
    brand_uuids = []
    content_type_uuids = []

    for item in labels:
        if item['category'].lower().strip() == 'brand':
            uuid = entity_uuids['brands'].get(item['name'])
            if uuid:
                brand_uuids.append(uuid)
        else:
            key = f"{item['category']}::{item['name']}"
            uuid = entity_uuids['content_types'].get(key)
            if uuid:
                content_type_uuids.append(uuid)

    return {
        "platform": platform_uuid,
        "brands": brand_uuids,
        "content_types": content_type_uuids,
    }


def _extract_enhanced_text_content(row):
    content = str(row.get('facebook_content', '') or '')
    labels = str(row.get('facebook_post_labels_names', '') or '')
    attachments = str(row.get('facebook_attachments', '') or '')
    labels_text = "; ".join([p.strip() for p in labels.split(',') if p.strip()])
    content_summary = " ".join(filter(None, [content, labels_text]))
    if len(content_summary) < 3 and attachments:
        content_summary = attachments[:500]
    return {"labels_text": labels_text, "content_summary": content_summary}


def ingest_facebook_posts(client, df: pd.DataFrame, entity_uuids, mode: str = "merge"):
    post_collection = client.collections.get("FacebookPost")

    new_posts_df = df
    if mode == "merge":
        try:
            existing_ids = {p.properties['facebook_id'] for p in post_collection.iterator()}
            new_posts_df = df[~df['facebook_id'].astype(str).isin(existing_ids)]
            print(f"📝 Found {len(df)} total posts, {len(new_posts_df)} are new")
        except Exception:
            print("📝 No existing posts found, processing all")

    if len(new_posts_df) == 0:
        print("📝 All posts already exist")
        return

    print("📝 Ingesting Facebook posts with relationships...")

    with post_collection.batch.fixed_size(batch_size=10) as batch:
        for _, row in new_posts_df.iterrows():
            impressions = _safe_int(row.get('facebook_insights_impressions'))
            video_views = _safe_int(row.get('facebook_insights_video_views'))
            engagements = _safe_int(row.get('facebook_insights_engagements'))
            interactions = _safe_int(row.get('facebook_insights_interactions'))
            if engagements == 0 and interactions > 0:
                engagements = interactions

            reactions_total = _safe_int(row.get('facebook_insights_reactions'))
            if reactions_total == 0:
                reactions_total = _safe_int(row.get('facebook_reactions'))

            likes = _safe_int(row.get('facebook_reaction_like'))
            shares = _safe_int(row.get('facebook_shares'))
            comments = _safe_int(row.get('facebook_comments'))
            clicks = _safe_int(row.get('facebook_insights_post_clicks'))

            engagement_rate = (engagements / impressions * 100) if impressions > 0 else 0
            view_rate = (video_views / impressions * 100) if impressions > 0 else 0
            like_rate = (likes / impressions * 100) if impressions > 0 else 0
            share_rate = (shares / impressions * 100) if impressions > 0 else 0
            comment_rate = (comments / impressions * 100) if impressions > 0 else 0
            click_rate = (clicks / impressions * 100) if impressions > 0 else 0
            reaction_rate = (reactions_total / impressions * 100) if impressions > 0 else 0
            completion_rate = _safe_float(row.get('facebook_insights_video_views_average_completion'))

            enhanced = extract_enhanced_text_content(row)

            post_data = {
                # Identity
                "profile_id": str(row.get('profile_id', '') or ''),
                "facebook_profileId": str(row.get('facebook_profileId', '') or ''),
                "facebook_id": str(row.get('facebook_id', '') or ''),
                "facebook_url": str(row.get('facebook_url', '') or ''),

                # Timing & type
                "created_time": str(row.get('created_time', '') or ''),
                "content_type": str(row.get('content_type', '') or ''),
                "network": str(row.get('network', '') or ''),
                "facebook_published": str(row.get('facebook_published', '') or ''),

                # Content & labels
                "facebook_content": str(row.get('facebook_content', '') or ''),
                "facebook_post_labels_names": str(row.get('facebook_post_labels_names', '') or ''),
                "facebook_post_labels": str(row.get('facebook_post_labels', '') or ''),
                "facebook_attachments": str(row.get('facebook_attachments', '') or ''),

                # Vector fields
                "labels_text": enhanced['labels_text'],
                "content_summary": enhanced['content_summary'],

                # Interaction breakdown
                "facebook_comments": comments,
                "facebook_comments_sentiment": str(row.get('facebook_comments_sentiment', '') or ''),
                "facebook_sentiment": _safe_float(row.get('facebook_sentiment')),
                "facebook_interactions": _safe_int(row.get('facebook_interactions')),
                "facebook_media_type": str(row.get('facebook_media_type', '') or ''),
                "facebook_reactions": reactions_total,
                "facebook_shares": shares,

                # Insights
                "facebook_insights_engagements": engagements,
                "facebook_insights_impressions": impressions,
                "facebook_insights_interactions": interactions,
                "facebook_insights_post_clicks": clicks,
                "facebook_insights_reach": _safe_int(row.get('facebook_insights_reach')),
                "facebook_insights_reactions": reactions_total,
                "facebook_insights_video_views": video_views,
                "facebook_insights_video_views_average_completion": completion_rate,

                # Reaction breakdown
                "facebook_reaction_anger": _safe_int(row.get('facebook_reaction_anger')),
                "facebook_reaction_haha": _safe_int(row.get('facebook_reaction_haha')),
                "facebook_reaction_like": likes,
                "facebook_reaction_love": _safe_int(row.get('facebook_reaction_love')),
                "facebook_reaction_sorry": _safe_int(row.get('facebook_reaction_sorry')),
                "facebook_reaction_wow": _safe_int(row.get('facebook_reaction_wow')),

                # Calculated metrics
                "engagement_rate": round(engagement_rate, 2),
                "view_rate": round(view_rate, 2),
                "like_rate": round(like_rate, 2),
                "share_rate": round(share_rate, 2),
                "comment_rate": round(comment_rate, 2),
                "click_rate": round(click_rate, 2),
                "reaction_rate": round(reaction_rate, 2),
                "completion_rate": round(completion_rate, 4),
            }

            relationships = map_facebook_post_relationships(row, entity_uuids)
            post_data.update(relationships)

            batch.add_object(post_data)


def ingest_facebook_knowledge_graph(client, csv_file: str, mode: str = "merge"):
    if not check_weaviate_vectorizer():
        raise RuntimeError("Weaviate vectorizer not available")

    print(f"📊 Loading Facebook data from {csv_file}")
    df = pd.read_csv(csv_file)

    print("🔍 Extracting Facebook entities...")
    entities = extract_facebook_entities(df)
    entity_uuids = generate_facebook_entity_uuids(entities)
    print(f"Found: {len(entities['platforms'])} platforms, {len(entities['brands'])} brands, {len(entities['content_types'])} content types")

    ensure_facebook_schema_exists(client, mode)

    ingest_facebook_entities(client, entities, entity_uuids, mode)
    ingest_facebook_posts(client, df, entity_uuids, mode)

    # Export comprehensive metrics for AI agents
    print("📈 Exporting Facebook AI-agent metrics...")
    dataset_id = generate_facebook_dataset_id(csv_file)
    metrics_exporter = FacebookMetricsExporter()
    consolidated_metrics = metrics_exporter.export_all_metrics(df, entities, dataset_id)

    print("✅ Facebook knowledge graph ingestion completed!")
    print(f"📊 AI-agent metrics saved to: ./metrics/facebook/")
    return consolidated_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Ingest Facebook data into knowledge graph')
    parser.add_argument('--csv-file', required=True, help='Facebook CSV file to ingest')
    parser.add_argument('--mode', choices=['merge', 'replace'], default='merge',
                        help='Ingestion mode: merge (safe, default) or replace (destructive)')
    parser.add_argument('--force', action='store_true', help='Force ingestion even if duplicate')
    return parser.parse_args()


def main():
    args = parse_args()
    csv_file = args.csv_file
    mode = args.mode

    print(f"📘 Facebook Knowledge Graph Ingestion")
    print(f"🔄 Mode: {mode} ({'safe' if mode == 'merge' else 'destructive'})")

    tracker = FacebookIngestionTracker()
    if not args.force and tracker.is_duplicate(csv_file):
        print(f"❌ File {csv_file} already ingested. Use --force to override.")
        exit(0)

    print("🔌 Ensuring Weaviate (Docker) is running...")
    ensure_weaviate_docker()
    print("🔌 Connecting to Weaviate...")
    additional = wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=120))
    client = weaviate.connect_to_local(additional_config=additional)

    try:
        if not Path(csv_file).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        consolidated = ingest_facebook_knowledge_graph(client, csv_file, mode)

        df = pd.read_csv(csv_file)
        tracker.record_ingestion(csv_file, len(df))
        print(f"✅ Facebook ingestion completed and logged")

        # Move processed CSV to processed folder
        dataset_id = generate_facebook_dataset_id(csv_file)
        moved_file = move_processed_facebook_csv(csv_file, dataset_id)
        print(f"📁 Dataset ID: {dataset_id}")

    except Exception as e:
        print(f"💥 Ingestion failed: {e}")
        raise
    finally:
        client.close()
        print("🔌 Connection closed")


if __name__ == "__main__":
    main()


