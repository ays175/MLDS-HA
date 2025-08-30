#!/usr/bin/env python3
"""
TikTok-specific entity extraction for knowledge graph
"""
import pandas as pd
import json
import uuid
from typing import Dict, List, Any

def extract_tiktok_entities(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Extract entities from TikTok DataFrame"""
    
    platforms = []
    brands = []
    content_types = []
    labels = []
    
    print("🔍 Extracting TikTok entities...")
    
    # Extract platforms (should just be TikTok)
    unique_networks = df['network'].dropna().unique()
    for network in unique_networks:
        platforms.append({
            'name': str(network),
            'type': 'social_platform',
            'description': f'{network} social media platform'
        })
    
    # Extract brands from tiktok_post_labels_names
    unique_brands = set()
    for labels_str in df['tiktok_post_labels_names'].dropna():
        if isinstance(labels_str, str) and labels_str.strip():
            # Parse labels like "[Brand] Sephora Collection, [Axis] Skincare"
            label_parts = [part.strip() for part in labels_str.split(',')]
            for part in label_parts:
                if '[Brand]' in part:
                    brand_name = part.replace('[Brand]', '').strip()
                    if brand_name:
                        unique_brands.add(brand_name)
    
    for brand in unique_brands:
        brands.append({
            'name': brand,
            'type': 'brand',
            'platform': 'TikTok'
        })
    
    # Extract content axes and categories from labels
    unique_axes = set()
    unique_assets = set()
    
    for labels_str in df['tiktok_post_labels_names'].dropna():
        if isinstance(labels_str, str) and labels_str.strip():
            label_parts = [part.strip() for part in labels_str.split(',')]
            for part in label_parts:
                if '[Axis]' in part:
                    axis_name = part.replace('[Axis]', '').strip()
                    if axis_name:
                        unique_axes.add(axis_name)
                elif '[Asset]' in part:
                    asset_name = part.replace('[Asset]', '').strip()
                    if asset_name:
                        unique_assets.add(asset_name)
    
    # Add axes as content categories
    for axis in unique_axes:
        content_types.append({
            'name': axis,
            'type': 'content_axis',
            'platform': 'TikTok'
        })
    
    # Add assets as content types
    for asset in unique_assets:
        content_types.append({
            'name': asset,
            'type': 'content_asset',
            'platform': 'TikTok'
        })
    
    # Extract video duration ranges
    duration_ranges = []
    df['tiktok_duration'] = pd.to_numeric(df['tiktok_duration'], errors='coerce')
    durations = df['tiktok_duration'].dropna()
    
    if not durations.empty:
        # Create duration categories
        duration_categories = [
            {'name': 'Short (0-15s)', 'min_duration': 0, 'max_duration': 15},
            {'name': 'Medium (16-30s)', 'min_duration': 16, 'max_duration': 30},
            {'name': 'Long (31-60s)', 'min_duration': 31, 'max_duration': 60},
            {'name': 'Extended (60s+)', 'min_duration': 61, 'max_duration': 999}
        ]
        
        for cat in duration_categories:
            count = len(durations[(durations >= cat['min_duration']) & (durations <= cat['max_duration'])])
            if count > 0:
                duration_ranges.append({
                    'name': cat['name'],
                    'type': 'duration_range',
                    'min_duration': cat['min_duration'],
                    'max_duration': cat['max_duration'],
                    'post_count': count
                })
    
    entities = {
        'platforms': platforms,
        'brands': brands,
        'content_types': content_types,
        'duration_ranges': duration_ranges
    }
    
    print(f"📱 Extracted {len(platforms)} platforms")
    print(f"🏷️ Extracted {len(brands)} brands")
    print(f"📝 Extracted {len(content_types)} content types")
    print(f"⏱️ Extracted {len(duration_ranges)} duration ranges")
    
    return entities

def generate_tiktok_entity_uuids(entities: Dict[str, List[Dict]]) -> Dict[str, Dict[str, str]]:
    """Generate consistent UUIDs for TikTok entities"""
    
    entity_uuids = {
        'platforms': {},
        'brands': {},
        'content_types': {},
        'duration_ranges': {}
    }
    
    # Generate UUIDs for each entity type
    for platform in entities['platforms']:
        entity_uuids['platforms'][platform['name']] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"platform:{platform['name']}"))
    
    for brand in entities['brands']:
        entity_uuids['brands'][brand['name']] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"brand:{brand['name']}"))
    
    for content_type in entities['content_types']:
        entity_uuids['content_types'][content_type['name']] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"content:{content_type['name']}"))
    
    for duration_range in entities['duration_ranges']:
        entity_uuids['duration_ranges'][duration_range['name']] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"duration:{duration_range['name']}"))
    
    return entity_uuids

def map_tiktok_post_relationships(row: pd.Series, entity_uuids: Dict) -> Dict:
    """Map TikTok post to entity relationships"""
    
    relationships = {
        'platform': None,
        'brands': [],
        'content_types': [],
        'duration_range': None
    }
    
    # Platform relationship
    if pd.notna(row.get('network')):
        platform_name = str(row['network'])
        if platform_name in entity_uuids['platforms']:
            relationships['platform'] = entity_uuids['platforms'][platform_name]
    
    # Brand relationships from labels
    if pd.notna(row.get('tiktok_post_labels_names')):
        labels_str = str(row['tiktok_post_labels_names'])
        label_parts = [part.strip() for part in labels_str.split(',')]
        
        for part in label_parts:
            if '[Brand]' in part:
                brand_name = part.replace('[Brand]', '').strip()
                if brand_name in entity_uuids['brands']:
                    brand_uuid = entity_uuids['brands'][brand_name]
                    if brand_uuid not in relationships['brands']:
                        relationships['brands'].append(brand_uuid)
    
    # Content type relationships from labels
    if pd.notna(row.get('tiktok_post_labels_names')):
        labels_str = str(row['tiktok_post_labels_names'])
        label_parts = [part.strip() for part in labels_str.split(',')]
        
        for part in label_parts:
            if '[Axis]' in part:
                axis_name = part.replace('[Axis]', '').strip()
                if axis_name in entity_uuids['content_types']:
                    content_uuid = entity_uuids['content_types'][axis_name]
                    if content_uuid not in relationships['content_types']:
                        relationships['content_types'].append(content_uuid)
            elif '[Asset]' in part:
                asset_name = part.replace('[Asset]', '').strip()
                if asset_name in entity_uuids['content_types']:
                    content_uuid = entity_uuids['content_types'][asset_name]
                    if content_uuid not in relationships['content_types']:
                        relationships['content_types'].append(content_uuid)
    
    # Duration range relationship
    if pd.notna(row.get('tiktok_duration')):
        try:
            duration = float(row['tiktok_duration'])
            
            # Find appropriate duration range
            for range_name, range_uuid in entity_uuids['duration_ranges'].items():
                if 'Short' in range_name and duration <= 15:
                    relationships['duration_range'] = range_uuid
                    break
                elif 'Medium' in range_name and 16 <= duration <= 30:
                    relationships['duration_range'] = range_uuid
                    break
                elif 'Long' in range_name and 31 <= duration <= 60:
                    relationships['duration_range'] = range_uuid
                    break
                elif 'Extended' in range_name and duration > 60:
                    relationships['duration_range'] = range_uuid
                    break
        except (ValueError, TypeError):
            pass
    
    return relationships

def extract_enhanced_text_content(row):
    """Extract and enhance text content for better vectorization"""
    
    # Extract brands text
    brands_text = ""
    if pd.notna(row.get('tiktok_post_labels_names')):
        labels_str = str(row['tiktok_post_labels_names'])
        brand_parts = [part.strip().replace('[Brand]', '').strip() 
                      for part in labels_str.split(',') 
                      if '[Brand]' in part and part.strip().replace('[Brand]', '').strip()]
        brands_text = " ".join(brand_parts)
    
    # Extract content themes
    content_themes = ""
    if pd.notna(row.get('tiktok_post_labels_names')):
        labels_str = str(row['tiktok_post_labels_names'])
        theme_parts = []
        
        for part in labels_str.split(','):
            part = part.strip()
            if '[Axis]' in part:
                theme = part.replace('[Axis]', '').strip()
                theme_parts.append(theme)
            elif '[Asset]' in part:
                asset = part.replace('[Asset]', '').strip()
                theme_parts.append(f"{asset} content")
            elif '[Package M]' in part:
                package = part.replace('[Package M]', '').strip()
                theme_parts.append(f"{package} campaign")
        
        content_themes = " ".join(theme_parts)
    
    # Generate AI-friendly content summary
    content_summary_parts = []
    
    # Add duration context
    duration = row.get('tiktok_duration', 0)
    if duration:
        try:
            duration_num = float(duration)
            if duration_num <= 15:
                content_summary_parts.append("short form video content")
            elif duration_num <= 30:
                content_summary_parts.append("medium length video content")
            elif duration_num <= 60:
                content_summary_parts.append("long form video content")
            else:
                content_summary_parts.append("extended video content")
        except:
            pass
    
    # Add brand context
    if brands_text:
        content_summary_parts.append(f"featuring {brands_text}")
    
    # Add theme context
    if content_themes:
        content_summary_parts.append(f"focused on {content_themes}")
    
    # Add performance context
    completion_rate = row.get('tiktok_insights_completion_rate', 0)
    if completion_rate:
        try:
            completion_num = float(completion_rate)
            if completion_num > 0.8:
                content_summary_parts.append("high engagement content")
            elif completion_num > 0.5:
                content_summary_parts.append("moderate engagement content")
            elif completion_num > 0.2:
                content_summary_parts.append("low engagement content")
        except:
            pass
    
    content_summary = " ".join(content_summary_parts)
    
    return {
        'labels_text': str(row.get('tiktok_post_labels_names', '')),
        'brands_text': brands_text,
        'content_themes': content_themes,
        'content_summary': content_summary
    }

def extract_tiktok_attachments_info(row: pd.Series) -> Dict[str, Any]:
    """Infer basic attachments info from a row.

    Heuristics:
    - If an 'attachments' JSON-like field exists, count items and infer type
    - Else use 'tiktok_media_type' if present
    - Fallback to single video
    """
    media_type = "video"
    media_count = 1

    # Try explicit media type column first
    mt = row.get('tiktok_media_type') or row.get('media_type')
    if pd.notna(mt):
        media_type = str(mt).strip().lower() or media_type

    # Try parsing attachments JSON/list
    attachments = row.get('attachments')
    if pd.notna(attachments):
        try:
            parsed = attachments
            if isinstance(attachments, str):
                parsed = json.loads(attachments)
            if isinstance(parsed, list):
                media_count = len(parsed) or media_count
                # Infer type from first item if available
                first = parsed[0] if parsed else None
                if isinstance(first, dict):
                    t = first.get('type') or first.get('mime') or first.get('media_type')
                    if t:
                        media_type = str(t).split('/')[-1].lower()
            elif isinstance(parsed, dict):
                media_count = int(parsed.get('count', media_count))
                t = parsed.get('type') or parsed.get('mime') or parsed.get('media_type')
                if t:
                    media_type = str(t).split('/')[-1].lower()
        except Exception:
            # Ignore parsing errors and keep defaults
            pass

    return {"media_type": media_type, "media_count": media_count}

def extract_label_insights(labels_str: str) -> Dict[str, Any]:
    """Parse TikTok label string like '[Brand] X, [Axis] Y' into categories.

    Returns counts and lists per category for downstream use.
    """
    result: Dict[str, Any] = {
        "brands": [],
        "axes": [],
        "assets": [],
        "packages": [],
    }

    if not labels_str or not isinstance(labels_str, str):
        return {**result, "counts": {k: 0 for k in result.keys()}}

    parts = [part.strip() for part in labels_str.split(',') if part.strip()]
    for part in parts:
        if '[Brand]' in part:
            name = part.replace('[Brand]', '').strip()
            if name:
                result["brands"].append(name)
        elif '[Axis]' in part:
            name = part.replace('[Axis]', '').strip()
            if name:
                result["axes"].append(name)
        elif '[Asset]' in part:
            name = part.replace('[Asset]', '').strip()
            if name:
                result["assets"].append(name)
        elif '[Package M]' in part or '[Package]' in part:
            name = part.replace('[Package M]', '').replace('[Package]', '').strip()
            if name:
                result["packages"].append(name)

    result["counts"] = {k: len(v) for k, v in result.items() if isinstance(v, list)}
    return result
