#!/usr/bin/env python3
"""
Instagram-specific entity extraction for knowledge graph
"""
import hashlib
import json
import re
from typing import Dict, List, Any

import pandas as pd


def _parse_label_items(label_text: str) -> List[Dict[str, str]]:
    """Extract category and name pairs from bracketed labels.

    Example: "[Axis] Makeup, [Brand] K18" ->
    [{"category": "Axis", "name": "Makeup"}, {"category": "Brand", "name": "K18"}]
    """
    if not isinstance(label_text, str):
        return []
    items: List[Dict[str, str]] = []
    for part in label_text.split(','):
        part = part.strip()
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", part)
        if m:
            category = m.group(1).strip()
            name = m.group(2).strip()
            if name:
                items.append({"category": category, "name": name})
    return items


def extract_instagram_entities(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Extract platforms, brands, and content types from Instagram DataFrame."""
    platforms = [{"name": "Instagram", "type": "social", "description": "Instagram platform"}]

    brand_names = set()
    content_types: Dict[str, Dict[str, str]] = {}

    for _, row in df.iterrows():
        labels = _parse_label_items(row.get('instagram_post_labels_names', ''))
        for item in labels:
            if item['category'].lower().strip() == 'brand':
                brand_names.add(item['name'])
            else:
                key = (item['category'], item['name'])
                content_types[key] = {
                    "name": item['name'],
                    "type": item['category'],
                    "platform": "Instagram",
                }

    brands = [{"name": b, "type": "brand", "platform": "Instagram"} for b in sorted(brand_names)]
    content_types_list = list(content_types.values())
    return {"platforms": platforms, "brands": brands, "content_types": content_types_list}


def _uuid_for(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def generate_instagram_entity_uuids(entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
    """Generate consistent UUIDs for Instagram entities."""
    uuids: Dict[str, Dict[str, str]] = {"platforms": {}, "brands": {}, "content_types": {}}
    for p in entities['platforms']:
        uuids['platforms'][p['name']] = _uuid_for(f"instagram:platform:{p['name']}")
    for b in entities['brands']:
        uuids['brands'][b['name']] = _uuid_for(f"instagram:brand:{b['name']}")
    for c in entities['content_types']:
        key = f"{c['type']}::{c['name']}"
        uuids['content_types'][key] = _uuid_for(f"instagram:contenttype:{key}")
    return uuids


def map_instagram_post_relationships(row: pd.Series, entity_uuids: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Map Instagram post to entity relationship UUIDs based on labels."""
    labels = _parse_label_items(row.get('instagram_post_labels_names', ''))

    platform_uuid = entity_uuids['platforms'].get('Instagram')
    brand_uuids: List[str] = []
    content_type_uuids: List[str] = []

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


def extract_instagram_enhanced_text(row: pd.Series) -> Dict[str, str]:
    """Return simple enhanced text fields for vectorization."""
    content = str(row.get('instagram_content', '') or '')
    labels = str(row.get('instagram_post_labels_names', '') or '')
    attachments = str(row.get('instagram_attachments', '') or '')
    labels_text = "; ".join([p.strip() for p in labels.split(',') if p.strip()])
    content_summary = " ".join(filter(None, [content, labels_text]))
    if len(content_summary) < 3 and attachments:
        content_summary = attachments[:500]
    return {"labels_text": labels_text, "content_summary": content_summary}


def extract_instagram_attachments_info(row: pd.Series) -> Dict[str, Any]:
    """Infer basic attachments info from a row (media_type, media_count).

    Heuristics:
    - If an 'instagram_attachments' JSON-like field exists, count items and infer type
    - Else use 'instagram_media_type' if present
    - Fallback to single photo
    """
    media_type = "photo"
    media_count = 1

    mt = row.get('instagram_media_type') or row.get('media_type')
    if pd.notna(mt):
        try:
            media_type = str(mt).strip().lower() or media_type
        except Exception:
            pass

    attachments = row.get('instagram_attachments')
    if pd.notna(attachments):
        try:
            parsed = attachments
            if isinstance(attachments, str):
                parsed = json.loads(attachments)
            if isinstance(parsed, list):
                media_count = len(parsed) or media_count
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
            pass

    return {"media_type": media_type, "media_count": media_count}
