#!/usr/bin/env python3
"""
Unified Entity Extractor - Single source of truth for all entity extraction
Replaces all platform-specific entity files
"""
import hashlib
import json
import re
import uuid
from typing import Dict, List, Any, Tuple, Set
import pandas as pd
import yaml
from pathlib import Path


class UnifiedEntityExtractor:
    """Extract entities for any platform using configuration"""
    
    def __init__(self, platform: str):
        self.platform = platform
        self.config = self._load_config(platform)
    
    def _load_config(self, platform: str) -> Dict[str, Any]:
        """Load platform configuration"""
        config_path = Path(f"ingestion/configs/{platform}.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract_entities(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all entities from dataframe"""
        entities = {}
        
        # Always create platform entity
        entities['platforms'] = [self._create_platform_entity()]
        
        # Extract configured entity types
        entity_config = self.config.get('entities', {})
        
        # Extract brands
        if 'brand_extraction' in entity_config:
            entities['brands'] = self._extract_brands(df, entity_config['brand_extraction'])
        
        # Extract content types
        if 'content_type_extraction' in entity_config:
            entities['content_types'] = self._extract_content_types(df, entity_config['content_type_extraction'])
        
        # Extract countries (add derived_country to dataframe)
        if 'country_extraction' in entity_config:
            self._extract_countries(df, entity_config['country_extraction'])
        
        # Platform-specific entities
        if self.platform == 'tiktok' and 'duration_ranges' in entity_config:
            entities['duration_ranges'] = entity_config['duration_ranges']
        
        if self.platform == 'customer_care':
            entities.update(self._extract_customer_care_entities(df))
        
        return entities
    
    def _create_platform_entity(self) -> Dict[str, Any]:
        """Create platform entity"""
        platform_config = self.config['platform']
        return {
            'name': platform_config['display_name'],
            'type': platform_config['type'],
            'description': platform_config['description']
        }
    
    def _extract_brands(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract brand entities based on configuration"""
        brands = set()
        
        if 'source_field' in config and 'pattern' in config:
            # Single field pattern extraction (social platforms)
            pattern = config['pattern']
            field = config['source_field']
            fallback_field = config.get('fallback_field')
            
            # Try structured field first (e.g., tiktok_post_labels)
            if field in df.columns:
                for text in df[field].dropna():
                    text_str = str(text)
                    if field == 'tiktok_post_labels':
                        # Use improved JSON parsing for structured labels
                        extracted_brands = self._parse_tiktok_labels_for_brands(text_str)
                        brands.update(extracted_brands)
                    else:
                        # Use regex pattern for other fields
                        matches = re.findall(pattern, text_str)
                        brands.update(matches)
            
            # Fallback to secondary field if no brands found
            if not brands and fallback_field and fallback_field in df.columns:
                for text in df[fallback_field].dropna():
                    matches = re.findall(pattern, str(text))
                    brands.update(matches)
        
        elif 'source_fields' in config and 'patterns' in config:
            # Multi-field pattern extraction (customer care)
            for field in config['source_fields']:
                if field in df.columns:
                    for pattern in config['patterns']:
                        for text in df[field].dropna():
                            matches = re.findall(pattern, str(text), re.IGNORECASE)
                            brands.update(matches)
        
        # Return sorted list of brand entities
        return [
            {
                'name': brand,
                'type': 'brand',
                'platform': self.config['platform']['display_name']
            }
            for brand in sorted(brands)
        ]
    
    def _extract_content_types(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract content type entities"""
        content_types = {}
        
        if 'source_field' in config:
            pattern = config['pattern']
            field = config['source_field']
            fallback_field = config.get('fallback_field')
            exclude = config.get('exclude_categories', [])
            
            # Try structured field first (e.g., tiktok_post_labels)
            if field in df.columns:
                for text in df[field].dropna():
                    text_str = str(text)
                    if field == 'tiktok_post_labels':
                        # Use improved JSON parsing for structured labels
                        extracted_content = self._parse_tiktok_labels_for_content(text_str)
                        for content_info in extracted_content:
                            category = content_info.get('category', 'content')
                            name = content_info.get('type', 'Unknown')
                            
                            if category.lower() not in [e.lower() for e in exclude]:
                                key = f"{category}::{name}"
                                content_types[key] = {
                                    'name': name,
                                    'type': category,
                                    'category': category
                                }
                    else:
                        # Use regex pattern for other fields
                        matches = re.findall(pattern, text_str)
                        for match in matches:
                            if isinstance(match, tuple) and len(match) >= 2:
                                category, name = match[0], match[1]
                            else:
                                # Handle single capture group
                                category = config.get('default_category', 'content')
                                name = match
                            
                                key = f"{category}::{name}"
                                content_types[key] = {
                                    'name': name,
                                    'type': category,
                                    'category': category
                                }
            
            # Fallback to secondary field if no content types found
            if not content_types and fallback_field and fallback_field in df.columns:
                for text in df[fallback_field].dropna():
                    matches = re.findall(pattern, str(text))
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 2:
                            category, name = match[0], match[1]
                        else:
                            # Handle single capture group
                            category = config.get('default_category', 'content')
                            name = match
                        
                        if category.lower() not in [e.lower() for e in exclude]:
                            key = f"{category}::{name}"
                            content_types[key] = {
                                'name': name,
                                'type': category,
                                'category': category
                            }
        
        return list(content_types.values())
    
    def _parse_tiktok_labels_for_brands(self, labels_json: str) -> List[str]:
        """Parse tiktok_post_labels JSON field to extract brands"""
        import json
        import re
        
        brands = []
        
        try:
            # Handle multiple JSON objects separated by commas
            if labels_json.strip() and not labels_json.strip().startswith('['):
                # Use regex to find JSON object boundaries
                pattern = r'\{[^{}]*\}'
                matches = re.findall(pattern, labels_json)
                
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict) and 'name' in obj:
                            name = obj['name'].strip()
                            if '[Brand]' in name:
                                brand = name.replace('[Brand]', '').strip()
                                if brand:
                                    brands.append(brand)
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            # Fallback: treat as string and parse manually
            if '[Brand]' in labels_json:
                parts = labels_json.split('[Brand]')
                for part in parts[1:]:  # Skip first part (before first [Brand])
                    # Extract text until next bracket or comma
                    brand_match = re.search(r'^([^,\[\]]+)', part)
                    if brand_match:
                        brand = brand_match.group(1).strip().strip('"').strip()
                        if brand:
                            brands.append(brand)
        
        return brands
    
    def _parse_tiktok_labels_for_content(self, labels_json: str) -> List[Dict[str, str]]:
        """Parse tiktok_post_labels JSON field to extract content types"""
        import json
        import re
        
        content_types = []
        
        try:
            # Handle multiple JSON objects separated by commas
            if labels_json.strip() and not labels_json.strip().startswith('['):
                # Use regex to find JSON object boundaries
                pattern = r'\{[^{}]*\}'
                matches = re.findall(pattern, labels_json)
                
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict) and 'name' in obj:
                            name = obj['name'].strip()
                            
                            # Extract content types
                            for tag, category in [('[Axis]', 'axis'), ('[Asset]', 'asset'), ('[Package M]', 'package')]:
                                if tag in name:
                                    content_name = name.replace(tag, '').strip()
                                    if content_name:
                                        content_types.append({'type': content_name, 'category': category})
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            # Fallback: treat as string and parse manually
            for tag, category in [('[Axis]', 'axis'), ('[Asset]', 'asset'), ('[Package M]', 'package')]:
                if tag in labels_json:
                    parts = labels_json.split(tag)
                    for part in parts[1:]:
                        content_match = re.search(r'^([^,\[\]]+)', part)
                        if content_match:
                            content_name = content_match.group(1).strip().strip('"').strip()
                            if content_name:
                                content_types.append({'type': content_name, 'category': category})
        
        return content_types
    
    def _extract_customer_care_entities(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Extract customer care specific entities"""
        entities = {}
        
        # Issue types
        if 'issue_type' in df.columns:
            issue_types = df['issue_type'].dropna().unique()
            entities['issue_types'] = [
                {
                    'name': str(it),
                    'type': 'issue_type',
                    'description': f'{it} support issues'
                }
                for it in issue_types
            ]
        
        # Channels
        if 'channel' in df.columns:
            channels = df['channel'].dropna().unique()
            entities['channels'] = [
                {
                    'name': str(ch),
                    'type': 'channel',
                    'description': f'{ch} communication channel'
                }
                for ch in channels
            ]
        
        # Priorities
        if 'priority' in df.columns:
            priorities = df['priority'].dropna().unique()
            entities['priorities'] = [
                {
                    'name': str(pr),
                    'type': 'priority',
                    'level': str(pr),
                    'description': f'{pr} priority cases'
                }
                for pr in priorities
            ]
        
        return entities
    
    def generate_entity_uuids(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
        """Generate deterministic UUIDs for all entities"""
        uuids = {}
        
        for entity_type, entity_list in entities.items():
            uuids[entity_type] = {}
            
            for entity in entity_list:
                # Generate deterministic UUID based on platform, type, and name
                name = entity.get('name', '')
                
                # Special handling for content types with category
                if entity_type == 'content_types' and 'type' in entity:
                    key = f"{entity['type']}::{entity['name']}"
                else:
                    key = name
                
                # Create deterministic UUID
                namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                entity_uuid = str(uuid.uuid5(
                    namespace, 
                    f"{self.platform}:{entity_type}:{key}"
                ))
                
                uuids[entity_type][key] = entity_uuid
        
        return uuids
    
    def map_relationships(self, row: pd.Series, entity_uuids: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Map row to entity relationships"""
        relationships = {}
        
        # Platform relationship
        platform_name = self.config['platform']['display_name']
        if 'platforms' in entity_uuids and platform_name in entity_uuids['platforms']:
            relationships['platform'] = entity_uuids['platforms'][platform_name]
        
        # Extract relationships based on platform
        if self.platform in ['facebook', 'instagram', 'tiktok']:
            relationships.update(self._map_social_relationships(row, entity_uuids))
        elif self.platform == 'customer_care':
            relationships.update(self._map_customer_care_relationships(row, entity_uuids))
        
        return relationships
    
    def _map_social_relationships(self, row: pd.Series, entity_uuids: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Map social media post relationships"""
        relationships = {
            'brands': [],
            'content_types': []
        }
        
        # Parse labels to find relationships
        label_field = f"{self.platform}_post_labels_names"
        if label_field not in row:
            # Try without platform prefix
            label_field = "post_labels_names"
        
        if label_field in row and pd.notna(row[label_field]):
            labels = self._parse_label_items(str(row[label_field]))
            
            for item in labels:
                category = item.get('category', '').lower().strip()
                name = item.get('name', '')
                
                if category == 'brand' and 'brands' in entity_uuids:
                    if name in entity_uuids['brands']:
                        relationships['brands'].append(entity_uuids['brands'][name])
                elif 'content_types' in entity_uuids:
                    key = f"{item['category']}::{name}"
                    if key in entity_uuids['content_types']:
                        relationships['content_types'].append(entity_uuids['content_types'][key])
        
        # TikTok duration range
        if self.platform == 'tiktok' and 'duration_range' in row:
            if 'duration_ranges' in entity_uuids:
                dr = row['duration_range']
                if dr in entity_uuids['duration_ranges']:
                    relationships['duration_range_ref'] = entity_uuids['duration_ranges'][dr]
        
        return relationships
    
    def _map_customer_care_relationships(self, row: pd.Series, entity_uuids: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Map customer care case relationships"""
        relationships = {}
        
        # Issue type
        if 'issue_type' in row and 'issue_types' in entity_uuids:
            issue_type = str(row['issue_type'])
            if issue_type in entity_uuids['issue_types']:
                relationships['issue_type_ref'] = entity_uuids['issue_types'][issue_type]
        
        # Channel
        if 'channel' in row and 'channels' in entity_uuids:
            channel = str(row['channel'])
            if channel in entity_uuids['channels']:
                relationships['channel_ref'] = entity_uuids['channels'][channel]
        
        # Priority
        if 'priority' in row and 'priorities' in entity_uuids:
            priority = str(row['priority'])
            if priority in entity_uuids['priorities']:
                relationships['priority_ref'] = entity_uuids['priorities'][priority]
        
        return relationships
    
    def _parse_label_items(self, label_text: str) -> List[Dict[str, str]]:
        """Parse bracketed labels into category/name pairs"""
        items = []
        
        # Pattern: [Category] Name
        pattern = r'\[([^\]]+)\]\s*([^,\[]+)'
        matches = re.findall(pattern, label_text)
        
        for category, name in matches:
            items.append({
                'category': category.strip(),
                'name': name.strip()
            })
        
        return items
    
    def extract_enhanced_content(self, row: pd.Series) -> Dict[str, Any]:
        """Extract enhanced content fields for any platform"""
        enhanced = {}
        
        if self.platform == 'facebook':
            enhanced.update(self._extract_facebook_enhanced_content(row))
        elif self.platform == 'instagram':
            enhanced.update(self._extract_instagram_enhanced_content(row))
        elif self.platform == 'tiktok':
            enhanced.update(self._extract_tiktok_enhanced_content(row))
        elif self.platform == 'customer_care':
            enhanced.update(self._extract_customer_care_enhanced_content(row))
        
        return enhanced
    
    def _extract_facebook_enhanced_content(self, row: pd.Series) -> Dict[str, Any]:
        """Facebook-specific content enhancement"""
        content = str(row.get('facebook_content', '') or '')
        labels = str(row.get('facebook_post_labels_names', '') or '')
        attachments = str(row.get('facebook_attachments', '') or '')
        
        # Create labels text
        labels_text = "; ".join([p.strip() for p in labels.split(',') if p.strip()])
        
        # Create content summary
        content_summary = " ".join(filter(None, [content, labels_text]))
        if len(content_summary) < 3 and attachments:
            content_summary = attachments[:500]
        
        # Extract attachment info
        media_type = "photo"
        media_count = 1
        
        if pd.notna(row.get('facebook_media_type')):
            media_type = str(row['facebook_media_type']).lower()
        
        if pd.notna(attachments):
            try:
                if attachments.startswith('[') or attachments.startswith('{'):
                    parsed = json.loads(attachments)
                    if isinstance(parsed, list):
                        media_count = len(parsed)
                    elif isinstance(parsed, dict):
                        media_count = parsed.get('count', 1)
            except:
                pass
        
        return {
            'labels_text': labels_text,
            'content_summary': content_summary,
            'media_type': media_type,
            'media_count': media_count
        }
    
    def _extract_instagram_enhanced_content(self, row: pd.Series) -> Dict[str, Any]:
        """Instagram-specific content enhancement"""
        content = str(row.get('instagram_content', '') or '')
        labels = str(row.get('instagram_post_labels_names', '') or '')
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', content)
        mentions = re.findall(r'@\w+', content)
        
        # Create content summary
        content_summary = content
        if not content and labels:
            content_summary = f"Instagram post with labels: {labels}"
        
        # Media type detection
        media_type = str(row.get('instagram_media_type', 'photo')).lower()
        if 'reel' in str(row.get('content_type', '')).lower():
            media_type = 'reel'
        elif 'video' in content.lower():
            media_type = 'video'
        
        return {
            'content_summary': content_summary,
            'hashtags': hashtags,
            'mentions': mentions,
            'hashtag_count': len(hashtags),
            'mention_count': len(mentions),
            'media_type': media_type
        }
    
    def _extract_tiktok_enhanced_content(self, row: pd.Series) -> Dict[str, Any]:
        """TikTok-specific content enhancement"""
        content = str(row.get('tiktok_content', '') or '')
        labels = str(row.get('tiktok_post_labels_names', '') or '')
        
        # Extract content themes
        themes = []
        if '#skincare' in content.lower():
            themes.append('skincare')
        if '#makeup' in content.lower():
            themes.append('makeup')
        if '#beauty' in content.lower():
            themes.append('beauty')
        if '#tutorial' in content.lower():
            themes.append('tutorial')
        
        # Duration categorization
        duration = float(row.get('duration_seconds', 0) or 0)
        if duration <= 15:
            duration_range = "0-15s"
        elif duration <= 30:
            duration_range = "16-30s"
        elif duration <= 60:
            duration_range = "31-60s"
        else:
            duration_range = "60s+"
        
        # Content summary
        content_summary = content
        if not content and labels:
            content_summary = f"TikTok video: {labels}"
        
        # Hashtag analysis
        hashtags = re.findall(r'#\w+', content)
        
        return {
            'content_themes': ', '.join(themes) if themes else 'general',
            'content_summary': content_summary,
            'duration_range': duration_range,
            'hashtags': hashtags,
            'hashtag_count': len(hashtags),
            'is_tutorial': 'tutorial' in content.lower(),
            'is_grwm': 'grwm' in content.lower() or 'get ready with me' in content.lower()
        }
    
    def _extract_customer_care_enhanced_content(self, row: pd.Series) -> Dict[str, Any]:
        """Customer care specific content enhancement"""
        # Generate content summary
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
        
        content_summary = " | ".join(parts) if parts else "Customer support case"
        
        # Detect urgency indicators
        urgent_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'critical']
        text = ' '.join([
            str(row.get('subject', '')),
            str(row.get('description', '')),
            str(row.get('resolution', ''))
        ]).lower()
        
        is_urgent = any(kw in text for kw in urgent_keywords)
        
        # Extract mentioned products/brands
        brands_mentioned = []
        products_mentioned = []
        
        brand_patterns = [
            r'\b(Sephora|Benefit|Fenty|Charlotte Tilbury|Urban Decay|Dior|MAC)\b'
        ]
        product_patterns = [
            r'\b(foundation|lipstick|mascara|serum|moisturizer|eyeshadow|concealer)\b'
        ]
        
        for pattern in brand_patterns:
            brands_mentioned.extend(re.findall(pattern, text, re.IGNORECASE))
        
        for pattern in product_patterns:
            products_mentioned.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return {
            'content_summary': content_summary,
            'is_urgent': is_urgent,
            'brands_mentioned': list(set(brands_mentioned)),
            'products_mentioned': list(set(products_mentioned)),
            'text_length': len(text),
            'has_resolution': pd.notna(row.get('resolution'))
        }
    
    def _extract_countries(self, df: pd.DataFrame, country_config: Dict[str, Any]) -> None:
        """Extract countries from URLs and add derived_country column to dataframe"""
        source_field = country_config.get('source_field')
        target_field = country_config.get('target_field', 'derived_country')
        mapping = country_config.get('mapping', {})
        
        if source_field not in df.columns:
            print(f"⚠️ Country extraction: source field '{source_field}' not found")
            df[target_field] = None
            return
        
        def extract_country_from_url(url):
            """Extract country from URL using handle mapping"""
            if pd.isna(url) or not isinstance(url, str):
                return None
            
            # Extract handle from URL (e.g., @sephorafrance from tiktok.com/@sephorafrance/video/...)
            if '/@' in url:
                handle = '@' + url.split('/@')[1].split('/')[0]
                return mapping.get(handle, None)
            
            return None
        
        # Apply country extraction
        df[target_field] = df[source_field].apply(extract_country_from_url)
        
        # Log results
        country_counts = df[target_field].value_counts()
        print(f"✅ Country extraction completed:")
        for country, count in country_counts.items():
            print(f"   {country}: {count} posts")
        
        if df[target_field].isna().sum() > 0:
            print(f"   Unknown: {df[target_field].isna().sum()} posts")


# Convenience functions for backward compatibility
def extract_entities(df: pd.DataFrame, platform: str) -> Dict[str, List[Dict[str, Any]]]:
    """Extract entities for specified platform"""
    extractor = UnifiedEntityExtractor(platform)
    return extractor.extract_entities(df)


def generate_entity_uuids(entities: Dict[str, List[Dict]], platform: str) -> Dict[str, Dict[str, str]]:
    """Generate UUIDs for entities"""
    extractor = UnifiedEntityExtractor(platform)
    return extractor.generate_entity_uuids(entities)


def map_relationships(row: pd.Series, entity_uuids: Dict[str, Dict[str, str]], platform: str) -> Dict[str, Any]:
    """Map row to entity relationships"""
    extractor = UnifiedEntityExtractor(platform)
    return extractor.map_relationships(row, entity_uuids)
