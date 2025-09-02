# Hardcoded Values Audit & Entity Conflict Resolution

## 🚨 Current Hardcoded Values Found

### 1. Collection Names (CRITICAL)
```python
# ❌ HARDCODED in unified_schema.py
collection_name = "FacebookPost"  # Should be from config
collection_name = "SemanticTopic"  # Should be configurable

# ✅ SOLUTION: Make all collection names configurable
collections:
  facebook_post: "FacebookPost"  # Can be changed per deployment
  semantic_topic: "SemanticTopic"
  topic_mapping: "PostTopicMapping"
```

### 2. Processing Thresholds (MEDIUM)
```python
# ❌ HARDCODED in unified_ingest.py
if len(df) > 10:  # Arbitrary threshold
sample_size = min(5000, len(df))  # Magic number
batch_size = 100  # Fixed batch size

# ✅ SOLUTION: Move to configuration
processing:
  semantic_analysis:
    min_documents: 20  # Configurable minimum
    max_sample_size: 5000  # Configurable sample
    batch_size: 100  # Configurable batch size
```

### 3. Sentiment Thresholds (MEDIUM)
```python
# ❌ HARDCODED in metrics export
if sentiment > 0.3:  # Arbitrary positive threshold
    category = "positive"
elif sentiment < -0.3:  # Arbitrary negative threshold
    category = "negative"

# ✅ SOLUTION: Configuration-driven thresholds
sentiment_analysis:
  thresholds:
    very_positive: 0.6
    positive: 0.2
    neutral_min: -0.2
    neutral_max: 0.2
    negative: -0.6
```

### 4. File Paths (HIGH)
```python
# ❌ HARDCODED paths
output_dir = Path("metrics/facebook")  # Fixed path
log_file = "data/facebook_ingestion_log.json"  # Fixed path

# ✅ SOLUTION: Environment variables + config
paths:
  metrics_base_dir: ${METRICS_DIR:-metrics}
  logs_base_dir: ${LOGS_DIR:-logs}
  data_base_dir: ${DATA_DIR:-data}
```

### 5. Model Parameters (LOW)
```python
# ❌ HARDCODED in sentiment analyzer
max_length = 512  # Token limit
confidence_threshold = 0.7  # Model confidence

# ✅ SOLUTION: Model configuration
models:
  sentiment:
    max_length: 512
    confidence_threshold: 0.7
    model_name: "cardiffnlp/twitter-roberta-base-sentiment-latest"
```

## 🔧 Comprehensive Hardcoded Values Fix

### 1. Create Global Configuration
```python
# config/global_config.py (NEW FILE)
import os
from pathlib import Path
from typing import Dict, Any
import yaml

class GlobalConfig:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.getenv('MLDS_CONFIG', 'config/global.yaml')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with environment variable substitution"""
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            return self._get_default_config()
        
        with open(config_path, 'r') as f:
            config_text = f.read()
        
        # Replace environment variables
        import re
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        config_text = re.sub(r'\$\{([^}:]+)(?::([^}]*))?\}', replace_env_var, config_text)
        
        return yaml.safe_load(config_text)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration if no file exists"""
        return {
            "paths": {
                "metrics_base_dir": os.getenv("METRICS_DIR", "metrics"),
                "logs_base_dir": os.getenv("LOGS_DIR", "logs"),
                "data_base_dir": os.getenv("DATA_DIR", "data"),
                "config_base_dir": os.getenv("CONFIG_DIR", "ingestion/configs")
            },
            "processing": {
                "default_batch_size": int(os.getenv("BATCH_SIZE", "100")),
                "max_sample_size": int(os.getenv("MAX_SAMPLE_SIZE", "5000")),
                "min_documents_for_semantic": int(os.getenv("MIN_DOCS_SEMANTIC", "20"))
            },
            "sentiment": {
                "thresholds": {
                    "very_positive": float(os.getenv("SENTIMENT_VERY_POS", "0.6")),
                    "positive": float(os.getenv("SENTIMENT_POS", "0.2")),
                    "neutral_min": float(os.getenv("SENTIMENT_NEU_MIN", "-0.2")),
                    "neutral_max": float(os.getenv("SENTIMENT_NEU_MAX", "0.2")),
                    "negative": float(os.getenv("SENTIMENT_NEG", "-0.6"))
                }
            },
            "models": {
                "sentiment": {
                    "max_length": int(os.getenv("SENTIMENT_MAX_LEN", "512")),
                    "confidence_threshold": float(os.getenv("SENTIMENT_CONF", "0.7")),
                    "model_name": os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
                }
            },
            "collections": {
                "semantic_topic": os.getenv("SEMANTIC_TOPIC_COLLECTION", "SemanticTopic"),
                "topic_mapping": os.getenv("TOPIC_MAPPING_COLLECTION", "PostTopicMapping"),
                "topic_alignment": os.getenv("TOPIC_ALIGNMENT_COLLECTION", "SemanticTopicAlignment")
            }
        }
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_path(self, path_key: str) -> Path:
        """Get path configuration as Path object"""
        path_str = self.get(f"paths.{path_key}")
        return Path(path_str) if path_str else Path(".")

# Global config instance
global_config = GlobalConfig()
```

### 2. Update All Files to Use Configuration
```python
# Example: unified_ingest.py modifications
from config.global_config import global_config

class UnifiedIngestionEngine:
    def __init__(self, platform: str, config_path: Optional[str] = None):
        # ... existing code ...
        self.global_config = global_config
        self.batch_size = self.global_config.get('processing.default_batch_size', 100)
        self.max_sample_size = self.global_config.get('processing.max_sample_size', 5000)
    
    def _run_semantic_analysis(self, client, df, dataset_id):
        # ✅ NO MORE HARDCODED VALUES
        min_docs = self.global_config.get('processing.min_documents_for_semantic', 20)
        sample_size = min(self.max_sample_size, len(df))
        
        if len(df) < min_docs:
            print(f"⚠️ Not enough content for semantic analysis (need {min_docs})")
            return {"status": "insufficient_data"}
```

## 🤝 Entity Conflict Resolution Strategy

### Problem: Enhanced vs Conservative Entity Extraction

**Conservative Extraction (Fast Mode):**
- High confidence only (>90%)
- Pattern-based matching
- Known entity databases
- Example: "Nike" found in labels

**Enhanced Extraction (Batch Mode):**
- ML-based NER models
- Lower confidence threshold (>70%)
- Context-aware extraction
- Example: "swoosh brand" → "Nike"

### Conflict Resolution Approach

#### 1. Entity Versioning System
```python
# Add to entity collections
class EntityVersion:
    entity_id: str          # Unique entity ID
    extraction_method: str  # "conservative", "enhanced", "manual"
    confidence: float       # Extraction confidence
    source_record_id: str   # Which record it came from
    created_at: datetime    # When discovered
    validated: bool         # Human validation status
    canonical: bool         # Is this the canonical version?

# Example entity evolution:
entities = [
    {
        "entity_id": "brand_001",
        "name": "Nike",
        "extraction_method": "conservative",
        "confidence": 0.95,
        "canonical": True,
        "source": "facebook_post_labels"
    },
    {
        "entity_id": "brand_001", 
        "name": "Nike Inc",  # Enhanced extraction found full name
        "extraction_method": "enhanced",
        "confidence": 0.85,
        "canonical": False,  # Not canonical yet
        "source": "ml_ner_model"
    }
]
```

#### 2. Conflict Resolution Rules
```python
# entity_resolver.py (NEW FILE)
class EntityConflictResolver:
    """Resolves conflicts between different entity extraction methods"""
    
    def resolve_entity_conflicts(self, conservative_entities: List[Dict], 
                                enhanced_entities: List[Dict]) -> List[Dict]:
        """Resolve conflicts between conservative and enhanced extraction"""
        
        resolved = []
        conflicts = []
        
        for enhanced in enhanced_entities:
            # Find potential matches in conservative entities
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
    
    def _find_potential_matches(self, enhanced_entity: Dict, 
                               conservative_entities: List[Dict]) -> List[Dict]:
        """Find potential matches using various strategies"""
        matches = []
        
        # Exact name match
        for conservative in conservative_entities:
            if self._names_match(enhanced_entity['name'], conservative['name']):
                matches.append(conservative)
        
        # Fuzzy name match (for typos, variations)
        if not matches:
            for conservative in conservative_entities:
                similarity = self._calculate_name_similarity(
                    enhanced_entity['name'], conservative['name']
                )
                if similarity > 0.85:  # High similarity threshold
                    matches.append(conservative)
        
        # Context-based matching (same posts, similar context)
        if not matches:
            matches = self._find_contextual_matches(enhanced_entity, conservative_entities)
        
        return matches
    
    def _merge_entities(self, conservative: Dict, enhanced: Dict) -> Dict:
        """Merge conservative and enhanced entity information"""
        return {
            'entity_id': conservative['entity_id'],
            'name': conservative['name'],  # Keep conservative name (higher confidence)
            'alternative_names': [enhanced['name']] if enhanced['name'] != conservative['name'] else [],
            'extraction_methods': ['conservative', 'enhanced'],
            'confidence': max(conservative['confidence'], enhanced['confidence']),
            'canonical': True,
            'enhanced_attributes': enhanced.get('attributes', {}),
            'validation_status': 'auto_merged'
        }
    
    def _mark_as_new_discovery(self, enhanced: Dict) -> Dict:
        """Mark enhanced entity as new discovery"""
        return {
            **enhanced,
            'extraction_method': 'enhanced_discovery',
            'canonical': True,
            'validation_status': 'needs_review'
        }

# Integration into batch processor
def process_entity_enhancement(self, platform: str) -> Dict[str, Any]:
    """Enhanced entity extraction with conflict resolution"""
    
    # Get records that need entity enhancement
    records = self._get_records_needing_enhancement(platform)
    
    # Run enhanced NER extraction
    enhanced_entities = self._run_enhanced_ner(records)
    
    # Get existing conservative entities
    conservative_entities = self._get_existing_entities(platform)
    
    # Resolve conflicts
    resolver = EntityConflictResolver()
    resolved_entities, conflicts = resolver.resolve_entity_conflicts(
        conservative_entities, enhanced_entities
    )
    
    # Update entities in Weaviate
    self._update_entities(resolved_entities)
    
    # Log conflicts for manual review
    if conflicts:
        self._log_entity_conflicts(conflicts, platform)
    
    return {
        "status": "success",
        "entities_processed": len(enhanced_entities),
        "entities_merged": len(resolved_entities),
        "conflicts_flagged": len(conflicts),
        "manual_review_needed": len(conflicts) > 0
    }
```

#### 3. Manual Review Interface
```python
# entity_review.py (NEW FILE)
class EntityReviewInterface:
    """Interface for manual entity conflict resolution"""
    
    def get_pending_conflicts(self) -> List[Dict]:
        """Get entities needing manual review"""
        # Query entities with validation_status = 'needs_review'
        pass
    
    def resolve_conflict(self, conflict_id: str, resolution: Dict):
        """Apply manual resolution to entity conflict"""
        # Update entity with human decision
        # Options: merge, keep_separate, mark_duplicate, delete
        pass
    
    def export_conflicts_for_review(self, output_file: str):
        """Export conflicts to CSV for batch review"""
        conflicts = self.get_pending_conflicts()
        df = pd.DataFrame(conflicts)
        df.to_csv(output_file, index=False)
```

## 🎯 Implementation Priority

### Phase 1: Remove Hardcoded Values (Week 1)
1. Create global configuration system
2. Update all files to use configuration
3. Add environment variable support
4. Test with different configurations

### Phase 2: Entity Conflict Resolution (Week 2)
1. Implement entity versioning
2. Create conflict resolution rules
3. Add manual review interface
4. Test with sample conflicts

### Phase 3: Monitoring & Validation (Week 3)
1. Add entity quality metrics
2. Implement conflict detection alerts
3. Create entity validation workflows
4. Performance testing

This approach ensures no hardcoded values remain and provides a robust system for handling entity conflicts between conservative and enhanced extraction methods.
