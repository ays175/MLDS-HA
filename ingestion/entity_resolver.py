#!/usr/bin/env python3
"""
Entity Conflict Resolution System
Resolves conflicts between conservative and enhanced entity extraction
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

# Add parent directory to path for config import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.global_config import get_global_config


@dataclass
class EntityVersion:
    """Represents a versioned entity with extraction metadata"""
    entity_id: str
    name: str
    entity_type: str
    confidence: float
    extraction_method: str  # 'conservative', 'enhanced', 'manual'
    source_text: str
    extracted_at: datetime
    platform: str
    alternatives: List[str] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


@dataclass
class EntityConflict:
    """Represents a conflict between entity extractions"""
    conflict_id: str
    conservative_entity: EntityVersion
    enhanced_entity: EntityVersion
    conflict_type: str  # 'name_mismatch', 'type_mismatch', 'multiple_matches'
    similarity_score: float
    resolution_status: str  # 'pending', 'auto_resolved', 'manual_required'
    resolved_entity: Optional[EntityVersion] = None
    resolution_notes: str = ""


class EntityConflictResolver:
    """Resolves conflicts between different entity extraction methods"""
    
    def __init__(self):
        self.config = get_global_config()
        self.similarity_threshold = self.config.get('entity_resolution.similarity_threshold', 0.8)
        self.confidence_threshold = self.config.get('entity_resolution.confidence_threshold', 0.8)
        self.auto_resolve_threshold = self.config.get('entity_resolution.auto_resolve_threshold', 0.9)
    
    def resolve_entity_conflicts(self, 
                                conservative_entities: List[Dict], 
                                enhanced_entities: List[Dict],
                                platform: str) -> Tuple[List[EntityVersion], List[EntityConflict]]:
        """Resolve conflicts between conservative and enhanced extraction"""
        
        # Convert to EntityVersion objects
        conservative_versions = self._convert_to_versions(conservative_entities, 'conservative', platform)
        enhanced_versions = self._convert_to_versions(enhanced_entities, 'enhanced', platform)
        
        resolved_entities = []
        conflicts = []
        
        # Process each enhanced entity
        for enhanced in enhanced_versions:
            matches = self._find_potential_matches(enhanced, conservative_versions)
            
            if not matches:
                # New entity discovered by enhanced extraction
                if enhanced.confidence > self.confidence_threshold:
                    resolved_entities.append(self._mark_as_new_discovery(enhanced))
                else:
                    # Low confidence new entity - flag for review
                    conflict = EntityConflict(
                        conflict_id=self._generate_conflict_id(enhanced, None),
                        conservative_entity=None,
                        enhanced_entity=enhanced,
                        conflict_type='low_confidence_new',
                        similarity_score=0.0,
                        resolution_status='manual_required',
                        resolution_notes=f"Low confidence ({enhanced.confidence:.2f}) new entity"
                    )
                    conflicts.append(conflict)
            
            elif len(matches) == 1:
                # Single match - attempt automatic resolution
                conservative = matches[0]
                similarity = self._calculate_similarity(conservative.name, enhanced.name)
                
                if similarity > self.auto_resolve_threshold:
                    # High similarity - auto merge
                    merged = self._merge_entities(conservative, enhanced)
                    resolved_entities.append(merged)
                else:
                    # Moderate similarity - flag for review
                    conflict = EntityConflict(
                        conflict_id=self._generate_conflict_id(conservative, enhanced),
                        conservative_entity=conservative,
                        enhanced_entity=enhanced,
                        conflict_type='name_mismatch',
                        similarity_score=similarity,
                        resolution_status='manual_required',
                        resolution_notes=f"Name similarity {similarity:.2f} below auto-resolve threshold"
                    )
                    conflicts.append(conflict)
            
            else:
                # Multiple matches - always flag for manual resolution
                conflict = EntityConflict(
                    conflict_id=self._generate_conflict_id(enhanced, matches[0]),
                    conservative_entity=matches[0],  # Primary match
                    enhanced_entity=enhanced,
                    conflict_type='multiple_matches',
                    similarity_score=max(self._calculate_similarity(enhanced.name, m.name) for m in matches),
                    resolution_status='manual_required',
                    resolution_notes=f"Enhanced entity matches {len(matches)} conservative entities"
                )
                conflicts.append(conflict)
        
        # Add unmatched conservative entities
        matched_conservative_ids = {c.conservative_entity.entity_id for c in conflicts if c.conservative_entity}
        for conservative in conservative_versions:
            if conservative.entity_id not in matched_conservative_ids:
                resolved_entities.append(conservative)
        
        return resolved_entities, conflicts
    
    def _convert_to_versions(self, entities: List[Dict], method: str, platform: str) -> List[EntityVersion]:
        """Convert entity dictionaries to EntityVersion objects"""
        versions = []
        
        for i, entity in enumerate(entities):
            version = EntityVersion(
                entity_id=f"{platform}_{method}_{i}_{entity.get('name', 'unknown')}",
                name=entity.get('name', ''),
                entity_type=entity.get('type', 'unknown'),
                confidence=float(entity.get('confidence', 0.0)),
                extraction_method=method,
                source_text=entity.get('source_text', ''),
                extracted_at=datetime.now(),
                platform=platform,
                alternatives=entity.get('alternatives', [])
            )
            versions.append(version)
        
        return versions
    
    def _find_potential_matches(self, enhanced: EntityVersion, conservative_list: List[EntityVersion]) -> List[EntityVersion]:
        """Find potential matches for an enhanced entity in conservative list"""
        matches = []
        
        for conservative in conservative_list:
            # Same entity type
            if conservative.entity_type != enhanced.entity_type:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(conservative.name, enhanced.name)
            
            if similarity > self.similarity_threshold:
                matches.append(conservative)
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: self._calculate_similarity(x.name, enhanced.name), reverse=True)
        
        return matches
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two entity names"""
        if not name1 or not name2:
            return 0.0
        
        # Normalize names
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Sequence similarity
        seq_sim = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Check if one is contained in the other
        if norm1 in norm2 or norm2 in norm1:
            containment_bonus = 0.2
        else:
            containment_bonus = 0.0
        
        return min(1.0, seq_sim + containment_bonus)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison"""
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common suffixes/prefixes
        suffixes = ['inc', 'corp', 'ltd', 'llc', 'co', 'company']
        for suffix in suffixes:
            if normalized.endswith(f' {suffix}'):
                normalized = normalized[:-len(suffix)-1].strip()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _mark_as_new_discovery(self, enhanced: EntityVersion) -> EntityVersion:
        """Mark an enhanced entity as a new discovery"""
        enhanced.extraction_method = 'enhanced_new'
        return enhanced
    
    def _merge_entities(self, conservative: EntityVersion, enhanced: EntityVersion) -> EntityVersion:
        """Merge conservative and enhanced entities"""
        # Use conservative as base (higher confidence in naming)
        merged = EntityVersion(
            entity_id=conservative.entity_id,
            name=conservative.name,  # Keep conservative name as primary
            entity_type=conservative.entity_type,
            confidence=max(conservative.confidence, enhanced.confidence),
            extraction_method='merged',
            source_text=conservative.source_text,
            extracted_at=conservative.extracted_at,
            platform=conservative.platform,
            alternatives=list(set(conservative.alternatives + [enhanced.name] + enhanced.alternatives))
        )
        
        return merged
    
    def _generate_conflict_id(self, entity1: Optional[EntityVersion], entity2: Optional[EntityVersion]) -> str:
        """Generate unique conflict ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if entity1 and entity2:
            return f"conflict_{entity1.platform}_{timestamp}_{hash(entity1.name + entity2.name) % 10000}"
        elif entity1:
            return f"conflict_{entity1.platform}_{timestamp}_{hash(entity1.name) % 10000}"
        else:
            return f"conflict_unknown_{timestamp}"
    
    def save_conflicts(self, conflicts: List[EntityConflict], output_dir: str = "data/entity_conflicts") -> str:
        """Save conflicts to JSON file for manual review"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"entity_conflicts_{timestamp}.json"
        
        # Convert to serializable format
        conflicts_data = []
        for conflict in conflicts:
            conflict_dict = {
                "conflict_id": conflict.conflict_id,
                "conflict_type": conflict.conflict_type,
                "similarity_score": conflict.similarity_score,
                "resolution_status": conflict.resolution_status,
                "resolution_notes": conflict.resolution_notes,
                "conservative_entity": self._entity_to_dict(conflict.conservative_entity) if conflict.conservative_entity else None,
                "enhanced_entity": self._entity_to_dict(conflict.enhanced_entity),
                "resolved_entity": self._entity_to_dict(conflict.resolved_entity) if conflict.resolved_entity else None
            }
            conflicts_data.append(conflict_dict)
        
        with open(filename, 'w') as f:
            json.dump(conflicts_data, f, indent=2, default=str)
        
        return str(filename)
    
    def _entity_to_dict(self, entity: EntityVersion) -> Dict[str, Any]:
        """Convert EntityVersion to dictionary"""
        return {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "confidence": entity.confidence,
            "extraction_method": entity.extraction_method,
            "source_text": entity.source_text,
            "extracted_at": entity.extracted_at.isoformat(),
            "platform": entity.platform,
            "alternatives": entity.alternatives
        }
    
    def get_resolution_summary(self, resolved: List[EntityVersion], conflicts: List[EntityConflict]) -> Dict[str, Any]:
        """Generate summary of resolution process"""
        return {
            "total_resolved": len(resolved),
            "total_conflicts": len(conflicts),
            "auto_resolved": len([c for c in conflicts if c.resolution_status == 'auto_resolved']),
            "manual_required": len([c for c in conflicts if c.resolution_status == 'manual_required']),
            "conflict_types": {
                "name_mismatch": len([c for c in conflicts if c.conflict_type == 'name_mismatch']),
                "multiple_matches": len([c for c in conflicts if c.conflict_type == 'multiple_matches']),
                "low_confidence_new": len([c for c in conflicts if c.conflict_type == 'low_confidence_new'])
            },
            "extraction_methods": {
                "conservative": len([e for e in resolved if e.extraction_method == 'conservative']),
                "enhanced": len([e for e in resolved if e.extraction_method == 'enhanced']),
                "merged": len([e for e in resolved if e.extraction_method == 'merged']),
                "enhanced_new": len([e for e in resolved if e.extraction_method == 'enhanced_new'])
            }
        }


def main():
    """CLI interface for entity conflict resolution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entity conflict resolution system")
    parser.add_argument("--conservative", required=True, help="Path to conservative entities JSON")
    parser.add_argument("--enhanced", required=True, help="Path to enhanced entities JSON") 
    parser.add_argument("--platform", required=True, help="Platform name")
    parser.add_argument("--output-dir", default="data/entity_conflicts", help="Output directory for conflicts")
    
    args = parser.parse_args()
    
    # Load entities
    with open(args.conservative, 'r') as f:
        conservative_entities = json.load(f)
    
    with open(args.enhanced, 'r') as f:
        enhanced_entities = json.load(f)
    
    # Resolve conflicts
    resolver = EntityConflictResolver()
    resolved, conflicts = resolver.resolve_entity_conflicts(
        conservative_entities, enhanced_entities, args.platform
    )
    
    # Save conflicts for manual review
    if conflicts:
        conflicts_file = resolver.save_conflicts(conflicts, args.output_dir)
        print(f"💾 Saved {len(conflicts)} conflicts to: {conflicts_file}")
    
    # Print summary
    summary = resolver.get_resolution_summary(resolved, conflicts)
    print("\n📊 Entity Resolution Summary:")
    print(f"  • Total resolved: {summary['total_resolved']}")
    print(f"  • Conflicts requiring manual review: {summary['manual_required']}")
    print(f"  • Auto-resolved: {summary['auto_resolved']}")
    print(f"  • Merged entities: {summary['extraction_methods']['merged']}")
    print(f"  • New discoveries: {summary['extraction_methods']['enhanced_new']}")


if __name__ == "__main__":
    main()
