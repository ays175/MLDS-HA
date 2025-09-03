"""
Metrics Configuration Management

This module handles YAML configuration loading and management for metrics generation.
It replaces the _load_platform_config method and centralizes all configuration logic.

Functions moved here:
- _load_platform_config (line 1221 from original)
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsConfig:
    """
    Centralized configuration management for metrics generation.
    
    This class handles loading platform-specific YAML configurations
    and provides easy access to metrics settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or "/Users/ays/mlds_ha/ingestion/configs"
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all platform configurations at startup."""
        platforms = ['tiktok', 'facebook', 'instagram', 'customer_care']
        
        for platform in platforms:
            try:
                config_file = Path(self.config_path) / f"{platform}.yaml"
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        self.configs[platform] = yaml.safe_load(f)
                    logger.info(f"✅ Loaded config for {platform}")
                else:
                    logger.warning(f"⚠️ Config file not found: {config_file}")
                    self.configs[platform] = {}
            except Exception as e:
                logger.error(f"❌ Error loading {platform} config: {e}")
                self.configs[platform] = {}
    
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """
        Get complete platform configuration.
        
        Original method: _load_platform_config (line 1221)
        
        Args:
            platform: Platform name (tiktok, facebook, instagram, customer_care)
            
        Returns:
            Dict containing platform configuration
        """
        return self.configs.get(platform, {})
    
    def get_metrics_config(self, platform: str) -> Dict[str, Any]:
        """Get metrics-specific configuration for a platform."""
        platform_config = self.get_platform_config(platform)
        return platform_config.get('metrics', {})
    
    def get_enabled_analyzers(self, platform: str) -> list:
        """Get list of enabled analyzers for a platform."""
        metrics_config = self.get_metrics_config(platform)
        return metrics_config.get('analyzers', [
            'brand_performance', 'content_type_performance', 'sentiment_analysis',
            'engagement_metrics', 'temporal_analysis', 'platform_specific'
        ])
    
    def get_analyzer_config(self, platform: str, analyzer_name: str) -> Dict[str, Any]:
        """Get configuration for a specific analyzer."""
        metrics_config = self.get_metrics_config(platform)
        return metrics_config.get(analyzer_name, {})
    
    def get_export_config(self, platform: str) -> Dict[str, Any]:
        """Get export configuration for a platform."""
        metrics_config = self.get_metrics_config(platform)
        return {
            'output_directory': metrics_config.get('output_directory', f'metrics/{platform}/latest/'),
            'archive_directory': metrics_config.get('archive_directory', f'metrics/{platform}/archived/'),
            'formats': metrics_config.get('export_formats', ['json'])
        }
    
    def get_field_mappings(self, platform: str) -> Dict[str, str]:
        """Get platform-specific field mappings."""
        platform_config = self.get_platform_config(platform)
        schema = platform_config.get('schema', {})
        
        # Extract key field mappings
        mappings = {}
        
        # Date field mapping
        if 'created_time' in schema:
            mappings['date_field'] = 'created_time'
        elif 'timestamp' in schema:
            mappings['date_field'] = 'timestamp'
        else:
            mappings['date_field'] = '_date'
        
        # Sentiment field mapping
        sentiment_fields = [f for f in schema.keys() if 'sentiment' in f.lower()]
        if sentiment_fields:
            mappings['sentiment_field'] = sentiment_fields[0]
        
        # Platform-specific ID field
        id_fields = [f for f in schema.keys() if f.endswith('_id')]
        if id_fields:
            mappings['id_field'] = id_fields[0]
        
        return mappings
    
    def validate_config(self, platform: str) -> bool:
        """Validate platform configuration."""
        config = self.get_platform_config(platform)
        
        # Basic validation
        if not config:
            logger.warning(f"⚠️ Empty configuration for {platform}")
            return False
        
        # Check required sections
        required_sections = ['schema']
        for section in required_sections:
            if section not in config:
                logger.warning(f"⚠️ Missing {section} in {platform} config")
                return False
        
        logger.info(f"✅ Configuration valid for {platform}")
        return True


# Global config instance
_config_instance = None

def get_config() -> MetricsConfig:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = MetricsConfig()
    return _config_instance
