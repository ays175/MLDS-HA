#!/usr/bin/env python3
"""
Global Configuration System
Centralized configuration management with environment variable support
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dotenv import load_dotenv

load_dotenv()


class GlobalConfig:
    """Centralized configuration management with environment variable support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv('MLDS_CONFIG', 'config/global.yaml')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable overrides"""
        config_path = Path(self.config_file)
        
        # Default configuration
        default_config = {
            'processing': {
                'batch_size': int(os.getenv('MLDS_BATCH_SIZE', '1000')),
                'sentiment_threshold': float(os.getenv('MLDS_SENTIMENT_THRESHOLD', '0.8')),
                'semantic_min_documents': int(os.getenv('MLDS_SEMANTIC_MIN_DOCS', '100')),
                'max_topics': int(os.getenv('MLDS_MAX_TOPICS', '50')),
                'topic_quality_threshold': float(os.getenv('MLDS_TOPIC_QUALITY', '0.3'))
            },
            'weaviate': {
                'host': os.getenv('WEAVIATE_HOST', 'localhost'),
                'port': int(os.getenv('WEAVIATE_PORT', '8080')),
                'grpc_port': int(os.getenv('WEAVIATE_GRPC_PORT', '50051')),
                'timeout': int(os.getenv('WEAVIATE_TIMEOUT', '30'))
            },
            'paths': {
                'data_dir': os.getenv('MLDS_DATA_DIR', 'data'),
                'metrics_dir': os.getenv('MLDS_METRICS_DIR', 'metrics'),
                'insights_dir': os.getenv('MLDS_INSIGHTS_DIR', 'insights'),
                'temp_dir': os.getenv('MLDS_TEMP_DIR', '/tmp/mlds')
            },
            'models': {
                'sentiment_model': os.getenv('MLDS_SENTIMENT_MODEL', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
                'embedding_model': os.getenv('MLDS_EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
                'clustering_algorithm': os.getenv('MLDS_CLUSTERING', 'kmeans'),
                'max_clusters': int(os.getenv('MLDS_MAX_CLUSTERS', '20'))
            },
            'collections': {
                'facebook': 'FacebookPost',
                'instagram': 'InstagramPost', 
                'tiktok': 'TikTokPost',
                'customer_care': 'CustomerCareCase'
            }
        }
        
        # Load from file if exists
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                # Merge with defaults
                config = self._deep_merge(default_config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
                config = default_config
        else:
            config = default_config
            
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'processing.batch_size')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set final value
        config[keys[-1]] = value
    
    def get_collection_name(self, platform: str) -> str:
        """Get Weaviate collection name for platform"""
        return self.get(f'collections.{platform}', f'{platform.title()}Post')
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.get('processing', {})
    
    def get_weaviate_config(self) -> Dict[str, Any]:
        """Get Weaviate connection configuration"""
        return self.get('weaviate', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.get('models', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.get('paths', {})
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save(self, config_file: Optional[str] = None) -> None:
        """Save current configuration to file"""
        file_path = config_file or self.config_file
        config_path = Path(file_path)
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, indent=2, default_flow_style=False)


# Global instance
_global_config = None

def get_global_config() -> GlobalConfig:
    """Get global configuration instance (singleton)"""
    global _global_config
    if _global_config is None:
        _global_config = GlobalConfig()
    return _global_config


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    return get_global_config().get(key_path, default)


def set_config_value(key_path: str, value: Any) -> None:
    """Convenience function to set configuration value"""
    get_global_config().set(key_path, value)


# Configuration constants (for backward compatibility)
class ConfigConstants:
    """Configuration constants derived from global config"""
    
    @staticmethod
    def batch_size() -> int:
        return get_config_value('processing.batch_size', 1000)
    
    @staticmethod
    def sentiment_threshold() -> float:
        return get_config_value('processing.sentiment_threshold', 0.8)
    
    @staticmethod
    def semantic_min_docs() -> int:
        return get_config_value('processing.semantic_min_documents', 100)
    
    @staticmethod
    def max_topics() -> int:
        return get_config_value('processing.max_topics', 50)
    
    @staticmethod
    def weaviate_host() -> str:
        return get_config_value('weaviate.host', 'localhost')
    
    @staticmethod
    def weaviate_port() -> int:
        return get_config_value('weaviate.port', 8080)
    
    @staticmethod
    def metrics_dir() -> str:
        return get_config_value('paths.metrics_dir', 'metrics')
    
    @staticmethod
    def insights_dir() -> str:
        return get_config_value('paths.insights_dir', 'insights')


if __name__ == "__main__":
    # Test the configuration system
    config = GlobalConfig()
    
    print("Current configuration:")
    print(f"  Batch size: {config.get('processing.batch_size')}")
    print(f"  Weaviate host: {config.get('weaviate.host')}")
    print(f"  Metrics dir: {config.get('paths.metrics_dir')}")
    print(f"  Facebook collection: {config.get_collection_name('facebook')}")
    
    # Test environment variable override
    os.environ['MLDS_BATCH_SIZE'] = '2000'
    config2 = GlobalConfig()
    print(f"  Batch size with env override: {config2.get('processing.batch_size')}")
