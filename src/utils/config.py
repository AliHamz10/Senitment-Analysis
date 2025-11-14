"""
Configuration Management
Loads and manages application configuration from YAML/JSON files.

Created by Ali Hamza & Zarmeena Jawad
"""
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Try to import yaml, but make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class Config:
    """Configuration manager for the application."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self._load_defaults()
            self._load_from_file()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self._config = {
            'camera': {
                'backend': 'avfoundation',  # macOS native
                'width': 1280,
                'height': 720,
                'fps': 30,
                'warmup_frames': 10,
                'prefer_builtin': True,  # Prefer Mac built-in camera
                'skip_continuity': True,  # Skip Continuity Camera (iPhone)
            },
            'model': {
                'mode': 'image',
                'model_name': 'joeddav/distilbert-base-uncased-go-emotions',
                'dl_backend': 'onnx',
                'recent_decay': 0.85,
                'bbox_lerp': 0.22,
            },
            'ui': {
                'colors': {
                    'neon_cyan': [200, 220, 255],
                    'neon_blue': [160, 90, 255],
                    'neon_accent': [180, 140, 255],
                },
                'fps_position': [10, 28],
                'header_position': [20, 40],
                'status_panel_position': [10, 60],
            },
            'persona': {
                'joy': 'AI Dreamer',
                'surprise': 'Curious Synth',
                'anger': 'Chrome Rebel',
                'sadness': 'Neon Loner',
                'confused': 'Quantum Puzzler',
                'happy': 'Sunset Coder',
                'excited': 'Pulse Rider',
                'fear': 'Circuit Warden',
                'disgust': 'Acid Critic',
                'neutral': 'Calm Sentinel',
            },
            'paths': {
                'assets': 'assets',
                'screenshots': 'screenshots',
                'models': 'assets',
            },
        }
    
    def _load_from_file(self):
        """Load configuration from file if it exists."""
        config_paths = [
            Path('config/config.yaml'),
            Path('config/config.json'),
            Path('config.yaml'),
            Path('config.json'),
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    if config_path.suffix == '.yaml':
                        if HAS_YAML:
                            with open(config_path, 'r') as f:
                                file_config = yaml.safe_load(f) or {}
                        else:
                            print("Warning: PyYAML not installed, skipping YAML config. Install with: pip install pyyaml")
                            continue
                    else:
                        with open(config_path, 'r') as f:
                            file_config = json.load(f) or {}
                    
                    # Merge with defaults
                    self._config = self._deep_merge(self._config, file_config)
                    break
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'camera.width')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'camera.width')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()

