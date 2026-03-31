import yaml
import os
from pathlib import Path


class Config:
    """Configuration loader for the project."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def project(self):
        return self._config.get('project', {})
    
    @property
    def data(self):
        return self._config.get('data', {})
    
    @property
    def model(self):
        return self._config.get('model', {})
    
    @property
    def training(self):
        return self._config.get('training', {})
    
    @property
    def hyperparams(self):
        return self._config.get('hyperparams', {})
    
    @property
    def api(self):
        return self._config.get('api', {})
    
    @property
    def risk(self):
        return self._config.get('risk', {})
    
    def get_path(self, key: str, *keys) -> Path:
        """Get a path from config, joining multiple keys."""
        base = self._config.get(key, {})
        for k in keys:
            base = base.get(k, '')
        return Path(base)
    
    def get_project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent


def get_config() -> Config:
    """Get configuration singleton."""
    return Config()
