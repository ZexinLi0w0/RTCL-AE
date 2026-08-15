"""
Configuration file loader utilities.
"""

import os
import yaml
import json
from typing import Dict, List, Any, Optional

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as config_file:
        try:
            config = json.load(config_file)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON configuration: {e}")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file (YAML or JSON).
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    file_extension = os.path.splitext(config_path)[1].lower()
    
    if file_extension in ['.yaml', '.yml']:
        return load_yaml_config(config_path)
    elif file_extension == '.json':
        return load_json_config(config_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_extension}")

def validate_batch_config(config: Dict[str, Any]) -> bool:
    """
    Validate batch configuration structure.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, raises ValueError otherwise
    """
    # Check if batches key exists
    if 'batches' not in config:
        raise ValueError("Configuration missing 'batches' key")
    
    batches = config['batches']
    if not isinstance(batches, list) or len(batches) == 0:
        raise ValueError("'batches' must be a non-empty list")
    
    # Check each batch configuration
    for i, batch in enumerate(batches):
        required_keys = ['index', 'train_batch_size', 'eval_batch_size']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Batch at index {i} missing required key: {key}")
        
        # Validate types
        if not isinstance(batch['train_batch_size'], int) or batch['train_batch_size'] <= 0:
            raise ValueError(f"Batch at index {i} has invalid train_batch_size: {batch['train_batch_size']}")
        
        if not isinstance(batch['eval_batch_size'], int) or batch['eval_batch_size'] <= 0:
            raise ValueError(f"Batch at index {i} has invalid eval_batch_size: {batch['eval_batch_size']}")
    
    return True

def get_config_value(config: Dict[str, Any], key: str, default: Any) -> Any:
    """
    Get a value from the configuration, or return a default if not found.
    
    Args:
        config: Configuration dictionary
        key: Key to look up
        default: Default value to return if key is not found
        
    Returns:
        Value from configuration or default
    """
    return config.get(key, default) 