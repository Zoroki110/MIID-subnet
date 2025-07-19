# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Yanez - MIID Team

"""
Configuration Validation Module

This module provides comprehensive configuration validation for the MIID subnet,
ensuring all required settings are properly configured and valid.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import bittensor as bt
from pathlib import Path

from MIID.utils.error_handling import MIIDConfigurationError, ErrorContext


class ConfigValidator:
    """Comprehensive configuration validator for MIID subnet."""
    
    # Required environment variables
    REQUIRED_ENV_VARS = {
        'miner': ['CHUTES_API_KEY'],
        'validator': ['WANDB_API_KEY'],
        'common': []
    }
    
    # Optional environment variables with defaults
    OPTIONAL_ENV_VARS = {
        'MIID_LOG_LEVEL': 'INFO',
        'MIID_MAX_WORKERS': '10',
        'MIID_TIMEOUT': '120',
        'MIID_CACHE_SIZE': '1000'
    }
    
    @classmethod
    def validate_miner_config(cls, config: Any) -> List[str]:
        """
        Validate miner configuration.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        with ErrorContext("Miner Configuration Validation"):
            # Check required attributes
            required_attrs = [
                'wallet', 'subtensor', 'metagraph', 'neuron', 'axon'
            ]
            
            for attr in required_attrs:
                if not hasattr(config, attr):
                    errors.append(f"Missing required configuration attribute: {attr}")
            
            # Validate neuron-specific settings
            if hasattr(config, 'neuron'):
                neuron_errors = cls._validate_neuron_config(config.neuron, 'miner')
                errors.extend(neuron_errors)
            
            # Check environment variables
            env_errors = cls._validate_environment_variables('miner')
            errors.extend(env_errors)
            
            # Validate API settings
            if hasattr(config, 'neuron') and hasattr(config.neuron, 'model_name'):
                if not cls._validate_model_name(config.neuron.model_name):
                    errors.append(f"Invalid model name: {config.neuron.model_name}")
        
        return errors
    
    @classmethod
    def validate_validator_config(cls, config: Any) -> List[str]:
        """
        Validate validator configuration.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        with ErrorContext("Validator Configuration Validation"):
            # Check required attributes
            required_attrs = [
                'wallet', 'subtensor', 'metagraph', 'neuron'
            ]
            
            for attr in required_attrs:
                if not hasattr(config, attr):
                    errors.append(f"Missing required configuration attribute: {attr}")
            
            # Validate neuron-specific settings
            if hasattr(config, 'neuron'):
                neuron_errors = cls._validate_neuron_config(config.neuron, 'validator')
                errors.extend(neuron_errors)
            
            # Check environment variables
            env_errors = cls._validate_environment_variables('validator')
            errors.extend(env_errors)
            
            # Validate wandb settings
            wandb_errors = cls._validate_wandb_config(config)
            errors.extend(wandb_errors)
        
        return errors
    
    @classmethod
    def _validate_neuron_config(cls, neuron_config: Any, neuron_type: str) -> List[str]:
        """Validate neuron-specific configuration."""
        errors = []
        
        # Common neuron settings
        if hasattr(neuron_config, 'name') and not neuron_config.name:
            errors.append("Neuron name cannot be empty")
        
        if hasattr(neuron_config, 'epoch_length'):
            if not isinstance(neuron_config.epoch_length, int) or neuron_config.epoch_length <= 0:
                errors.append("Epoch length must be a positive integer")
        
        if hasattr(neuron_config, 'full_path'):
            path = Path(neuron_config.full_path)
            if not path.parent.exists():
                errors.append(f"Parent directory does not exist: {path.parent}")
        
        # Type-specific validation
        if neuron_type == 'miner':
            if hasattr(neuron_config, 'model_name') and not neuron_config.model_name:
                errors.append("Miner model name cannot be empty")
        
        elif neuron_type == 'validator':
            if hasattr(neuron_config, 'num_concurrent_forwards'):
                if (not isinstance(neuron_config.num_concurrent_forwards, int) or 
                    neuron_config.num_concurrent_forwards <= 0):
                    errors.append("Number of concurrent forwards must be a positive integer")
        
        return errors
    
    @classmethod
    def _validate_environment_variables(cls, component: str) -> List[str]:
        """Validate required environment variables."""
        errors = []
        
        # Check required variables for component
        required_vars = cls.REQUIRED_ENV_VARS.get(component, [])
        required_vars.extend(cls.REQUIRED_ENV_VARS.get('common', []))
        
        for var in required_vars:
            if not os.getenv(var):
                errors.append(f"Missing required environment variable: {var}")
        
        # Validate specific environment variable formats
        if component == 'miner':
            chutes_key = os.getenv('CHUTES_API_KEY')
            if chutes_key and not cls._validate_api_key_format(chutes_key):
                errors.append("CHUTES_API_KEY appears to have invalid format")
        
        if component == 'validator':
            wandb_key = os.getenv('WANDB_API_KEY')
            if wandb_key and not cls._validate_api_key_format(wandb_key):
                errors.append("WANDB_API_KEY appears to have invalid format")
        
        return errors
    
    @classmethod
    def _validate_api_key_format(cls, api_key: str) -> bool:
        """Validate API key format (basic check)."""
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Basic format check: should be alphanumeric and reasonable length
        if len(api_key) < 10 or len(api_key) > 200:
            return False
        
        # Should contain alphanumeric characters
        if not re.match(r'^[a-zA-Z0-9_\-]+$', api_key):
            return False
        
        return True
    
    @classmethod
    def _validate_model_name(cls, model_name: str) -> bool:
        """Validate model name format."""
        if not model_name or not isinstance(model_name, str):
            return False
        
        # Basic format check for model names
        if len(model_name.strip()) == 0:
            return False
        
        # Should not contain dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`']
        if any(char in model_name for char in dangerous_chars):
            return False
        
        return True
    
    @classmethod
    def _validate_wandb_config(cls, config: Any) -> List[str]:
        """Validate Weights & Biases configuration."""
        errors = []
        
        # Check if wandb is disabled
        if hasattr(config, 'wandb') and hasattr(config.wandb, 'off') and config.wandb.off:
            return errors  # No validation needed if disabled
        
        # Check wandb project name
        if hasattr(config, 'wandb') and hasattr(config.wandb, 'project_name'):
            project_name = config.wandb.project_name
            if project_name and not cls._validate_project_name(project_name):
                errors.append(f"Invalid wandb project name: {project_name}")
        
        # Check wandb entity
        if hasattr(config, 'wandb') and hasattr(config.wandb, 'entity'):
            entity = config.wandb.entity
            if entity and not cls._validate_wandb_entity(entity):
                errors.append(f"Invalid wandb entity: {entity}")
        
        return errors
    
    @classmethod
    def _validate_project_name(cls, project_name: str) -> bool:
        """Validate wandb project name format."""
        if not project_name or not isinstance(project_name, str):
            return False
        
        # Project names should be reasonable length and format
        if len(project_name) > 128 or len(project_name) < 1:
            return False
        
        # Should match wandb naming conventions
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', project_name):
            return False
        
        return True
    
    @classmethod
    def _validate_wandb_entity(cls, entity: str) -> bool:
        """Validate wandb entity format."""
        if not entity or not isinstance(entity, str):
            return False
        
        # Entity names should be reasonable length and format
        if len(entity) > 128 or len(entity) < 1:
            return False
        
        # Should match wandb naming conventions
        if not re.match(r'^[a-zA-Z0-9_\-]+$', entity):
            return False
        
        return True
    
    @classmethod
    def validate_network_config(cls, config: Any) -> List[str]:
        """Validate network-related configuration."""
        errors = []
        
        with ErrorContext("Network Configuration Validation"):
            # Validate netuid
            if hasattr(config, 'netuid'):
                if not isinstance(config.netuid, int) or config.netuid < 0:
                    errors.append("Network UID (netuid) must be a non-negative integer")
            
            # Validate subtensor network
            if hasattr(config, 'subtensor'):
                if hasattr(config.subtensor, 'network'):
                    network = config.subtensor.network
                    if network and not cls._validate_network_name(network):
                        errors.append(f"Invalid subtensor network: {network}")
                
                if hasattr(config.subtensor, 'chain_endpoint'):
                    endpoint = config.subtensor.chain_endpoint
                    if endpoint and not cls._validate_endpoint_url(endpoint):
                        errors.append(f"Invalid chain endpoint URL: {endpoint}")
        
        return errors
    
    @classmethod
    def _validate_network_name(cls, network: str) -> bool:
        """Validate network name."""
        valid_networks = ['local', 'finney', 'test', 'archive']
        return network.lower() in valid_networks
    
    @classmethod
    def _validate_endpoint_url(cls, url: str) -> bool:
        """Validate endpoint URL format."""
        if not url or not isinstance(url, str):
            return False
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^(ws|wss|http|https)://'  # protocol
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))
    
    @classmethod
    def validate_logging_config(cls, config: Any) -> List[str]:
        """Validate logging configuration."""
        errors = []
        
        with ErrorContext("Logging Configuration Validation"):
            if hasattr(config, 'logging'):
                # Validate logging directory
                if hasattr(config.logging, 'logging_dir'):
                    log_dir = Path(config.logging.logging_dir)
                    if not log_dir.exists():
                        try:
                            log_dir.mkdir(parents=True, exist_ok=True)
                        except Exception as e:
                            errors.append(f"Cannot create logging directory: {e}")
                
                # Validate log level
                if hasattr(config.logging, 'level'):
                    level = config.logging.level
                    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                    if level and level.upper() not in valid_levels:
                        errors.append(f"Invalid log level: {level}")
        
        return errors
    
    @classmethod
    def validate_all(cls, config: Any, component_type: str) -> Tuple[bool, List[str]]:
        """
        Validate all configuration aspects.
        
        Args:
            config: Configuration object to validate
            component_type: Type of component ('miner' or 'validator')
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors = []
        
        try:
            # Component-specific validation
            if component_type == 'miner':
                all_errors.extend(cls.validate_miner_config(config))
            elif component_type == 'validator':
                all_errors.extend(cls.validate_validator_config(config))
            else:
                all_errors.append(f"Unknown component type: {component_type}")
            
            # Common validations
            all_errors.extend(cls.validate_network_config(config))
            all_errors.extend(cls.validate_logging_config(config))
            
        except Exception as e:
            all_errors.append(f"Configuration validation failed: {e}")
        
        is_valid = len(all_errors) == 0
        return is_valid, all_errors
    
    @classmethod
    def apply_defaults(cls, config: Any) -> None:
        """Apply default values for optional configuration settings."""
        
        # Apply optional environment variable defaults
        for env_var, default_value in cls.OPTIONAL_ENV_VARS.items():
            if not os.getenv(env_var):
                os.environ[env_var] = default_value
        
        # Apply neuron defaults if needed
        if hasattr(config, 'neuron'):
            if not hasattr(config.neuron, 'epoch_length'):
                config.neuron.epoch_length = 360
            
            if not hasattr(config.neuron, 'events_retention_size'):
                config.neuron.events_retention_size = "2 GB"
        
        bt.logging.info("Applied default configuration values")


def validate_and_setup_config(config: Any, component_type: str) -> None:
    """
    Validate configuration and raise exception if invalid.
    
    Args:
        config: Configuration object to validate
        component_type: Type of component ('miner' or 'validator')
        
    Raises:
        MIIDConfigurationError: If configuration is invalid
    """
    # Apply defaults first
    ConfigValidator.apply_defaults(config)
    
    # Validate configuration
    is_valid, errors = ConfigValidator.validate_all(config, component_type)
    
    if not is_valid:
        error_message = f"Configuration validation failed for {component_type}:\n"
        error_message += "\n".join(f"- {error}" for error in errors)
        raise MIIDConfigurationError(error_message)
    
    bt.logging.info(f"{component_type.capitalize()} configuration validated successfully")