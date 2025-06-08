"""
Configuration management module for chess puzzle rating prediction.

This module provides functions to load, validate, and access configuration settings
from YAML or JSON files. It centralizes all configuration parameters used across
the project to make the codebase more maintainable and flexible.
"""

import os
import yaml
import json
import jsonschema
from typing import Dict, Any, Optional, Union


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file (YAML or JSON)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing configuration parameters

    Raises
    ------
    ConfigurationError
        If the file cannot be loaded or has an invalid format
    """
    if not os.path.exists(config_path):
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        file_ext = os.path.splitext(config_path)[1].lower()
        with open(config_path, 'r') as f:
            if file_ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_ext == '.json':
                config = json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {file_ext}")
        return config
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigurationError(f"Error parsing configuration file: {e}")


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Validate configuration against a JSON schema.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate
    schema : Dict[str, Any]
        JSON schema to validate against

    Raises
    ------
    ConfigurationError
        If the configuration is invalid
    """
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ConfigurationError(f"Invalid configuration: {e}")


# Default configuration schema
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "data_paths": {
            "type": "object",
            "properties": {
                "train_file": {"type": "string"},
                "test_file": {"type": "string"},
                "submission_file": {"type": "string"}
            },
            "required": ["train_file", "test_file", "submission_file"]
        },
        "logging": {
            "type": "object",
            "properties": {
                "log_dir": {"type": "string"},
                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "log_to_console": {"type": "boolean"},
                "log_to_file": {"type": "boolean"},
                "log_file_name": {"type": ["string", "null"]}
            }
        },
        "progress_tracking": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "log_interval": {"type": "integer", "minimum": 1},
                "store_metrics": {"type": "boolean"}
            }
        },
        "dashboards": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "output_dir": {"type": "string"},
                "auto_generate": {"type": "boolean"}
            }
        },
        "training": {
            "type": "object",
            "properties": {
                "n_splits_lgbm": {"type": "integer", "minimum": 1},
                "random_state": {"type": "integer", "minimum": 0},
                "lgbm_early_stopping_rounds": {"type": "integer", "minimum": 1}
            },
            "required": ["n_splits_lgbm", "random_state", "lgbm_early_stopping_rounds"]
        },
        "autoencoder": {
            "type": "object",
            "properties": {
                "epochs": {"type": "integer", "minimum": 1},
                "batch_size_per_gpu": {"type": "integer", "minimum": 1},
                "learning_rate": {"type": "number", "minimum": 0},
                "valid_split": {"type": "number", "minimum": 0, "maximum": 1},
                "early_stopping_patience": {"type": "integer", "minimum": 1},
                "model_save_dir": {"type": "string"},
                "force_retrain": {"type": "boolean"},
                "embedding_dim": {"type": "integer", "minimum": 1},
                "sequence_length": {"type": "integer", "minimum": 1},
                "kl_weight": {"type": "number", "minimum": 0},
                "contrastive_weight": {"type": "number", "minimum": 0}
            },
            "required": ["epochs", "batch_size_per_gpu", "learning_rate", "valid_split", 
                         "early_stopping_patience", "model_save_dir", "force_retrain"]
        },
        "gpu": {
            "type": "object",
            "properties": {
                "target_num_gpus": {"type": "integer", "minimum": 0}
            },
            "required": ["target_num_gpus"]
        },
        "lgbm_params": {
            "type": "object",
            "properties": {
                "objective": {"type": "string"},
                "metric": {"type": "string"},
                "n_estimators": {"type": "integer", "minimum": 1},
                "learning_rate": {"type": "number", "minimum": 0},
                "feature_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                "bagging_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                "bagging_freq": {"type": "integer", "minimum": 0},
                "lambda_l1": {"type": "number", "minimum": 0},
                "lambda_l2": {"type": "number", "minimum": 0},
                "num_leaves": {"type": "integer", "minimum": 2},
                "min_child_samples": {"type": "integer", "minimum": 1},
                "verbose": {"type": "integer"},
                "n_jobs": {"type": "integer"},
                "boosting_type": {"type": "string"}
            },
            "required": ["objective", "metric", "n_estimators", "learning_rate"]
        },
        "feature_engineering": {
            "type": "object",
            "properties": {
                "material_values": {
                    "type": "object",
                    "properties": {
                        "pawn": {"type": "integer"},
                        "knight": {"type": "integer"},
                        "bishop": {"type": "integer"},
                        "rook": {"type": "integer"},
                        "queen": {"type": "integer"},
                        "king": {"type": "integer"}
                    }
                },
                "king_safety_weights": {
                    "type": "object",
                    "properties": {
                        "pawn_shield": {"type": "integer"},
                        "king_attackers": {"type": "integer"},
                        "king_open_files": {"type": "integer"},
                        "castling_bonus": {"type": "integer"}
                    }
                },
                "tactical_advantage_weights": {
                    "type": "object",
                    "properties": {
                        "pins": {"type": "integer"},
                        "forks": {"type": "integer"},
                        "discovered_attacks": {"type": "integer"}
                    }
                },
                "text_vectorization": {
                    "type": "object",
                    "properties": {
                        "themes_min_df": {"type": "integer", "minimum": 1},
                        "openings_min_df": {"type": "integer", "minimum": 1}
                    }
                }
            }
        }
    },
    "required": ["data_paths", "training", "autoencoder", "gpu", "lgbm_params"]
}


# Global configuration object
_config = None


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the configuration. If not already loaded, load it from the specified path.
    If no path is specified, look for a default configuration file.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. If None, look for default locations.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing configuration parameters

    Raises
    ------
    ConfigurationError
        If the configuration cannot be loaded or is invalid
    """
    global _config

    if _config is not None and config_path is None:
        return _config

    if config_path is None:
        # Look for configuration in default locations
        default_locations = [
            os.path.join(os.getcwd(), 'config.yaml'),
            os.path.join(os.getcwd(), 'config.yml'),
            os.path.join(os.getcwd(), 'config.json'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yml'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json'),
        ]

        for loc in default_locations:
            if os.path.exists(loc):
                config_path = loc
                break

        if config_path is None:
            raise ConfigurationError("No configuration file found in default locations")

    _config = load_config(config_path)
    validate_config(_config, CONFIG_SCHEMA)
    return _config


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration values.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing default configuration parameters
    """
    return {
        "data_paths": {
            "train_file": "/raid/sroziewski/chess/training_data_02_01.csv",
            "test_file": "/raid/sroziewski/chess/testing_data_cropped.csv",
            "submission_file": "/raid/sroziewski/chess/submission_lgbm_pt_ae_no_engine_direct_v1.txt"
        },
        "logging": {
            "log_dir": "logs",
            "log_level": "INFO",
            "log_to_console": True,
            "log_to_file": True,
            "log_file_name": None
        },
        "progress_tracking": {
            "enabled": True,
            "log_interval": 10,
            "store_metrics": True
        },
        "dashboards": {
            "enabled": True,
            "output_dir": "dashboards",
            "auto_generate": True
        },
        "training": {
            "n_splits_lgbm": 5,
            "random_state": 42,
            "lgbm_early_stopping_rounds": 100
        },
        "autoencoder": {
            "epochs": 40,
            "batch_size_per_gpu": 2048,
            "learning_rate": 1e-3,
            "valid_split": 0.1,
            "early_stopping_patience": 5,
            "model_save_dir": "trained_models_no_engine_direct_v1",
            "force_retrain": False,
            "embedding_dim": 16,
            "sequence_length": 11,
            "kl_weight": 0.001,
            "contrastive_weight": 0.1
        },
        "gpu": {
            "target_num_gpus": 4
        },
        "lgbm_params": {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "num_leaves": 42,
            "min_child_samples": 20,
            "verbose": -1,
            "n_jobs": -1,
            "boosting_type": "gbdt"
        },
        "feature_engineering": {
            "material_values": {
                "pawn": 1,
                "knight": 3,
                "bishop": 3,
                "rook": 5,
                "queen": 9,
                "king": 0
            },
            "king_safety_weights": {
                "pawn_shield": 2,
                "king_attackers": -3,
                "king_open_files": -2,
                "castling_bonus": 4
            },
            "tactical_advantage_weights": {
                "pins": 1,
                "forks": 2,
                "discovered_attacks": 3
            },
            "text_vectorization": {
                "themes_min_df": 20,
                "openings_min_df": 10
            }
        }
    }


def create_default_config_file(output_path: str, format: str = 'yaml') -> None:
    """
    Create a default configuration file.

    Parameters
    ----------
    output_path : str
        Path where the configuration file will be saved
    format : str, optional
        Format of the configuration file ('yaml' or 'json'), by default 'yaml'

    Raises
    ------
    ConfigurationError
        If the file cannot be created or the format is invalid
    """
    default_config = get_default_config()

    try:
        with open(output_path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            elif format.lower() == 'json':
                json.dump(default_config, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported configuration format: {format}")
        print(f"Default configuration file created at: {output_path}")
    except Exception as e:
        raise ConfigurationError(f"Error creating default configuration file: {e}")
