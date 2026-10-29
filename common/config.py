"""
common/config.py
=================
Single source of truth for loading the project's YAML configuration and
setting up a consistent logger. Every module imports `get_config()` /
`get_logger()` from here instead of re-implementing config/log setup.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    Load and cache the project YAML configuration.

    Args:
        config_path: Optional override path to a YAML config file. If not
            provided, falls back to the CONFIG_PATH env var, then the
            default `configs/config.yaml`.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the resolved config path does not exist.
    """
    path = Path(config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_logger(name: str) -> logging.Logger:
    """
    Create (or fetch) a module-level logger configured with the project's
    log level and a consistent format.

    Args:
        name: Usually `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance.
    """
    cfg = get_config()
    level_name = cfg.get("project", {}).get("log_level", "INFO")
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level_name, logging.INFO))
        logger.propagate = False
    return logger
