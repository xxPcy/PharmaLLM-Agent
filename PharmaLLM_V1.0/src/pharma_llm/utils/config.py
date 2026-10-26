from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or validated."""


def load_yaml_config(path: str | Path, *, required: bool = True) -> ConfigDict:
    """Load a YAML config file as a dictionary."""

    config_path = Path(path)
    if not config_path.exists():
        if required:
            raise ConfigError(f"Config file does not exist: {config_path}")
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {config_path}")
    return data


def deep_update(base: ConfigDict, updates: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge updates into base and return base."""

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_merged_configs(*paths: str | Path) -> ConfigDict:
    """Load and recursively merge multiple YAML config files."""

    merged: ConfigDict = {}
    for path in paths:
        deep_update(merged, load_yaml_config(path))
    return merged


def get_required(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Read a required dotted key from a nested mapping."""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        current = current[part]
    return current

