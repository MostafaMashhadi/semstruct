"""YAML configuration. Dataset files override configs/default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from semstruct.paths import project_root


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(dataset: str, config_path: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    with (root / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    dataset_path = root / "configs" / "datasets" / f"{dataset}.yaml"
    if dataset_path.is_file():
        with dataset_path.open(encoding="utf-8") as handle:
            config = deep_merge(config, yaml.safe_load(handle) or {})
    if config_path is not None:
        with Path(config_path).open(encoding="utf-8") as handle:
            config = deep_merge(config, yaml.safe_load(handle) or {})
    config["dataset"] = dataset
    return config
