"""Repository paths. The working root is the directory that contains configs/."""

from __future__ import annotations

import os
from pathlib import Path

DATASETS = ("mondial", "imdb", "wikipedia")


def project_root() -> Path:
    override = os.environ.get("SEMSTRUCT_ROOT")
    if override:
        return Path(override).resolve()
    cwd = Path.cwd()
    if (cwd / "configs" / "default.yaml").is_file():
        return cwd
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs" / "default.yaml").is_file():
        return candidate
    return cwd


def raw_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "raw"


def processed_dir(dataset: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "processed" / dataset


def output_dir(dataset: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "outputs" / dataset


def model_dir(dataset: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "outputs" / "models" / dataset


def resolve_datasets(name: str) -> list[str]:
    if name == "all":
        return list(DATASETS)
    if name not in DATASETS:
        known = ", ".join((*DATASETS, "all"))
        raise SystemExit(f"Unknown dataset {name!r}. Choose one of: {known}.")
    return [name]
