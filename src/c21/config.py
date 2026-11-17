"""Load configs/default.yaml and resolve paths from the repository root."""

from __future__ import annotations

from pathlib import Path

import yaml


def repo_root() -> Path:
    """Folder that contains pyproject.toml (works from any working directory)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config. Entries under `paths` become absolute Path objects."""
    root = repo_root()
    cfg_path = Path(path) if path else root / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"] = {
        key: Path(value) if Path(value).is_absolute() else (root / value).resolve()
        for key, value in cfg["paths"].items()
    }
    cfg["config_path"] = cfg_path.resolve()
    return cfg
