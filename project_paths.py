"""Repository-relative paths shared by the public workflow.

The raw archive defaults to ``data/raw/MIMIC_IV.zip`` and can be overridden
with ``MIMIC_IV_ZIP``. Regenerated data and runtime outputs live under
``work/`` by default and can be redirected with ``AKI_WORK_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MIMIC_IV_ZIP = Path(os.environ.get("MIMIC_IV_ZIP", RAW_DATA_DIR / "MIMIC_IV.zip")).expanduser().resolve()
EXTERNAL_DIR = REPO_ROOT / "external"
UPSTREAM_DIR = EXTERNAL_DIR / "aki-timeseries-continuous-monitoring"
WORK_DIR = Path(os.environ.get("AKI_WORK_DIR", REPO_ROOT / "work")).expanduser().resolve()


def repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative configuration or manifest path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0].lower() == "work":
        return (WORK_DIR.joinpath(*parts[1:])).resolve()
    return (REPO_ROOT / path).resolve()
