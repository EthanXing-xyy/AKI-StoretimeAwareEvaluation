"""Read required MIMIC-IV tables directly from the course archive."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd


def find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ZIP member ending in {suffix!r}, found {matches}")
    return matches[0]


def read_table(zip_path: Path, suffix: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(find_member(archive, suffix)) as handle:
            return pd.read_csv(handle, **kwargs)


def iter_table(
    zip_path: Path,
    suffix: str,
    chunksize: int,
    **kwargs,
) -> Iterator[pd.DataFrame]:
    archive = zipfile.ZipFile(zip_path)
    handle = archive.open(find_member(archive, suffix))
    try:
        yield from pd.read_csv(handle, chunksize=chunksize, **kwargs)
    finally:
        handle.close()
        archive.close()
