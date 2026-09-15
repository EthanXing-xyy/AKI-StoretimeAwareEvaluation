"""Validate the local course-provided MIMIC-IV archive without extracting it."""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from project_paths import MIMIC_IV_ZIP  # noqa: E402


REQUIRED_MEMBERS = (
    "hosp/admissions.csv",
    "hosp/patients.csv",
    "hosp/labevents.csv",
    "hosp/d_labitems.csv",
    "icu/icustays.csv",
    "icu/chartevents.csv",
    "icu/outputevents.csv",
    "icu/d_items.csv",
)


def matches(names: list[str], suffix: str) -> list[str]:
    return [name for name in names if name.endswith(suffix) or name.endswith(f"{suffix}.gz")]


def main() -> int:
    archive = MIMIC_IV_ZIP
    print(f"Archive path: {archive}")
    print(f"File exists: {archive.is_file()}")
    if not archive.is_file():
        print("\nMIMIC_IV.zip not found.\n")
        print("Copy the course-provided archive to:\n")
        print(REPO_ROOT / "data" / "raw" / "MIMIC_IV.zip")
        print("\nor set the MIMIC_IV_ZIP environment variable.")
        return 1

    print(f"Archive size: {archive.stat().st_size:,} bytes")
    try:
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
    except zipfile.BadZipFile:
        print("ERROR: the file is not a readable ZIP archive.")
        return 2

    missing = []
    for suffix in REQUIRED_MEMBERS:
        found = matches(names, suffix)
        status = "PASS" if len(found) == 1 else "FAIL"
        print(f"[{status}] {suffix}: {len(found)} matching member(s)")
        if len(found) != 1:
            missing.append(suffix)

    versions = sorted({match.group(1) for name in names if (match := re.search(r"/mimiciv/(\d+(?:\.\d+)+)/", name))})
    print(f"Detected MIMIC-IV version: {', '.join(versions) if len(versions) == 1 else 'not uniquely identifiable'}")
    if missing:
        print("ERROR: required members are missing or ambiguous: " + ", ".join(missing))
        return 3
    print("Data preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
