# Dataset setup

This repository intentionally excludes the course-provided MIMIC-IV archive.

Expected default path:

```text
data/raw/MIMIC_IV.zip
```

Copy the archive supplied for the course into that location. Do not rename its internal files and do not extract the entire archive manually. The pipeline streams only required CSV members.

Alternatively, point to an existing copy in PowerShell:

```powershell
$env:MIMIC_IV_ZIP="D:\datasets\MIMIC_IV.zip"
```

Validate the archive with:

```powershell
python scripts/check_data.py
```

The check reports the resolved path, existence, size, required members, and MIMIC-IV version when it can be determined reliably. It never downloads data and never contacts credential-protected PhysioNet endpoints.

The corrected pipeline requires `MIMIC_IV.zip` only. `physionet.org.zip` is not a formal pipeline requirement.

Never commit raw MIMIC data. `data/raw/*` is ignored except for `.gitkeep`.
