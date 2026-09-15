"""Read-only five-model forward-pass check for the refactored release paths.

Set ``AKI_VALIDATION_BASELINE`` to a local completed corrected Baseline directory.
The script reads one held-out processed input, loads the released 0 h artifacts,
runs inference on CPU, prints results, and writes nothing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Step2_Continuous_Monitoring" / "scripts"))

from project_paths import repo_path  # noqa: E402
import cm_adapter as cm  # noqa: E402


def main() -> int:
    source_value = os.environ.get("AKI_VALIDATION_BASELINE")
    if not source_value:
        print("AKI_VALIDATION_BASELINE is not set; point it to a completed corrected Baseline directory.")
        return 2
    source = Path(source_value).expanduser().resolve()
    processed = source / "data" / "processed" / "mimiciv" / "full" / "horizon_0"
    master = pd.read_parquet(processed / "master.parquet")
    stay_id = int(master.loc[master.split.eq("test"), "stay_id"].iloc[0])
    features = list(cm.import_baseline(REPO_ROOT / "Step1_Baseline_Reproduction").config.ALL_FEATURES)
    flat = pd.read_parquet(processed / "flat_features.parquet").set_index("stay_id")
    with np.load(processed / "sequences" / f"{stay_id}.npz") as arrays:
        inputs = {
            stay_id: {
                "flat": flat.loc[stay_id, features].to_dict(),
                "irregular": arrays["irregular"].copy(),
                "cnn": arrays["cnn"].copy(),
                "demo": arrays["demo"].copy(),
            }
        }

    paths, _ = cm.load_config()
    baseline = cm.import_baseline(paths["baseline_root"])
    manifest = pd.read_csv(REPO_ROOT / "Step1_Baseline_Reproduction" / "metadata" / "artifact_manifest.csv", dtype=str, keep_default_na=False)
    manifest = manifest[manifest.prediction_window_hours.eq("0")]
    rows = []
    for artifact in manifest.to_dict("records"):
        for column in cm.PATH_COLUMNS:
            artifact[column] = str(repo_path(artifact[column])) if artifact[column] else ""
        model = cm.FrozenModel(artifact, baseline, paths, "cpu")
        probability = float(model.predict([stay_id], inputs)[0])
        integrity = model.integrity_after()
        rows.append(
            {
                "model": artifact["model"],
                "stay_id_redacted": True,
                "probability_finite": bool(np.isfinite(probability) and 0 <= probability <= 1),
                "parameters_unchanged": integrity["parameters_unchanged"],
                "artifact_files_unchanged": integrity["artifact_files_unchanged"],
            }
        )
    passed = len(rows) == 5 and all(row["probability_finite"] and row["parameters_unchanged"] and row["artifact_files_unchanged"] for row in rows)
    print(json.dumps({"passed": passed, "models_checked": len(rows), "checks": rows}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
