# Step 1: Corrected Baseline Reproduction

## Purpose

Reproduce the paper's five baseline model families on MIMIC-IV using the corrected cohort pipeline. This is an adapted MIMIC-IV reproduction, not an exact numerical replication of the paper's private NHIS development setup.

## Input

- `data/raw/MIMIC_IV.zip`, or the archive selected by `MIMIC_IV_ZIP`;
- the pinned model implementations under `external/aki-timeseries-continuous-monitoring`;
- `configs/baseline.yaml` and `configs/paths.yaml`.

## Method

The pipeline streams the required MIMIC-IV 3.1 tables, constructs an adult ICU cohort, applies the project AKI labeling logic, and creates a seed-42 subject-level 70/15/15 split. Urine-output events are bounded to their ICU stay by `intime <= charttime <= outtime` before KDIGO staging.

It prepares static and temporal inputs and trains logistic regression, XGBoost, LSTM-Attention, Masked CNN, and ITE-Transformer at 0, 48, and 72 hour prediction horizons.

## Run

From the repository root:

```powershell
python scripts/check_data.py
python Step1_Baseline_Reproduction/scripts/prepare_mimic_events.py
python Step1_Baseline_Reproduction/scripts/build_mimic_cohort.py
python Step1_Baseline_Reproduction/scripts/run_all_baselines.py --scope full --horizons 0 48 72 --device cuda
python Step1_Baseline_Reproduction/scripts/finalize_model_artifacts.py
python Step1_Baseline_Reproduction/scripts/summarize_baselines.py --scope full
python Step1_Baseline_Reproduction/scripts/validate_baselines.py
python Step1_Baseline_Reproduction/scripts/build_artifact_manifest.py
```

`run_all_baselines.py` builds missing model inputs for each requested horizon before training. Use `AKI_WORK_DIR` to redirect regenerated data, models, and logs from the default `work/` location.

## Output

- `checkpoints/`: 15 formal model binaries with configuration, feature order, train-derived scaling information, preprocessing bundles where applicable, and training history;
- `metrics/`: validated aggregate metrics for all formal experiments;
- `metadata/artifact_manifest.csv`: authoritative experiment-to-artifact mapping;
- regenerated patient-level inputs and predictions under ignored `work/` paths.

## Validation

`validate_baselines.py` checks the 15 experiment outputs and recomputes metrics. `build_artifact_manifest.py` records artifact hashes and the fixed test-cohort identity. The released test cohort contains 2,168 stays and 124 AKI cases; its canonical SHA-256 is `d56d72c99d71e56e5144b30f7f0a4238f670dad4a24dd739d3ef3b13c7222723`.

## Notes

The released checkpoint snapshot contains only corrected formal runs. Fresh training writes below `work/Step1_Baseline_Reproduction` and does not overwrite the published snapshot.
