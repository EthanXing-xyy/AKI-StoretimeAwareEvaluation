# Agent Guide

Instructions for Codex and other coding agents working in this repository.

## Research question

The primary question is whether laboratory result-availability timing (`charttime` versus `storetime`) materially affects retrospective continuous AKI prediction evaluation in MIMIC-IV.

## Scientific workflow

1. `Step1_Baseline_Reproduction`: corrected MIMIC-IV cohort construction, feature preparation, patient-level split, and formal training of logistic regression, XGBoost, LSTM-Attention, Masked CNN, and ITE-Transformer at 0, 48, and 72 hour horizons.
2. `Step2_Continuous_Monitoring`: simulated monitoring at -72, -60, -48, -36, -24, -12, and 0 hours using frozen Step 1 artifacts.
3. `Step3_Data_Availability_Audit`: a read-only audit of laboratory `charttime`/`storetime` mismatch. No model inference or training belongs here.
4. `Step4_Storetime_Aware_Stress_Test`: frozen-model paired comparison of charttime-only versus storetime-aware laboratory availability.
5. `Result`: compact aggregate outputs for readers.

## Non-negotiable scientific constraints

- Use only the corrected baseline in this release. Its urine-output logic bounds rows to the ICU interval.
- The corrected test cohort has 2,168 stays, 124 AKI cases, and SHA-256 `d56d72c99d71e56e5144b30f7f0a4238f670dad4a24dd739d3ef3b13c7222723`.
- Never restore obsolete baseline artifacts, non-formal checkpoints, or removed exploratory analyses.
- Step 2 and Step 4 must never retrain, refit preprocessing, create a new split, tune thresholds on the test set, or change the held-out cohort.
- Step 4 changes one core rule only: a laboratory row is usable when `charttime <= cutoff AND storetime <= cutoff`. Vitals retain charttime behavior.
- Keep the `onset_creatinine_only` mechanism experiment.
- `storetime` is a proxy for result availability, not the exact clinician-view time.
- Do not claim that temporal leakage is absent or that storetime has no effect.

## Data

The course-provided archive must be placed at `data/raw/MIMIC_IV.zip`. It is intentionally absent and ignored by Git. `MIMIC_IV_ZIP` may override the location. Do not download from PhysioNet or embed credentials. The corrected pipeline does not require `physionet.org.zip`.

Run:

```powershell
python scripts/check_data.py
```

## Upstream source

`external/aki-timeseries-continuous-monitoring` is a clean snapshot of the authors' official repository at commit `6794acb7f615b5864171f8aa7c8ae610049cb947` (`6794acb`). Do not edit it. Implement adaptations in the numbered step wrappers. The Step 1 model loaders and Step 2/Step 4 reference-time, clinical-faithfulness, and alert-burden components reuse upstream modules through adapters.

## Artifacts

Precomputed and committed:

- 15 corrected formal checkpoint files and their preprocessing/configuration artifacts;
- aggregate Step 1 metrics;
- aggregate Step 2 monitoring metrics and validation gates;
- aggregate Step 3 audit tables and figures;
- aggregate Step 4 paired comparisons, bootstrap intervals, figures, and protocol notes;
- compact final `Result` tables and figures.

Regenerated locally under ignored `work/`:

- extracted/interim/processed MIMIC data;
- subject split files and patient-level cohorts;
- patient-level static and monitoring predictions;
- runtime logs, caches, and audit row tables;
- newly trained model runs.

Do not commit raw archives, patient-level data, predictions, or generated runtime logs.

## Compute expectations

- CPU-only: data preflight, extraction, cohort audit, Step 3 aggregation, reporting, logistic regression, and XGBoost if configured for CPU.
- GPU recommended: three deep-learning training jobs and full Step 2/Step 4 inference. The validated host used PyTorch 2.8.0+cu128 with CUDA 12.8.

## Reproduction order

1. Read `README.md` and run `python scripts/check_data.py`.
2. Run the Step 1 extraction, corrected cohort build, model-input build/training, finalization, summary, validation, and runtime manifest scripts.
3. Confirm the 15-artifact manifest and corrected cohort hash before Step 2.
4. Run the Step 2 dry-run validation gate, then all formal model/horizon pairs and merge/report.
5. Run Step 3 extraction and aggregate audit scripts.
6. Run Step 4 charttime null controls first. Only then run storetime and onset-creatinine-only variants, paired comparison, and report generation.
7. Verify `Result` remains derived from validated aggregate tables.

## Working rules

- Resolve paths from repository files using `pathlib`; never add personal drive paths.
- Default runtime storage is `work/`; `AKI_WORK_DIR` may override it.
- Preserve the official upstream tree byte-for-byte.
- Prefer wrappers and adapters over edits to upstream source.
- Keep public prose conservative: “Substantial laboratory result-availability mismatch was observed, but storetime-aware frozen-model stress testing showed little material effect on discrimination for most evaluated models.”
