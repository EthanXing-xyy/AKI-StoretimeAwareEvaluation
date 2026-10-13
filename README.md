**English** | [中文](README_zh.md)

# AKI Prediction: Storetime-aware Continuous Monitoring Study

> **Dataset setup:** This repository intentionally does not include the course-provided MIMIC-IV archive. Copy `MIMIC_IV.zip` to `data/raw/MIMIC_IV.zip` before reproducing the workflow. See [data/README.md](data/README.md).

## Overview

This repository is a clean, reproducible release of an adapted MIMIC-IV acute kidney injury (AKI) prediction study. It reproduces five baseline architectures, evaluates their frozen predictions under simulated continuous monitoring, audits laboratory result-availability timing, and performs a frozen-model storetime-aware stress test.

## Research question

Does the use of laboratory `charttime`, rather than a proxy for result availability (`storetime`), materially affect retrospective continuous AKI prediction evaluation?

## Scientific workflow

1. [Step 1 — Baseline reproduction](Step1_Baseline_Reproduction/README.md): build the corrected MIMIC-IV cohort and train 5 models at 0, 48, and 72 hour prediction horizons.
2. [Step 2 — Continuous monitoring](Step2_Continuous_Monitoring/README.md): freeze the 15 formal models, preprocessing, thresholds, split, and cohort; score seven monitoring time points.
3. [Step 3 — Data availability audit](Step3_Data_Availability_Audit/README.md): quantify the mismatch between laboratory `charttime` and `storetime` without inference or retraining.
4. [Step 4 — Storetime-aware stress test](Step4_Storetime_Aware_Stress_Test/README.md): repeat frozen-model inference while requiring both `charttime <= cutoff` and `storetime <= cutoff` for laboratory rows.
5. [Result](Result/README.md): a compact set of final tables and figures.

## Key finding

Substantial laboratory result-availability mismatch was observed, but storetime-aware frozen-model stress testing showed little material effect on discrimination for most evaluated models.

This finding is deliberately limited. `storetime` is a proxy for result availability, not the exact moment a clinician viewed a result. The stress test does not prove that temporal leakage is absent, that timing never matters, or that other EHR models are robust.

## Repository structure

```text
References/                         Reference papers
data/raw/                           Local archive location (ignored by Git)
Step1_Baseline_Reproduction/        Corrected static baseline pipeline and artifacts
Step2_Continuous_Monitoring/        Frozen-model monitoring evaluation
Step3_Data_Availability_Audit/      charttime/storetime data audit
Step4_Storetime_Aware_Stress_Test/  Frozen-model stress test
Result/                             Compact validated findings
external/                           Pinned official upstream working tree
work/                               Regenerated local outputs (ignored by Git)
```

## Dataset setup

This repository intentionally does not include the course-provided MIMIC-IV archive.

Copy the course-provided:

```text
MIMIC_IV.zip
```

to:

```text
data/raw/MIMIC_IV.zip
```

The corrected executable pipeline uses this archive only. `physionet.org.zip` is not a formal pipeline requirement. To keep the archive elsewhere, set an environment variable in PowerShell:

```powershell
$env:MIMIC_IV_ZIP="D:\somewhere\MIMIC_IV.zip"
python scripts/check_data.py
```

The script never downloads data or contacts PhysioNet.

## Environment setup

The validated environment used Python 3.11.0, PyTorch 2.8.0+cu128, and CUDA 12.8. Exact Python package versions are listed in `requirements.txt`.

The 15 formal checkpoint binaries use Git LFS. Install Git LFS before cloning or checking out the repository, then enable it once per user account with `git lfs install`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_data.py
```

CPU is sufficient for extraction, auditing, logistic regression, and most reporting. XGBoost can run on CPU. CUDA is strongly recommended for LSTM-Attention, Masked CNN, ITE-Transformer, and repeated continuous inference.

## Reproduction

Run commands from the repository root. Scripts resolve paths from their own location, not the current working directory. Large regenerated artifacts go to `work/`; set `AKI_WORK_DIR` to redirect them.

### Step 1

```powershell
python Step1_Baseline_Reproduction/scripts/prepare_mimic_events.py
python Step1_Baseline_Reproduction/scripts/build_mimic_cohort.py
python Step1_Baseline_Reproduction/scripts/run_all_baselines.py --scope full --horizons 0 48 72 --device cuda
python Step1_Baseline_Reproduction/scripts/finalize_model_artifacts.py
python Step1_Baseline_Reproduction/scripts/summarize_baselines.py --scope full
python Step1_Baseline_Reproduction/scripts/validate_baselines.py
python Step1_Baseline_Reproduction/scripts/build_artifact_manifest.py
```

### Step 2

```powershell
python Step2_Continuous_Monitoring/scripts/run_monitoring.py --stage dry-run
python Step2_Continuous_Monitoring/scripts/run_monitoring.py --stage full
```

For memory-constrained systems, run one model/horizon pair per process and merge with `merge_monitoring_runs.py`.

### Step 3

```powershell
python Step3_Data_Availability_Audit/scripts/01_extract_storetime.py
python Step3_Data_Availability_Audit/scripts/02_delay_distribution.py
python Step3_Data_Availability_Audit/scripts/03_timepoint_pending.py
python Step3_Data_Availability_Audit/scripts/04_onset_creatinine.py
python Step3_Data_Availability_Audit/scripts/05_figures_and_summary.py
```

### Step 4

See [Step4_Storetime_Aware_Stress_Test/README.md](Step4_Storetime_Aware_Stress_Test/README.md) for the required null-control gate and sequential frozen-inference commands.

## Results

Validated, aggregate results are in [Result](Result/README.md). No raw MIMIC data or patient-level predictions are included.

## Reference papers

See [References/README.md](References/README.md).

## Upstream code

The official repository at `https://github.com/Jovinus/aki-timeseries-continuous-monitoring` is included as a working-tree snapshot at commit `6794acb7f615b5864171f8aa7c8ae610049cb947`. Its nested `.git` directory is intentionally excluded. See [external/README.md](external/README.md).

## Licensing

- Project-authored code is available under the [MIT License](LICENSE).
- Papers under `References/` retain their original publication licenses; the project MIT License does not cover them.
- Code in `external/` retains the upstream project's original license; the project MIT License does not replace or extend that license.

## Reproducibility notes

- The public baseline is the corrected run after bounding urine-output rows to each ICU stay (`intime <= charttime <= outtime`).
- The frozen test cohort contains 2,168 stays and 124 AKI cases; canonical SHA-256: `d56d72c99d71e56e5144b30f7f0a4238f670dad4a24dd739d3ef3b13c7222723`.
- Exactly 15 formal checkpoints are included. Non-formal and obsolete models are excluded.
- Re-running Step 1 writes models and patient-level artifacts under ignored `work/`; it does not overwrite the released checkpoint snapshot.

## Limitations

This is an adapted MIMIC-IV reproduction, not an exact numerical replication of the original paper's private NHIS development setup. The study is single-dataset and retrospective. `storetime` is only a result-availability proxy, and the frozen-model design does not estimate performance after retraining on availability-aware inputs.
