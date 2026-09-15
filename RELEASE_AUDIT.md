# Clean Release Audit

## 1. Scope and safety

- Source directory: `A:\AKIPrediction`
- Clean release directory: `A:\AKIPrediction_GitHub`
- Audit date: 2026-09-15 (Asia/Shanghai)
- The source directory was read and copied only. No source file was moved, deleted, overwritten, or edited.
- No GitHub upload, Git push, remote creation, repository creation, or release publication was performed.
- The clean release is initialized as a single local Git repository; no nested repository or remote is present.

## 2. Source-to-destination mapping

| Source | Destination | Selection rule |
|---|---|---|
| Reference papers | `References` | Both PDF references copied byte-for-byte; English index added |
| corrected `Baseline/scripts`, `Baseline/src`, `Baseline/configs` | `Step1_Baseline_Reproduction` | Corrected cohort/training pipeline only; paths adapted to the release root |
| corrected `Baseline/models/*/horizon_*/full` | `Step1_Baseline_Reproduction/checkpoints` | The 15 manifest-authorized formal experiments only |
| corrected Baseline aggregate metrics and metadata | `Step1_Baseline_Reproduction/metrics`, `metadata` | Aggregate metrics, cohort summary, feature mapping, manifest, checkpoint inventory |
| validated `ContinuousMonitoring` code and aggregate outputs | `Step2_Continuous_Monitoring` | Main monitoring path, seven time points, alert burden, clinical faithfulness, rankings, validation gates |
| validated `DataAvailabilityAudit` code and aggregate outputs | `Step3_Data_Availability_Audit` | Join integrity, delay, pending-lab, onset-creatinine, and decision-criteria main analyses |
| validated `StoretimeStressTest` code and aggregate outputs | `Step4_Storetime_Aware_Stress_Test` | Charttime null control, storetime comparison, onset-creatinine mechanism, bootstrap, alerts, ranking, protocol notes |
| selected validated aggregate outputs | `Result` | Five reader-facing figures and three concise tables |
| official upstream tracked working tree | `external/aki-timeseries-continuous-monitoring` | All 113 tracked files at the pinned commit; nested `.git` excluded |

All copied project scripts were cleaned to use repository-relative `pathlib` paths. `project_paths.py` centralizes the repository, archive, upstream, and working-output locations.

## 3. Explicit exclusions

The following were intentionally not copied:

- `.venv`, caches, runtime logs, temporary DuckDB files, and debug output;
- `archive/Baseline_pre_fix` and all obsolete artifacts associated with the urine-output ICU-boundary bug;
- the obsolete test cohort and hash beginning `de496`;
- smoke-test checkpoints and dry-run outputs;
- old comparison tables, old cohort artifacts, pre/post comparison outputs, and historical diagnostic scripts;
- large interim/processed datasets, split files, raw patient-level cohorts, NPZ sequences, and patient-level predictions;
- the regenerable Step 3 event-SQL consistency table, because its validation rows contained stay identifiers;
- the exploratory `complete_trajectory` / complete-trajectory analysis, cohort composition discussion, changing risk set discussion, the 265-stay analysis, landmarking discussion, and their dedicated scripts/tables/figures;
- `physionet.org.zip`, because executable dependency scanning showed that the corrected scientific pipeline does not read the notes archive;
- `MIMIC_IV.zip`, because course data must not be distributed or committed;
- the unrelated local `external/mimic-code` checkout, which the corrected executable baseline does not import or read.

The corrected urine-output condition retained in Step 1 is:

```sql
WHERE u.charttime >= s.intime AND u.charttime <= s.outtime
```

## 4. Corrected baseline identity

- Included cohort: 12,627 subjects, 14,023 admissions, 14,374 ICU stays.
- AKI: 798 stays (5.55%).
- Split seed: 42, subject-level 70/15/15.
- Held-out test: 1,895 subjects, 2,168 stays, 124 AKI.
- Canonical test-cohort SHA-256: `d56d72c99d71e56e5144b30f7f0a4238f670dad4a24dd739d3ef3b13c7222723`.
- Formal experiments: 5 models x 3 horizons = 15.

## 5. Corrected checkpoint inventory

| Model | Horizon | Checkpoint bytes | SHA-256 status |
|---|---:|---:|---|
| Logistic regression | 0 h | 1,307 | verified |
| Logistic regression | 48 h | 1,307 | verified |
| Logistic regression | 72 h | 1,307 | verified |
| XGBoost | 0 h | 1,555,155 | verified |
| XGBoost | 48 h | 1,448,535 | verified |
| XGBoost | 72 h | 1,573,118 | verified |
| LSTM-Attention | 0 h | 496,373 | verified |
| LSTM-Attention | 48 h | 496,373 | verified |
| LSTM-Attention | 72 h | 496,373 | verified |
| Masked CNN | 0 h | 50,943,110 | verified |
| Masked CNN | 48 h | 50,943,110 | verified |
| Masked CNN | 72 h | 50,943,110 | verified |
| ITE-Transformer | 0 h | 1,828,213 | verified |
| ITE-Transformer | 48 h | 1,828,213 | verified |
| ITE-Transformer | 72 h | 1,828,213 | verified |

- Total checkpoint bytes: 164,383,817 (156.769 MiB).
- Largest single file: 50,943,110 bytes (48.583 MiB).
- Files over 100 MB: 0.
- Detailed paths and hashes: `Step1_Baseline_Reproduction/metadata/checkpoint_inventory.csv`.

No single file exceeds 100 MB. Git LFS is configured in `.gitattributes` for the 15 formal `model_best.pt` and `model_best.joblib` paths only. Local attribute resolution passed for all 15 files, while preprocessing bundles remain ordinary Git files. No LFS upload was performed.

## 6. Upstream integrity

- URL: `https://github.com/Jovinus/aki-timeseries-continuous-monitoring`
- Commit: `6794acb7f615b5864171f8aa7c8ae610049cb947`
- Original upstream working tree status: clean.
- Tracked files compared: 113.
- Missing, changed, or extra release files: 0 / 0 / 0.
- Nested `.git` present in clean release: no.

Three upstream-owned command-line scripts contain nested-quote f-strings that do not parse under Python 3.11. They fail identically in the original clean upstream checkout and were left unchanged. The model-definition and analysis components actually loaded by the project wrappers passed import/forward validation. The three unused upstream files are:

- `train_cnn/scripts/experiment_holdout_inference.py`
- `train_lstm_attention/experiment_inference.py`
- `train_transformer/experiment_inference.py`

## 7. Dataset requirement

Only the course-provided `MIMIC_IV.zip` is required. Its default location is:

```text
data/raw/MIMIC_IV.zip
```

`MIMIC_IV_ZIP` can override the default. Local validation used an external course archive through that environment variable; no machine-specific data path is stored in the repository. The archive was not copied. The validator identified MIMIC-IV 3.1 and found exactly one of every required member:

- `hosp/admissions.csv`
- `hosp/patients.csv`
- `hosp/labevents.csv`
- `hosp/d_labitems.csv`
- `icu/icustays.csv`
- `icu/chartevents.csv`
- `icu/outputevents.csv`
- `icu/d_items.csv`

The separate notes archive is not an executable dependency.

## 8. Absolute-path and prohibited-branch audit

Recursive scans covered `.py`, `.ps1`, `.bat`, `.json`, `.yaml`, `.yml`, `.toml`, `.md`, `.txt`, and `.csv` files.

- Executable/configuration references to personal drive paths: 0.
- Main scientific README/result references to personal drive paths: 0.
- Prohibited obsolete hash references outside this audit's exclusion record: 0.
- Prohibited exploratory-branch terms outside this audit's exclusion record: 0.
- Raw MIMIC archives in the release: 0.
- Patient-level prediction/cohort artifacts in the release: 0.
- Secret/token/credential findings: 0. A code symbol named `cls_token` in the official Transformer implementation is a model token, not a credential.

Absolute source and destination paths appear in this audit only to document provenance. They are not used by executable code.

## 9. Validation commands and results

Representative local commands:

```powershell
$env:MIMIC_IV_ZIP="<path-to-course-data>\MIMIC_IV.zip"
python scripts/check_data.py

$env:AKI_VALIDATION_BASELINE="<completed-corrected-baseline-directory>"
python scripts/release_null_control.py
```

Results:

| Validation | Result |
|---|---|
| Required directory/file structure | PASS |
| Public project Python syntax and import checks (39 files, excluding unmodified upstream tree) | PASS, 39/39 |
| Step 1 path resolution from outside the repository | PASS |
| Step 2 frozen-policy enforcement and path resolution | PASS |
| Data preflight using the local course archive | PASS; MIMIC-IV 3.1, 8/8 required members |
| Missing-data message with no override | PASS; clear non-downloading instructions, exit 1 |
| Step 1 manifest cardinality and unique model/horizon pairs | PASS; 15/15 |
| Checkpoint SHA-256 and configuration identity | PASS; 15/15 |
| Feature lists and preprocessing bundles | PASS |
| Corrected test cohort count/hash metadata | PASS |
| Upstream tracked-tree byte comparison | PASS; 113/113 identical |
| References PDF byte comparison | PASS; 2/2 identical |
| Step 2/3/4 script import and CLI path checks | PASS |
| Five-model read-only 0 h forward-pass null control | PASS; 5/5, no artifact changes |
| Internal relative Markdown links | PASS; 12/12 |
| Result tables against retained validated aggregate sources | PASS |
| Result inventory | PASS; 5 figures, 3 tables |
| Raw/patient-level/sensitive-file audit | PASS |

The five-model null control read one held-out processed input from the completed local corrected Baseline, redacted the stay identifier from output, loaded the release checkpoints and release upstream model code, ran CPU inference, and confirmed finite probabilities plus unchanged model/artifact fingerprints. It performed no training and wrote no patient-level output.

## 10. Result contents

Figures:

1. `01_lab_availability_delay.png`
2. `02_pending_lab_availability.png`
3. `03_delta_auroc_by_timepoint.png`
4. `04_delta_auroc_t0_forest.png`
5. `05_onset_creatinine_analysis.png`

Tables:

1. `availability_summary.csv`
2. `primary_storetime_results.csv`
3. `key_alert_changes.csv`

The primary storetime table contains all 15 model/horizon comparisons. Fourteen did not meet the prespecified material-change rule; the 72 h Masked CNN did.

## 11. Release size and upload planning

- Total clean-release files: 354.
- Total clean-release bytes: 174,399,614.
- Total clean-release size: 166.320 MiB.
- Files over 100 MB: 0.
- Git LFS status: installed locally; path-specific attribute resolution and staged-pointer checks pass for all 15 formal checkpoint binaries.
- License status: project-authored code is covered by the root MIT License. The papers and upstream snapshot retain their original licenses and are not relicensed by the root license.

## 12. Final publication status

The local repository is on branch `main` with a clean working tree and no remote. No GitHub upload, push, LFS push, remote creation, repository creation, or release was performed. The release awaits explicit user approval for any network publication step.
