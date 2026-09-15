# Step 3: Laboratory Data Availability Audit

## Purpose

Determine whether a substantial mismatch exists between laboratory `charttime` and the `storetime` result-availability proxy in the frozen held-out evaluation cohort. This audit does not test whether model performance changes; Step 4 answers that question.

## Input

- the Step 1 corrected cohort and interim laboratory events regenerated under `work/`;
- Step 2 monitoring cutoffs and eligibility records regenerated under `work/`;
- `hosp/labevents.csv` streamed from the local MIMIC-IV archive.

## Method

The audit verifies the `labevent_id` join, summarizes `storetime - charttime` delay, counts charttime-visible but not-yet-stored rows at each monitoring cutoff, examines pending creatinine, and evaluates availability of onset-defining creatinine measurements. It performs neither model inference nor training.

## Run

After Steps 1 and 2:

```powershell
python Step3_Data_Availability_Audit/scripts/00_integrity.py snapshot --label before
python Step3_Data_Availability_Audit/scripts/01_extract_storetime.py
python Step3_Data_Availability_Audit/scripts/02_delay_distribution.py
python Step3_Data_Availability_Audit/scripts/03_timepoint_pending.py
python Step3_Data_Availability_Audit/scripts/04_onset_creatinine.py
python Step3_Data_Availability_Audit/scripts/05_figures_and_summary.py
python Step3_Data_Availability_Audit/scripts/00_integrity.py snapshot --label after
python Step3_Data_Availability_Audit/scripts/00_integrity.py compare --label-before before --label-after after
```

## Output

- `tables/`: aggregate join, delay, pending-lab, onset-creatinine, decision, and data-integrity results;
- `figures/`: aggregate availability figures;
- `summary/data_availability_audit.md`;
- row-level working files under ignored `work/Step3_Data_Availability_Audit` paths.

## Validation

The workflow checks join coverage, event-field consistency, monitoring-slice counts, and before/after hashes of Steps 1 and 2. Raw laboratory data are streamed read-only from the archive.

## Notes

`storetime` is a proxy for when a result became available in the record system, not the exact time a clinician viewed it. Missing values are reported explicitly.
