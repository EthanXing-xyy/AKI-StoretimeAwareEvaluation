# Final Results

## Research question

Does use of laboratory `charttime`, rather than the `storetime` result-availability proxy, materially inflate retrospective continuous AKI prediction performance in MIMIC-IV?

## Finding

There is substantial laboratory result-availability mismatch. Across 190,321 laboratory rows in the frozen test ICU windows, the median `storetime - charttime` delay was 52 minutes (IQR 26-73), and 38.7% were delayed by more than one hour. At the primary 0 h cutoff, 2,145 of 150,186 charttime-visible laboratory rows (1.43%) were not yet stored. All 24 AKI stays whose onset was creatinine-defined had an onset-defining creatinine not yet stored at the exact 0 h cutoff.

## Frozen-model stress test

Storetime-aware frozen-model inference caused little material change in discrimination for most evaluated models. Fourteen of 15 model/horizon comparisons did not meet the prespecified material-change rule at 0 h. The exception was the 72 h Masked CNN: delta AUROC -0.0229 with 95% CI -0.0442 to -0.0060.

Accordingly, the evaluated models appear largely robust to this specific source of retrospective timing mismatch. This statement applies only to the frozen-model MIMIC-IV stress test. It does not prove absence of temporal leakage, does not imply that storetime has no effect, and should not be generalized to all EHR models.

`storetime` is a proxy for result availability, not the exact time a clinician viewed a result. No model was retrained, no preprocessing was refit, no threshold was retuned, and the held-out cohort was unchanged.

## Figures

1. `figures/01_lab_availability_delay.png` — laboratory availability delay by feature.
2. `figures/02_pending_lab_availability.png` — pending laboratory availability over monitoring time.
3. `figures/03_delta_auroc_by_timepoint.png` — storetime-aware delta AUROC over time.
4. `figures/04_delta_auroc_t0_forest.png` — paired 0 h delta AUROC and 95% bootstrap intervals.
5. `figures/05_onset_creatinine_analysis.png` — onset-creatinine mechanism analysis.

## Tables

- `tables/availability_summary.csv`
- `tables/primary_storetime_results.csv`
- `tables/key_alert_changes.csv`

Every number in this folder is derived from the validated aggregate outputs retained in Steps 3 and 4.
