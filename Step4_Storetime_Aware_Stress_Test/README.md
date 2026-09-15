# Step 4: Storetime-aware Frozen-model Stress Test

## Purpose

Measure the effect of a storetime-aware laboratory availability rule while holding the models, preprocessing, thresholds, cohort, and monitoring design fixed.

## Input

- all frozen Step 1 model and preprocessing artifacts;
- the fixed Step 1 held-out test cohort;
- Step 2 reference times, monitoring timepoints, and charttime predictions;
- Step 3 laboratory storetime joins and audit tables;
- the prespecified rules and protocol-deviation records in `configs/`.

## Method

The core comparison changes only laboratory availability:

```text
charttime-only:  charttime <= cutoff
storetime-aware: charttime <= cutoff AND storetime <= cutoff
```

Vitals retain their charttime rule. Missing `storetime` is conservatively treated as available and reported. The `onset_creatinine_only` mechanism variant tests whether delayed availability of the creatinine defining AKI onset explains model differences.

## Run

First run the offline slicing check and the five-model horizon-0 charttime null control:

```powershell
python Step4_Storetime_Aware_Stress_Test/scripts/00_integrity.py snapshot --label before
python Step4_Storetime_Aware_Stress_Test/scripts/check_patch_offline.py --availability storetime
$models = "logistic_regression","xgboost","lstm_attention","masked_cnn","ite_transformer"
foreach ($model in $models) {
  python Step4_Storetime_Aware_Stress_Test/scripts/run_stress.py monitor --availability charttime --horizons 0 --models $model
}
python Step4_Storetime_Aware_Stress_Test/scripts/check_null_control.py
```

After the null control passes, run every formal checkpoint under both availability variants, then generate downstream comparisons:

```powershell
$variants = "storetime","onset_creatinine_only"
$horizons = 0,48,72
foreach ($variant in $variants) {
  foreach ($horizon in $horizons) {
    foreach ($model in $models) {
      python Step4_Storetime_Aware_Stress_Test/scripts/run_stress.py monitor --availability $variant --horizons $horizon --models $model
    }
  }
  python Step4_Storetime_Aware_Stress_Test/scripts/run_stress.py downstream --availability $variant
}
python Step4_Storetime_Aware_Stress_Test/scripts/compare_vs_charttime.py
python Step4_Storetime_Aware_Stress_Test/scripts/make_stress_report.py
python Step4_Storetime_Aware_Stress_Test/scripts/00_integrity.py snapshot --label after
python Step4_Storetime_Aware_Stress_Test/scripts/00_integrity.py compare --label-before before --label-after after
```

## Output

- `tables/`: paired discrimination, bootstrap intervals, threshold-metric and alert changes, onset mechanism results, rankings, and validation gates;
- `figures/`: primary and mechanism-result figures;
- `summary/storetime_stress_test.md`;
- patient-level working predictions under ignored `work/` paths only.

## Validation

`check_patch_offline.py` validates availability slicing without inference. The charttime null control must reproduce all five horizon-0 architectures before other variants can run. Paired comparisons use 2,000 stay-level bootstrap resamples stratified by AKI status with seed 42. A material change requires a 95% confidence interval excluding zero and an absolute delta AUROC of at least 0.02.

## Notes

Four non-AKI stay-timepoints became ineligible because every pre-cutoff observation was a pending laboratory row; paired metrics use the eligibility intersection. Eligibility at -12 and 0 hours is unchanged. A Masked CNN identity check exhibited a TF32 batch-shape numerical difference; CPU and TF32-disabled checks confirmed numerical reproducibility behavior rather than a storetime effect. `storetime` remains an availability proxy, not an exact clinician-view timestamp.
