# Storetime-aware Frozen-model Stress Test

The stress test reused the corrected 2,168-stay held-out cohort, all 15 checkpoints, preprocessing, feature order, validation-selected thresholds, reference-time logic, and seven monitoring time points. Only laboratory availability changed from `charttime <= cutoff` to `charttime <= cutoff AND storetime <= cutoff`.

At the primary 0 h matched-reference endpoint, 14 of 15 model/horizon comparisons did not satisfy the prespecified material-change rule. The 72 h Masked CNN was the exception (delta AUROC -0.0229; 95% CI -0.0442 to -0.0060).

Substantial laboratory result-availability mismatch was observed, but storetime-aware frozen-model stress testing showed little material effect on discrimination for most evaluated models.

This result is specific to MIMIC-IV, this frozen-model design, and `storetime` as an availability proxy. It does not prove that temporal leakage is absent and does not generalize to every EHR model.

Four non-AKI stay-timepoints became ineligible under the storetime-aware rule. Paired metrics used the eligibility intersection; the 0 h primary endpoint was unaffected. A separate Masked CNN TF32 batch-shape difference is documented in `configs/protocol_deviations.json` and is not interpreted as a storetime effect.
