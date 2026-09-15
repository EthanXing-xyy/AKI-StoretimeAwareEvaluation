# Retrospective Continuous-monitoring Results

All 15 corrected formal checkpoints were evaluated at -72, -60, -48, -36, -24, -12, and 0 hours under matched and discharge reference-time definitions. Metrics use the stays eligible at each time point. Alert decisions use each checkpoint's frozen validation-selected threshold.

Released validation tables confirm:

- the corrected held-out cohort identity;
- no event later than its prediction cutoff entered an encoded slice;
- static inputs and probabilities were reproduced within the prespecified tolerances;
- parameters and preprocessing artifacts were unchanged during inference;
- aggregate results cover all model, horizon, reference, and time-point combinations.

The summary tables include discrimination, alert burden, clinical-faithfulness trends, and static-versus-continuous rankings. Patient-level predictions are intentionally excluded from the public release.
