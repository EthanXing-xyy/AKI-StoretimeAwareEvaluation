# Laboratory Result-availability Audit

This read-only audit found substantial mismatch between laboratory `charttime` and the `storetime` availability proxy in the corrected held-out MIMIC-IV cohort.

- 190,321 laboratory rows were evaluated inside test ICU windows.
- Median `storetime - charttime` delay was 52 minutes (IQR 26-73).
- 38.7% of laboratory rows had delays greater than one hour.
- At the primary 0 h matched-reference cutoff, 2,145 of 150,186 charttime-visible laboratory rows (1.43%) were not yet stored.
- All 24 creatinine-defined AKI onsets had an onset-defining creatinine with `storetime` after the exact 0 h cutoff.

The audit establishes that availability mismatch exists. It does not establish that model performance changes materially; that is evaluated in Step 4.

`storetime` is a proxy for result availability, not the exact time a clinician viewed a result.
