Read AGENTS.md and README.md before running or modifying anything.

- Put the course archive at `data/raw/MIMIC_IV.zip`, or set `MIMIC_IV_ZIP`.
- Follow Step 1 -> Step 2 -> Step 3 -> Step 4 -> Result.
- Use the corrected baseline only: 2,168 test stays, 124 AKI cases, hash beginning `d56d72c9`.
- Keep exactly 15 formal checkpoints; do not restore non-formal or obsolete artifacts.
- Keep upstream code unchanged at commit `6794acb`.
- Do not restore removed exploratory analyses.
- In Step 2 and Step 4, never retrain, refit preprocessing, resplit, or tune on the test set.
- Treat `storetime` as an availability proxy, not an exact clinician-view timestamp.
