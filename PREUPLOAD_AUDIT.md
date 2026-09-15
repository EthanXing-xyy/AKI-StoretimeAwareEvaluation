# Pre-upload Audit

- Final repository path: `A:\AKIPrediction_GitHub`
- Top-level scientific structure: `References`, `data`, `Step1_Baseline_Reproduction`, `Step2_Continuous_Monitoring`, `Step3_Data_Availability_Audit`, `Step4_Storetime_Aware_Stress_Test`, `Result`, and `external`
- Reference directory: rename to `References` completed; repository-wide search for the former directory name returned zero matches
- Reference integrity: both PDFs are present and retain their pre-rename SHA-256 values
  - `41746_2026_Article_2722.pdf`: `45e1d61e5d31108699f312b27f97d7c32198cc03659cdea919edc318b8734aaa`
  - `jmir-2025-1-e73987.pdf`: `0580bd59d4d0318e6cb73221740ef1554c9112b2260e16aa37cdea2626d56865`
- Publication payload: 354 files, 174399614 bytes (166.320 MiB), excluding local `.git` metadata
- Formal model artifacts: 15/15 checkpoint hashes verified; corrected cohort hash is `d56d72c99d71e56e5144b30f7f0a4238f670dad4a24dd739d3ef3b13c7222723`
- Git LFS: 15/15 formal checkpoints are tracked as LFS pointers; no other nonempty release artifact is selected by the path-specific rules
- Results: all 5 figures and 3 summary tables match their retained validated sources
- Dataset exclusion: no MIMIC archive, raw record, patient identifier table, or patient-level prediction dump is tracked
- Upstream integrity: 113/113 files match official commit `6794acb7f615b5864171f8aa7c8ae610049cb947`; upstream `LICENSE` is retained; nested `.git` metadata is excluded
- Five-model null control: PASS for logistic regression, XGBoost, LSTM-Attention, Masked CNN, and ITE-Transformer
- Initial release commit: `743659806f84d39a59c284d7a9ed6ce60f4bae82`
- Current branch: `main`
- Expected status after the audit commit: clean working tree
- Git remote: none
- Validation status: PASS

No GitHub repository was created. No remote was added. No GitHub upload, push, LFS push, or release was performed.
