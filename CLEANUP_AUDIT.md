# Publication Cleanup Audit

- Date: 2026-09-15 (Asia/Shanghai)
- Scope: local publication cleanup of this release only
- Files removed: 2
- Removed category: empty, unreferenced Python package placeholders under Step 1
- Debug/dead code removed: no executable debug code was present; the two unused placeholder files were removed
- Documentation updated: root README, Step 1-4 READMEs, References README, external README, module docstrings, and release audit
- Reproducibility artifacts retained: 15 formal checkpoints, artifact hashes, corrected cohort hash, preprocessing metadata, random seed, frozen-policy gates, null controls, integrity checks, protocol deviations, the four ineligible stay-timepoints note, the TF32 numerical note, and the storetime proxy limitation
- License: project-authored code is MIT licensed; papers and upstream code are explicitly excluded from the project license and retain their original licenses
- Git LFS: `.gitattributes` tracks only the 15 formal `model_best.pt` and `model_best.joblib` files; local attribute resolution passed for 15/15 files, and `preprocessing.joblib` remains outside LFS
- Final repository inventory: 354 files, 174,403,709 bytes (166.324 MiB)
- Validation: PASS for Python syntax/imports, Markdown links, JSON/YAML parsing, data preflight, Step 1 manifest and checkpoint hashes, corrected cohort hash, frozen-policy gates, upstream integrity, five-model null control, Result consistency, prohibited-path/content scans, and file-size inventory
- Remote status: no GitHub upload, push, remote creation, or release was performed
