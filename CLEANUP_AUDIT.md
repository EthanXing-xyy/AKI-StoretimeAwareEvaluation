# Publication Cleanup Audit

- Date: 2026-09-15 (Asia/Shanghai)
- Scope: local publication cleanup of this release only
- Files removed: 3
- Removed categories: two empty, unreferenced Python package placeholders under Step 1; one regenerable Step 3 validation table containing stay identifiers
- Debug/dead code removed: no executable debug code was present; the two unused placeholder files were removed
- Documentation updated: root README, Step 1-4 READMEs, References README, external README, module docstrings, and release audit
- Reproducibility artifacts retained: 15 formal checkpoints, artifact hashes, corrected cohort hash, preprocessing metadata, random seed, frozen-policy gates, null controls, integrity checks, protocol deviations, the four ineligible stay-timepoints note, the TF32 numerical note, and the storetime proxy limitation
- License: project-authored code is MIT licensed; papers and upstream code are explicitly excluded from the project license and retain their original licenses
- Git LFS: `.gitattributes` tracks only the 15 formal `model_best.pt` and `model_best.joblib` files; local attribute resolution and staged-pointer checks passed for 15/15 files, and `preprocessing.joblib` remains outside LFS
- Final repository inventory: 354 files, 174399614 bytes (166.320 MiB), excluding local `.git` metadata
- Validation: PASS for Python syntax/imports, Markdown links, JSON/YAML parsing, data preflight, Step 1 manifest and checkpoint hashes, corrected cohort hash, frozen-policy gates, upstream integrity, five-model null control, Result consistency, prohibited-path/content scans, and file-size inventory
- Local Git status: initialized on `main`; initial release and pre-upload audit commits created; working tree clean; no remote configured
- Remote status: no GitHub upload, push, LFS push, remote creation, repository creation, or release was performed
