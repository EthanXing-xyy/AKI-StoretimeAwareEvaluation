# %%
"""
Run ALL revision round 1 steps sequentially.
Results → results/
Figures → figures/ (PDF only)
Sampling: 2000 patients per dataset for FI computation.
"""

import sys
from pathlib import Path
import time
import logging

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_step(step_name: str, func):
    logger.info(f"\n{'#'*70}")
    logger.info(f"# {step_name}")
    logger.info(f"{'#'*70}")
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        logger.info(f"# {step_name} DONE in {elapsed:.0f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"# {step_name} FAILED after {elapsed:.0f}s: {e}", exc_info=True)
        return False


# ── Step 1 ───────────────────────────────────────────────────────────
def step1():
    from experiments.step1_reference_time_matching import main
    main()

# ── Step 2 ───────────────────────────────────────────────────────────
def step2():
    from experiments.step2_baseline_evaluation import main
    main()

# ── Step 3 (DL + Baseline FI, 2000 samples) ─────────────────────────
def step3():
    from experiments.step3_feature_importance_single import main
    main()

# ── Step 4 ───────────────────────────────────────────────────────────
def step4():
    from experiments.step4_feature_importance_online import main
    main()

# ── Step 5 ───────────────────────────────────────────────────────────
def step5():
    from experiments.step5_feature_subset import main
    main()

# ── Step 6 ───────────────────────────────────────────────────────────
def step6():
    from experiments.step6_sensitivity_analysis import main
    main()

# ── Step 7 ───────────────────────────────────────────────────────────
def step7():
    from experiments.step7_alert_burden import main
    main()

# ── Step 8 ───────────────────────────────────────────────────────────
def step8():
    from experiments.step8_missingness import main
    main()

# ── Figures ──────────────────────────────────────────────────────────
def figures():
    from experiments.update_figures import main
    main()


def main():
    t_total = time.time()
    logger.info("=" * 70)
    logger.info("REVISION ROUND 1 — FULL PIPELINE")
    logger.info("=" * 70)

    results = {}
    for name, func in [
        ("Step 1: Reference Time Matching", step1),
        ("Step 2: Baseline Evaluation", step2),
        ("Step 3: Permutation Feature Importance", step3),
        ("Step 4: Online Feature Importance", step4),
        ("Step 5: Feature Subset", step5),
        ("Step 6: Sensitivity Analysis", step6),
        ("Step 7: Alert Burden", step7),
        ("Step 8: Missingness", step8),
        ("Figures: Update All (PDF)", figures),
    ]:
        results[name] = run_step(name, func)

    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*70}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        logger.info(f"  [{status}] {name}")
    logger.info(f"\nTotal time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
