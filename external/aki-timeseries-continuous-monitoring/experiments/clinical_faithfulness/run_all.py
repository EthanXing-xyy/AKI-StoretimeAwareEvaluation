# %%
"""
Clinical Faithfulness Analysis — Run All Steps

Steps:
  A: Time-window metric computation (AUROC, PPV, etc. with bootstrap CI)
  B: Mann-Kendall trend test
  C: Trend estimation (Sen's slope + OLS)
  D: Figure generation
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from experiments.clinical_faithfulness.step_a_compute_metrics import run_step_a
from experiments.clinical_faithfulness.step_b_mann_kendall import run_step_b
from experiments.clinical_faithfulness.step_c_trend_estimation import run_step_c
from experiments.clinical_faithfulness.step_d_figures import run_step_d


def main():
    logger.info("=" * 70)
    logger.info("Clinical Faithfulness Analysis — Full Pipeline")
    logger.info("=" * 70)

    # Step A
    metrics_df = run_step_a()
    logger.info(f"Step A done: {len(metrics_df)} rows\n")

    # Step B
    mk_df = run_step_b(metrics_df)
    logger.info(f"Step B done: {len(mk_df)} rows\n")

    # Step C
    trend_df = run_step_c(metrics_df, mk_df)
    logger.info(f"Step C done: {len(trend_df)} rows\n")

    # Step D
    run_step_d()
    logger.info("Step D done\n")

    logger.info("=" * 70)
    logger.info("All steps complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
