"""Aggregate formal baseline metrics and generate summary figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import load_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    paths = load_paths()
    files = sorted((paths["results_root"] / "metrics" / args.scope).glob("horizon_*/*.json"))
    rows = []
    for path in files:
        row = json.loads(path.read_text(encoding="utf-8"))
        row.pop("confusion_matrix", None)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No metrics found for scope={args.scope}")
    result = pd.DataFrame(rows).sort_values(["prediction_window", "model"])
    preferred_columns = [
        "model", "prediction_window", "auroc", "auprc", "accuracy",
        "sensitivity", "specificity", "ppv", "npv", "f1", "brier_score",
        "threshold", "n_train", "n_val", "n_test", "n_positive",
        "prevalence", "seed", "scope",
    ]
    result = result[[column for column in preferred_columns if column in result.columns]]
    out = paths["results_root"] / "baseline_summary"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.scope == "full" else "_smoke"
    result.to_csv(out / f"baseline_results{suffix}.csv", index=False)
    (out / f"baseline_results{suffix}.md").write_text(result.to_markdown(index=False), encoding="utf-8")
    for metric in ["auroc", "auprc"]:
        pivot = result.pivot(index="model", columns="prediction_window", values=metric)
        ax = pivot.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), rot=25)
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("")
        ax.set_title(f"MIMIC-IV adapted baseline: {metric.upper()}")
        plt.tight_layout()
        plt.savefig(paths["results_root"] / "figures" / f"{metric}_comparison{suffix}.png", dpi=180)
        plt.close()

    # Reliability diagrams are generated from the saved, common-cohort test
    # predictions. Quantile bins remain useful under the low AKI prevalence.
    for horizon in sorted(result.prediction_window.unique()):
        figure, axis = plt.subplots(figsize=(7, 6))
        prediction_dir = paths["results_root"] / "predictions" / args.scope / f"horizon_{int(horizon)}"
        for model_name in sorted(result.loc[result.prediction_window == horizon, "model"]):
            prediction_path = prediction_dir / f"{model_name}.parquet"
            if not prediction_path.exists():
                continue
            predictions = pd.read_parquet(prediction_path)
            observed, predicted = calibration_curve(
                predictions["y_true"], predictions["y_prob"], n_bins=10, strategy="quantile"
            )
            axis.plot(predicted, observed, marker="o", label=model_name)
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="ideal")
        axis.set(xlabel="Mean predicted probability", ylabel="Observed AKI frequency", xlim=(0, 1), ylim=(0, 1), title=f"Test calibration: horizon {int(horizon)}h")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(paths["results_root"] / "figures" / f"calibration_h{int(horizon)}{suffix}.png", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
