"""Validate formal baseline predictions, metrics, and cohort identity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import binary_metrics
from src.utils.config import load_paths


MODELS = ["logistic_regression", "xgboost", "lstm_attention", "masked_cnn", "ite_transformer"]
HORIZONS = [0, 48, 72]
NUMERIC_METRICS = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "brier_score"]


def main() -> None:
    paths = load_paths()
    checks: list[dict] = []
    reference_ids = None

    for horizon in HORIZONS:
        data_dir = paths["processed_data_root"] / "mimiciv" / "full" / f"horizon_{horizon}"
        master = pd.read_parquet(data_dir / "master.parquet")
        leakage = master.groupby("subject_id")["split"].nunique().max() > 1
        checks.append({"check": f"h{horizon}_patient_split_leakage", "passed": not bool(leakage), "detail": "no subject occurs in multiple splits" if not leakage else "leakage detected"})
        expected = master.loc[master.split == "test", ["subject_id", "hadm_id", "stay_id", "label"]].sort_values("stay_id").reset_index(drop=True)
        horizon_ids = tuple(expected.stay_id.astype(int))
        if reference_ids is None:
            reference_ids = horizon_ids
        checks.append({"check": f"h{horizon}_same_test_stays", "passed": horizon_ids == reference_ids, "detail": f"n_test={len(expected)}"})

        for model_name in MODELS:
            model_dir = paths["model_root"] / model_name / f"horizon_{horizon}" / "full"
            prediction_path = paths["results_root"] / "predictions" / "full" / f"horizon_{horizon}" / f"{model_name}.parquet"
            metric_path = paths["results_root"] / "metrics" / "full" / f"horizon_{horizon}" / f"{model_name}.json"
            required = [model_dir / "config.json", model_dir / "training_history.csv", model_dir / "feature_list.json", model_dir / "scaling_info.parquet", prediction_path, metric_path]
            required.append(model_dir / ("model_best.joblib" if model_name in ["logistic_regression", "xgboost"] else "model_best.pt"))
            if model_name in ["logistic_regression", "xgboost"]:
                required.append(model_dir / "preprocessing.joblib")
            missing = [str(path) for path in required if not path.exists()]
            checks.append({"check": f"{model_name}_h{horizon}_artifacts", "passed": not missing, "detail": "complete" if not missing else f"missing: {missing}"})
            if missing:
                continue

            predictions = pd.read_parquet(prediction_path).sort_values("stay_id").reset_index(drop=True)
            identity_ok = predictions[["subject_id", "hadm_id", "stay_id"]].equals(expected[["subject_id", "hadm_id", "stay_id"]])
            labels_ok = np.array_equal(predictions.y_true.to_numpy(int), expected.label.to_numpy(int))
            probabilities_ok = bool(np.isfinite(predictions.y_prob).all() and predictions.y_prob.between(0, 1).all())
            saved = json.loads(metric_path.read_text(encoding="utf-8"))
            recomputed = binary_metrics(predictions.y_true.to_numpy(int), predictions.y_prob.to_numpy(float), float(saved["threshold"]))
            # Parquet preserves model float32 probabilities; the original JSON
            # metric was computed before the round-trip, so allow sub-nanolevel
            # floating-point serialization differences.
            metrics_ok = all(np.isclose(float(saved[key]), float(recomputed[key]), atol=1e-9, rtol=1e-9) for key in NUMERIC_METRICS)
            matrix_ok = saved["confusion_matrix"] == recomputed["confusion_matrix"]
            checks.append({"check": f"{model_name}_h{horizon}_prediction_integrity", "passed": bool(identity_ok and labels_ok and probabilities_ok and metrics_ok and matrix_ok), "detail": f"identity={identity_ok}, labels={labels_ok}, probabilities={probabilities_ok}, metrics={metrics_ok}, confusion_matrix={matrix_ok}"})

    summary_path = paths["results_root"] / "baseline_summary" / "baseline_results.csv"
    summary = pd.read_csv(summary_path)
    expected_pairs = {(model, horizon) for model in MODELS for horizon in HORIZONS}
    actual_pairs = set(zip(summary.model, summary.prediction_window.astype(int)))
    checks.append({"check": "summary_15_experiments", "passed": len(summary) == 15 and actual_pairs == expected_pairs, "detail": f"rows={len(summary)}, unique_pairs={len(actual_pairs)}"})

    passed = sum(item["passed"] for item in checks)
    report = {"passed": passed == len(checks), "checks_passed": passed, "checks_total": len(checks), "checks": checks}
    out_dir = paths["results_root"] / "baseline_summary"
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Baseline validation report", "", f"Overall: **{'PASS' if report['passed'] else 'FAIL'}** ({passed}/{len(checks)} checks)", "", "| Check | Result | Detail |", "|---|---:|---|"]
    lines.extend(f"| {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |" for item in checks)
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["passed", "checks_passed", "checks_total"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
