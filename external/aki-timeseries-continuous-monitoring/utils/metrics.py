# %%
"""
Shared metric computation utilities for revision experiments.
Wraps existing project functions and adds bootstrap CI support.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def decode_pred_proba(series: pd.Series) -> pd.Series:
    """Decode prediction probabilities from various storage formats."""
    def _decode_single(val):
        if isinstance(val, (float, int, np.floating, np.integer)):
            return float(val)
        if isinstance(val, bytes):
            if len(val) == 2:
                return float(np.frombuffer(val, dtype=np.float16)[0])
            elif len(val) == 4:
                return float(np.frombuffer(val, dtype=np.float32)[0])
            else:
                return np.nan
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    decoded = series.apply(_decode_single)
    # Apply sigmoid if values look like logits (outside [0,1])
    if decoded.dropna().max() > 1.0 or decoded.dropna().min() < 0.0:
        decoded = decoded.apply(lambda x: sigmoid(x) if not np.isnan(x) else np.nan)
    return decoded


def calculate_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold maximizing Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute standard binary classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0

    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "threshold": threshold,
        "n": len(y_true),
        "n_events": int(y_true.sum()),
    }


def bootstrap_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute metrics with bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    point_est = compute_binary_metrics(y_true, y_prob, threshold)

    alpha = 1 - ci_level
    metric_keys = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1"]
    boot_results = {k: [] for k in metric_keys}

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t, y_p = y_true[idx], y_prob[idx]
        if len(np.unique(y_t)) < 2:
            continue
        m = compute_binary_metrics(y_t, y_p, threshold)
        for k in metric_keys:
            boot_results[k].append(m[k])

    result = {}
    for k in metric_keys:
        result[k] = point_est[k]
        if boot_results[k]:
            arr = np.array(boot_results[k])
            result[f"{k}_ci_lower"] = float(np.percentile(arr, alpha / 2 * 100))
            result[f"{k}_ci_upper"] = float(np.percentile(arr, (1 - alpha / 2) * 100))
        else:
            result[f"{k}_ci_lower"] = np.nan
            result[f"{k}_ci_upper"] = np.nan

    result["threshold"] = threshold
    result["n"] = point_est["n"]
    result["n_events"] = point_est["n_events"]
    return result


def isotonic_recalibration_cv(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Apply isotonic recalibration with cross-validation."""
    calibrated = np.copy(y_prob)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, val_idx in skf.split(y_prob, y_true):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob[train_idx], y_true[train_idx])
        calibrated[val_idx] = iso.predict(y_prob[val_idx])

    return calibrated


def compute_brier_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Compute Brier score before and after isotonic recalibration."""
    brier_before = brier_score_loss(y_true, y_prob)
    calibrated = isotonic_recalibration_cv(y_true, y_prob, n_folds, seed)
    brier_after = brier_score_loss(y_true, calibrated)
    improvement = (brier_before - brier_after) / brier_before * 100 if brier_before > 0 else 0.0

    return {
        "brier_before": brier_before,
        "brier_after": brier_after,
        "improvement_pct": improvement,
    }
