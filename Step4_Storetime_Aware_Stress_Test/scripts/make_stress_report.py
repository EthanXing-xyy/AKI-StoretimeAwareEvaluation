"""Create the public all-eligible storetime stress-test report and figures."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import st_patch as st  # noqa: E402

TABLES = st.STRESS_ROOT / "comparison" / "tables"
FIGURES = st.STRESS_ROOT / "comparison" / "figures"
SUMMARY = st.STRESS_ROOT / "summary"
MODELS = ["logistic_regression", "xgboost", "lstm_attention", "masked_cnn", "ite_transformer"]
COLORS = dict(zip(MODELS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]))


def save_delta_trajectory(frame: pd.DataFrame) -> Path:
    data = frame[
        frame.availability.eq("storetime")
        & frame.reference_mode.eq("matched")
        & frame.cohort.eq("all_eligible")
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, horizon in zip(axes, [0, 48, 72]):
        for model in MODELS:
            part = data[data.prediction_horizon.eq(horizon) & data.model.eq(model)].sort_values("monitoring_timepoint")
            axis.plot(part.monitoring_timepoint, part.delta_auroc, marker="o", label=model, color=COLORS[model])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=f"{horizon} h model", xlabel="Hours relative to reference")
    axes[0].set_ylabel("Delta AUROC (storetime-aware - charttime-only)")
    axes[-1].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    path = FIGURES / "delta_auroc_by_timepoint_storetime.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def save_t0_forest(frame: pd.DataFrame) -> Path:
    data = frame[
        frame.availability.eq("storetime")
        & frame.reference_mode.eq("matched")
        & frame.cohort.eq("all_eligible")
        & frame.monitoring_timepoint.eq(0)
    ].copy()
    data["label"] = data.model + " / " + data.prediction_horizon.astype(str) + " h"
    data = data.sort_values(["prediction_horizon", "model"])
    figure, axis = plt.subplots(figsize=(8, 6))
    y = range(len(data))
    axis.errorbar(
        data.delta_auroc,
        y,
        xerr=[data.delta_auroc - data.delta_auroc_ci_low, data.delta_auroc_ci_high - data.delta_auroc],
        fmt="o",
        color="#0072B2",
        capsize=2,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvline(-0.02, color="#D55E00", linestyle="--", linewidth=0.8)
    axis.axvline(0.02, color="#D55E00", linestyle="--", linewidth=0.8)
    axis.set(yticks=list(y), yticklabels=data.label, xlabel="Delta AUROC with 95% paired bootstrap CI")
    figure.tight_layout()
    path = FIGURES / "delta_auroc_t0_forest.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    boot = pd.read_csv(TABLES / "paired_discrimination_bootstrap.csv")
    primary = pd.read_csv(TABLES / "prespecified_rule_evaluation.csv")
    burden = pd.read_csv(TABLES / "alert_burden_deltas.csv")
    gates = pd.read_csv(TABLES / "pairing_and_identity_gates.csv")
    written = [save_delta_trajectory(boot), save_t0_forest(boot)]
    focus = primary[primary.availability.eq("storetime")].sort_values(["prediction_horizon", "model"])
    report = [
        "# Storetime-aware frozen-model stress test",
        "",
        "Laboratory rows were usable only when both `charttime <= cutoff` and `storetime <= cutoff`. Models, preprocessing, validation-selected thresholds, held-out cohort, reference-time logic, and monitoring time points were frozen.",
        "",
        "`storetime` is a proxy for result availability, not the exact time a clinician viewed a result.",
        "",
        "## Primary paired discrimination result",
        "",
        focus.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Substantial laboratory result-availability mismatch was observed, but storetime-aware frozen-model stress testing showed little material effect on discrimination for most evaluated models.",
        "",
        "## Validation gates",
        "",
        gates.to_markdown(index=False),
        "",
        "## Alert-burden comparisons",
        "",
        burden.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Figures",
        "",
        *[f"- `{path.relative_to(st.STRESS_ROOT).as_posix()}`" for path in written],
        "",
    ]
    (SUMMARY / "storetime_stress_test.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
