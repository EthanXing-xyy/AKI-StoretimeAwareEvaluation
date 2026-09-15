"""Figures and summary tables for retrospective continuous monitoring.

From the repository root after full inference and upstream analyses:
    python Step2_Continuous_Monitoring/scripts/make_figures_and_summary.py

Every figure has a table twin in summary/continuous_monitoring_results.md / .csv.
Model colours are reference categorical slots 1-5 (validated light mode: CVD dE >= 9.1, normal dE >= 19.6);
three slots are below 3:1 contrast, so lines carry direct end labels and markers.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import cm_adapter as cm  # noqa: E402

TIMEPOINTS = [-72, -60, -48, -36, -24, -12, 0]
MODEL_ORDER = ["ite_transformer", "lstm_attention", "masked_cnn", "xgboost", "logistic_regression"]
MODEL_COLORS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MODEL_LABELS = {"ite_transformer": "ITE-Transformer", "lstm_attention": "LSTM-Attention", "masked_cnn": "Masked CNN", "xgboost": "XGBoost", "logistic_regression": "Logistic regression"}
MODEL_SHORT = {"ite_transformer": "TF", "lstm_attention": "LSTM", "masked_cnn": "CNN", "xgboost": "XGB", "logistic_regression": "LR"}
MARKERS = {"ite_transformer": "^", "lstm_attention": "o", "masked_cnn": "s", "xgboost": "D", "logistic_regression": "v"}
HORIZON_STYLES = {0: "-", 48: (0, (5, 2)), 72: (0, (1, 1.5))}
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TAU_CMAP = LinearSegmentedColormap.from_list("tau", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY, "text.color": INK,
})


def auroc_limits(values) -> tuple[float, float]:
    """Never clip below-chance trajectories: extend the floor to the data, in 0.05 steps."""
    low = np.nanmin(np.asarray(values, dtype=float))
    return min(0.45, np.floor((low - 0.02) * 20) / 20), 1.0


def plain_log_axis(ax, values) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    ticks = [t for t in (1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50) if finite.min() / 1.3 <= t <= finite.max() * 1.3]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def style_axis(ax, ylabel: str, title: str | None = None, room_for_labels: bool = True) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks(TIMEPOINTS)
    ax.set_xlim(-75, 17 if room_for_labels else 3)
    ax.set_xlabel("Hours relative to reference time")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def label_line_ends(ax, items: list[tuple[float, str]]) -> None:
    """Direct labels at t=0 in secondary ink, spread vertically so they never overlap."""
    items = sorted((y, text) for y, text in items if np.isfinite(y))
    if not items:
        return
    low, high = ax.get_ylim()
    gap = 0.06 * (high - low)
    placed = [items[0][0]]
    for y, _ in items[1:]:
        placed.append(max(y, placed[-1] + gap))
    overflow = placed[-1] - (high - 0.02 * (high - low))
    if overflow > 0:
        placed = [y - overflow for y in placed]
    for (_, text), y in zip(items, placed):
        ax.text(2.5, y, text, color=SECONDARY, fontsize=8, va="center")


def model_legend(fig, y: float = 1.0) -> None:
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], marker=MARKERS[m], linewidth=2, markersize=5, label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, y), labelcolor=INK)


def plot_models(ax, frame: pd.DataFrame, metric: str, linestyle="-", label_ends: bool = True) -> None:
    ends = []
    for model in MODEL_ORDER:
        sub = frame[frame.model.eq(model)].sort_values("monitoring_timepoint")
        if sub.empty:
            continue
        ax.plot(sub.monitoring_timepoint, sub[metric], color=MODEL_COLORS[model], marker=MARKERS[model], markersize=4.5, linewidth=2, linestyle=linestyle, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ends.append((sub.loc[sub.monitoring_timepoint.eq(0), metric].iloc[0], MODEL_SHORT[model]))
    if label_ends:
        label_line_ends(ax, ends)


def horizon_panels(metrics: pd.DataFrame, figures: Path, mode: str) -> list[Path]:
    written = []
    panels = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("ppv", "PPV"), ("nne", "NNE = 1/PPV (log scale)")]
    for horizon in cm.HORIZONS:
        frame = metrics[metrics.prediction_horizon.eq(horizon)]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
        for ax, (metric, label) in zip(axes.flat, panels):
            is_log = metric == "nne"
            plot_models(ax, frame, metric, label_ends=False)
            if is_log:
                ax.set_yscale("log")
                plain_log_axis(ax, frame.nne)
            elif metric == "auroc":
                ax.set_ylim(*auroc_limits(frame.auroc))
            else:
                ax.set_ylim(0, 1.02 if metric != "auprc" else min(1.0, np.nanmax(frame[metric]) * 1.15 + 0.02))
            style_axis(ax, label, f"{label}" + ("" if metric in ("auroc", "auprc") else " at frozen validation threshold"), room_for_labels=not is_log)
            if not is_log:
                label_line_ends(ax, [(frame[(frame.model == m) & (frame.monitoring_timepoint == 0)][metric].iloc[0], MODEL_SHORT[m]) for m in MODEL_ORDER])
        fig.suptitle(f"Prediction-horizon family {horizon} h — held-out test, {mode} reference, all eligible stays", x=0.01, ha="left", color=INK, fontsize=11)
        model_legend(fig, 0.955)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        path = figures / f"monitoring_metrics_h{horizon}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    paths, monitoring = cm.load_config()
    primary = monitoring["reference_time"]["primary"]
    figures, summary = paths["figures_root"], paths["summary_root"]
    metrics = pd.read_csv(paths["metrics_root"] / "continuous_monitoring_metrics.csv")
    slices = pd.read_csv(summary / "time_slice_validation.csv")
    faith = pd.read_csv(summary / "clinical_faithfulness_summary.csv")
    burden = pd.read_csv(summary / "alert_burden_summary.csv")
    static = pd.read_csv(paths["baseline_results_root"] / "metrics" / "baseline_results.csv").rename(columns={"prediction_window": "prediction_horizon"})
    main_frame = metrics[metrics.reference_mode.eq(primary) & metrics.cohort.eq("all_eligible")]

    written = horizon_panels(main_frame, figures, primary)

    # AUROC / AUPRC overview: rows = horizon family, columns = metric.
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    for row, horizon in enumerate(cm.HORIZONS):
        frame = main_frame[main_frame.prediction_horizon.eq(horizon)]
        for col, (metric, label) in enumerate([("auroc", "AUROC"), ("auprc", "AUPRC")]):
            ax = axes[row, col]
            plot_models(ax, frame, metric, label_ends=False)
            ax.set_ylim(auroc_limits(main_frame.auroc) if metric == "auroc" else (0, min(1.0, np.nanmax(main_frame.auprc) * 1.1 + 0.02)))
            style_axis(ax, label, f"{label} — {horizon} h models")
            label_line_ends(ax, [(frame[(frame.model == m) & (frame.monitoring_timepoint == 0)][metric].iloc[0], MODEL_SHORT[m]) for m in MODEL_ORDER])
    model_legend(fig, 0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    written.append(figures / "auroc_auprc_by_timepoint_overview.png")
    fig.savefig(written[-1], dpi=200)
    plt.close(fig)

    # Paper Fig. 2 layout: all 15 checkpoints on one AUROC axis (colour = model, line style = horizon).
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for horizon in cm.HORIZONS:
        plot_models(ax, main_frame[main_frame.prediction_horizon.eq(horizon)], "auroc", linestyle=HORIZON_STYLES[horizon], label_ends=False)
    ax.set_ylim(*auroc_limits(main_frame.auroc))
    style_axis(ax, "AUROC", f"AUROC trajectories, 5 models x 3 horizons ({primary} reference)", room_for_labels=False)
    ax.axhline(0.5, color=AXIS, linewidth=0.8, zorder=0)
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], marker=MARKERS[m], linewidth=2, label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    handles += [Line2D([0], [0], color=SECONDARY, linestyle=HORIZON_STYLES[h], linewidth=2, label=f"{h} h model") for h in cm.HORIZONS]
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    written.append(figures / "paper_fig2_auroc_all_checkpoints.png")
    fig.savefig(written[-1], dpi=200)
    plt.close(fig)

    # Eligible stays per time point (reference modes as small multiples).
    counts = slices[slices.prediction_horizon.eq(slices.prediction_horizon.min())]
    modes = list(monitoring["reference_time"]["modes"])
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4), sharey=True)
    for ax, mode in zip(np.atleast_1d(axes), modes):
        sub = counts[counts.reference_mode.eq(mode)].sort_values("monitoring_timepoint")
        ax.plot(sub.monitoring_timepoint, sub.n_non_aki_eligible, color="#2a78d6", marker="o", linewidth=2, label="Non-AKI")
        ax.plot(sub.monitoring_timepoint, sub.n_aki_eligible, color="#eb6834", marker="s", linewidth=2, label="AKI")
        style_axis(ax, "Eligible stays", f"{mode} reference", room_for_labels=False)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    written.append(figures / "eligible_stays_by_timepoint.png")
    fig.savefig(written[-1], dpi=200)
    plt.close(fig)

    # Reference-time sensitivity: matched vs discharge, per checkpoint.
    fig, axes = plt.subplots(3, 5, figsize=(17, 9.5), sharex=True, sharey=True)
    all_elig = metrics[metrics.cohort.eq("all_eligible")]
    for row, horizon in enumerate(cm.HORIZONS):
        for col, model in enumerate(MODEL_ORDER):
            ax = axes[row, col]
            for mode, color, marker in zip(modes, ["#2a78d6", "#eb6834"], ["o", "s"]):
                sub = all_elig[all_elig.reference_mode.eq(mode) & all_elig.prediction_horizon.eq(horizon) & all_elig.model.eq(model)].sort_values("monitoring_timepoint")
                ax.plot(sub.monitoring_timepoint, sub.auroc, color=color, marker=marker, markersize=4, linewidth=2, label=f"{mode} reference")
            style_axis(ax, "AUROC" if col == 0 else "", f"{MODEL_SHORT[model]} {horizon} h", room_for_labels=False)
            ax.set_ylim(*auroc_limits(all_elig.auroc))
            if row < 2:
                ax.set_xlabel("")
    handles = [Line2D([0], [0], color=c, marker=mk, linewidth=2, label=f"{m} reference") for m, c, mk in zip(modes, ["#2a78d6", "#eb6834"], ["o", "s"])]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    written.append(figures / "reference_time_sensitivity_auroc.png")
    fig.savefig(written[-1], dpi=200)
    plt.close(fig)


    # Clinical faithfulness heatmap: Mann-Kendall tau (diverging, grey midpoint = no trend).
    pf = faith[faith.reference_mode.eq(primary)].copy()
    pf["model_key"] = pf.model.map({v: k for k, v in cm.UPSTREAM_MODEL_NAMES.items()})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.06})
    for panel, (ax, (column, label)) in enumerate(zip(axes, [("auroc", "AUROC"), ("auprc", "AUPRC")])):
        grid_tau = np.full((len(MODEL_ORDER), 3), np.nan)
        for i, model in enumerate(MODEL_ORDER):
            for j, horizon in enumerate(cm.HORIZONS):
                row = pf[pf.model_key.eq(model) & pf.prediction_horizon.eq(horizon)]
                if row.empty:
                    continue
                tau, p = row[f"{column}_tau"].iloc[0], row[f"{column}_p"].iloc[0]
                grid_tau[i, j] = tau
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.text(j, i, f"蟿={tau:.2f}\n{stars}", ha="center", va="center", fontsize=8, color=INK)
        image = ax.imshow(grid_tau, cmap=TAU_CMAP, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(3), [f"{h} h" for h in cm.HORIZONS])
        ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER] if panel == 0 else [])
        ax.set_title(f"Mann-Kendall tau, {label} toward reference", color=INK, fontsize=10, loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Kendall 蟿 (positive = improves toward onset)")
    written.append(figures / "clinical_faithfulness_tau_heatmap.png")
    fig.savefig(written[-1], dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- tables
    results = metrics.merge(static[["model", "prediction_horizon", "auroc", "auprc"]].rename(columns={"auroc": "static_auroc", "auprc": "static_auprc"}), on=["model", "prediction_horizon"], how="left")
    results.to_csv(summary / "continuous_monitoring_results.csv", index=False)

    ranking_rows = []
    for horizon in cm.HORIZONS + ["all"]:
        s = static if horizon == "all" else static[static.prediction_horizon.eq(horizon)]
        m = main_frame if horizon == "all" else main_frame[main_frame.prediction_horizon.eq(horizon)]
        agg = m.groupby(["model", "prediction_horizon"]).agg(mean_continuous_auroc=("auroc", "mean"), auroc_t0=("auroc", lambda x: x[m.loc[x.index, "monitoring_timepoint"].eq(0)].iloc[0]), auroc_t_minus_72=("auroc", lambda x: x[m.loc[x.index, "monitoring_timepoint"].eq(-72)].iloc[0]), mean_continuous_auprc=("auprc", "mean")).reset_index()
        agg = agg.merge(s[["model", "prediction_horizon", "auroc", "auprc"]].rename(columns={"auroc": "static_auroc", "auprc": "static_auprc"}), on=["model", "prediction_horizon"])
        agg = agg.merge(pf[["model_key", "prediction_horizon", "auroc_tau", "auroc_p", "clinically_faithful"]].rename(columns={"model_key": "model"}), on=["model", "prediction_horizon"], how="left")
        agg["rank_static_auroc"] = agg.static_auroc.rank(ascending=False).astype(int)
        agg["rank_mean_continuous_auroc"] = agg.mean_continuous_auroc.rank(ascending=False).astype(int)
        agg["ranking_scope"] = f"horizon_{horizon}" if horizon != "all" else "all_15_checkpoints"
        ranking_rows.append(agg)
    ranking = pd.concat(ranking_rows, ignore_index=True)
    ranking.to_csv(summary / "static_vs_continuous_ranking.csv", index=False)

    def pivot(frame: pd.DataFrame, value: str) -> pd.DataFrame:
        table = frame.pivot_table(index=["prediction_horizon", "model"], columns="monitoring_timepoint", values=value).reindex(columns=TIMEPOINTS)
        table.columns = [f"{c:+d} h" if c else "0 h" for c in table.columns]
        order = pd.MultiIndex.from_tuples([(h, m) for h in cm.HORIZONS for m in MODEL_ORDER])
        return table.reindex(order).reset_index().rename(columns={"level_0": "horizon", "level_1": "model"})

    def md(frame: pd.DataFrame) -> str:
        return frame.to_markdown(index=False, floatfmt=".3f")

    frozen_cols = ["prediction_horizon", "model", "monitoring_timepoint", "N", "N_positive", "N_negative", "threshold", "sensitivity", "specificity", "ppv", "npv", "f1", "accuracy", "nne", "metric_notes"]
    frozen = main_frame[main_frame.monitoring_timepoint.isin([-72, -36, 0])][frozen_cols].sort_values(["prediction_horizon", "model", "monitoring_timepoint"])
    eligible = counts[["reference_mode", "monitoring_timepoint", "n_patients_eligible", "n_aki_eligible", "n_non_aki_eligible", "n_ineligible_cutoff_before_icu_admission", "n_ineligible_no_observation", "future_events_detected"]]
    faith_cols = ["reference_mode", "prediction_horizon", "model", "auroc_t_minus_72", "auroc_t0", "delta_auroc_72_to_0", "auroc_tau", "auroc_p", "auroc_trend", "auroc_sens_slope_per_12h", "auprc_tau", "auprc_p", "ppv_frozen_threshold_tau", "ppv_frozen_threshold_p", "clinically_faithful"]
    burden_cols = ["reference_mode", "prediction_horizon", "model", "frozen_threshold", "frozen_ppv_t0", "frozen_nne_t0", "frozen_fpr_t0", "frozen_nne_min_over_timepoints", "frozen_nne_max_over_timepoints", "frozen_pct_aki_alerted_any_timepoint", "frozen_pct_non_aki_alerted_any_timepoint", "frozen_mean_alerts_per_non_aki_patient", "frozen_median_lead_time_h", "upstream_thr05_nne_at_onset", "upstream_thr05_sensitivity_at_onset", "upstream_thr05_fpr_at_onset", "upstream_thr05_pct_non_aki_alerted"]
    discharge = metrics[metrics.reference_mode.eq("discharge") & metrics.cohort.eq("all_eligible")]
    lines = [
        "# Continuous monitoring results", "",
        f"Held-out test cohort of the rebuilt baseline; primary reference = **{primary}** (non-AKI pseudo-onset, upstream step 1). Metrics use every stay eligible at the time point (cutoff inside the ICU stay, >= 1 observation). Thresholds are the frozen validation-selected values. Undefined metrics are NaN with a reason in `metric_notes`.", "",
        "## Static baseline (rebuilt, for comparison)", "", md(static[["prediction_horizon", "model", "auroc", "auprc", "threshold", "n_test", "n_positive"]].sort_values(["prediction_horizon", "model"])), "",
        "## Eligible stays per time point", "", md(eligible), "",
        "## AUROC by time point (primary reference)", "", md(pivot(main_frame, "auroc")), "",
        "## AUPRC by time point (primary reference)", "", md(pivot(main_frame, "auprc")), "",
        "## Frozen-threshold metrics at -72, -36 and 0 h", "", md(frozen), "",
        "## Clinical faithfulness (upstream steps A-C)", "", md(faith[faith_cols]), "",
        "## Alert burden", "", md(burden[burden_cols]), "",
        "## Static vs continuous ranking", "", md(ranking[["ranking_scope", "model", "prediction_horizon", "static_auroc", "rank_static_auroc", "mean_continuous_auroc", "rank_mean_continuous_auroc", "auroc_t_minus_72", "auroc_t0", "auroc_tau", "clinically_faithful"]]), "",
        "## Sensitivity: discharge reference, AUROC", "", md(pivot(discharge, "auroc")), "",
        "## Figures", "", *[f"- `figures/{p.name}`" for p in written], "",
    ]
    (summary / "continuous_monitoring_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
