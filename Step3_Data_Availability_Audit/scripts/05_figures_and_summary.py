"""Generate aggregate figures and the public data-availability audit summary."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402


def main() -> None:
    ac.FIGURES.mkdir(parents=True, exist_ok=True)
    ac.SUMMARY.mkdir(parents=True, exist_ok=True)
    delay = pd.read_csv(ac.TABLES / "lab_delay_by_feature.csv")
    pending = pd.read_csv(ac.TABLES / "pending_by_timepoint_feature_group.csv")
    differential = pd.read_csv(ac.TABLES / "differential_availability_aki_vs_non_aki.csv")
    onset = pd.read_csv(ac.TABLES / "onset_creatinine_summary.csv")

    feature = delay[delay.scope.eq("test_cohort_icu_window") & ~delay.feature.eq("ALL_LABS")].sort_values("median_min")
    figure, axis = plt.subplots(figsize=(8, 9))
    axis.barh(feature.feature, feature.median_min, color="#0072B2")
    axis.set(xlabel="Median storetime - charttime delay (minutes)", ylabel="Laboratory feature")
    figure.tight_layout()
    figure.savefig(ac.FIGURES / "lab_delay_by_feature.png", dpi=200)
    plt.close(figure)

    line = pending[
        pending["mode"].eq("matched")
        & pending.cohort.eq("all_eligible")
        & pending.feature.eq("ALL_LABS")
        & pending.grp.isin(["AKI", "non-AKI"])
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    for group, color in [("AKI", "#D55E00"), ("non-AKI", "#0072B2")]:
        part = line[line.grp.eq(group)].sort_values("monitoring_timepoint")
        axis.plot(part.monitoring_timepoint, part.pct_pending_rows, marker="o", label=group, color=color)
    axis.set(xlabel="Hours relative to reference", ylabel="Pending laboratory rows (%)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(ac.FIGURES / "pending_by_timepoint_matched.png", dpi=200)
    plt.close(figure)

    all_delay = delay[delay.scope.eq("test_cohort_icu_window") & delay.feature.eq("ALL_LABS")].iloc[0]
    t0 = pending[
        pending["mode"].eq("matched")
        & pending.cohort.eq("all_eligible")
        & pending.monitoring_timepoint.eq(0)
        & pending.grp.eq("all")
        & pending.feature.eq("ALL_LABS")
    ].iloc[0]
    onset_row = onset[onset.subset.eq("all_test_aki")].iloc[0]
    diff_t0 = differential[
        differential["mode"].eq("matched")
        & differential.cohort.eq("all_eligible")
        & differential.monitoring_timepoint.eq(0)
    ]
    report = [
        "# Laboratory result-availability audit",
        "",
        "This is a data audit only. It asks whether a mismatch exists between laboratory `charttime` and the `storetime` availability proxy. It does not test whether model performance changes; that question is evaluated in Step 4.",
        "",
        "`storetime` is a proxy for result availability, not the exact time a clinician viewed a result.",
        "",
        "## Aggregate findings",
        "",
        f"- Laboratory rows in the held-out ICU window: {int(all_delay.n_rows):,}.",
        f"- Median delay: {all_delay.median_min:.1f} minutes (IQR {all_delay.q1_min:.1f}-{all_delay.q3_min:.1f}).",
        f"- Rows delayed more than one hour: {all_delay.pct_gt_1h:.1f}%.",
        f"- At the primary 0 h cutoff, {int(t0.n_pending_rows):,} of {int(t0.n_visible_rows):,} charttime-visible rows ({t0.pct_pending_rows:.2f}%) were not yet stored.",
        f"- Onset-defining creatinine not yet stored at 0 h: {int(onset_row.n_onset_creatinine_pending_at_t0)}/{int(onset_row.n_creatinine_defines_onset)}.",
        "",
        "## Prespecified decision criteria",
        "",
        pd.read_csv(ac.TABLES / "decision_criteria.csv").to_markdown(index=False),
        "",
        "## Stay-level differential availability at 0 h",
        "",
        diff_t0.to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    (ac.SUMMARY / "data_availability_audit.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
