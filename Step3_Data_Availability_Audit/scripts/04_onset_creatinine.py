"""Audit onset-defining creatinine availability and differential exposure.

aki_dt is min(charttime) of the first KDIGO-positive creatinine or urine row
(Step 1 build_mimic_cohort.py). With an inclusive slice (charttime <= cutoff) the
creatinine that defines onset is itself an input at timepoint 0 for creatinine-onset AKI stays.
This step measures how often that result was not yet stored (storetime proxy) at aki_dt, and
compares pending/last-value exposure between AKI and non-AKI stays at each timepoint.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402

DIFFERENTIAL_METRICS = {
    "stay_any_pending_lab": "any_pending",
    "stay_creatinine_pending": "creatinine_pending",
    "stay_any_last_value_changed": "any_last_value_changed",
    "stay_creatinine_last_value_changed": "creatinine_last_value_changed",
}


def onset_table(test: pd.DataFrame) -> pd.DataFrame:
    aki = test[test.label.eq(1)][["stay_id", "hadm_id", "intime", "outtime", "aki_dt"]]
    con = ac.connect()
    con.register("aki", aki)
    table = con.execute(f"""
        WITH cr AS (
          SELECT a.stay_id, max(k.aki_stage_creatinine) AS creatinine_stage
          FROM aki a JOIN read_parquet('{ac.posix(ac.INTERIM / "kdigo_creatinine.parquet")}') k
            ON a.stay_id = k.stay_id AND k.charttime = a.aki_dt AND k.aki_stage_creatinine > 0
          GROUP BY 1
        ), uo AS (
          SELECT a.stay_id, max(k.aki_stage_urine) AS urine_stage
          FROM aki a JOIN read_parquet('{ac.posix(ac.INTERIM / "kdigo_urine.parquet")}') k
            ON a.stay_id = k.stay_id AND k.charttime = a.aki_dt AND k.aki_stage_urine > 0
          GROUP BY 1
        ), defining AS (  -- rows behind kdigo_creatinine at aki_dt (itemid 50912, 0.1-150 mg/dL)
          SELECT a.stay_id, count(*) AS n_defining_rows, min(r.storetime) AS storetime_min, max(r.storetime) AS storetime_max,
                 sum(r.storetime_raw_missing::INT) AS n_missing_storetime, avg(l.valuenum) AS onset_creatinine
          FROM aki a
          JOIN read_parquet('{ac.posix(ac.INTERIM_LABS)}') l ON l.hadm_id = a.hadm_id AND l.charttime = a.aki_dt
          LEFT JOIN read_parquet('{ac.posix(ac.STORETIME)}') r ON l.labevent_id = r.labevent_id
          WHERE l.itemid = 50912 AND l.valuenum BETWEEN 0.1 AND 150
          GROUP BY 1
        )
        SELECT a.stay_id, a.hadm_id, a.intime, a.aki_dt, cr.creatinine_stage, uo.urine_stage,
               CASE WHEN cr.stay_id IS NOT NULL AND uo.stay_id IS NOT NULL THEN 'creatinine_and_urine'
                    WHEN cr.stay_id IS NOT NULL THEN 'creatinine' WHEN uo.stay_id IS NOT NULL THEN 'urine'
                    ELSE 'unresolved' END AS onset_source,
               d.n_defining_rows, d.onset_creatinine, d.storetime_min, d.storetime_max, d.n_missing_storetime,
               date_diff('second', a.aki_dt, d.storetime_max) / 60.0 AS onset_creatinine_delay_min
        FROM aki a LEFT JOIN cr USING (stay_id) LEFT JOIN uo USING (stay_id) LEFT JOIN defining d USING (stay_id)
        ORDER BY a.stay_id
    """).df()
    con.close()
    table["creatinine_defines_onset"] = table.onset_source.isin(["creatinine", "creatinine_and_urine"])
    pending = (table.creatinine_defines_onset & (table.storetime_max > table.aki_dt)).astype("boolean")
    pending[table.creatinine_defines_onset & table.storetime_max.isna()] = pd.NA  # unknown, not "available"
    table["onset_creatinine_pending_at_t0"] = pending
    return table


def main() -> None:
    test = ac.load_test_cohort()
    onset = onset_table(test)
    flags = pd.read_parquet(ac.WORK_ROOT / "Step3_Data_Availability_Audit/data/stay_timepoint_flags.parquet")
    for mode in ac.MODES:
        t0 = flags[flags["mode"].eq(mode) & flags.cohort.eq("all_eligible") & flags.monitoring_timepoint.eq(0)].set_index("stay_id")
        onset[f"{mode}_eligible_t0"] = onset.stay_id.isin(t0.index)
        onset[f"{mode}_t0_creatinine_last_value_changed"] = onset.stay_id.map(t0.creatinine_last_value_changed).astype("boolean")
    if onset.creatinine_defines_onset.any() and onset.loc[onset.creatinine_defines_onset, "n_defining_rows"].isna().any():
        raise RuntimeError("creatinine-onset stay without a defining lab row")
    ac.write(onset, "onset_creatinine_availability.csv")

    summary_rows = []
    for label, frame in {"all_test_aki": onset}.items():
        cr = frame[frame.creatinine_defines_onset]
        delay = cr.onset_creatinine_delay_min
        summary_rows.append({
            "subset": label, "n_aki": len(frame),
            "n_onset_creatinine_only": int(frame.onset_source.eq("creatinine").sum()),
            "n_onset_urine_only": int(frame.onset_source.eq("urine").sum()),
            "n_onset_creatinine_and_urine": int(frame.onset_source.eq("creatinine_and_urine").sum()),
            "n_onset_unresolved": int(frame.onset_source.eq("unresolved").sum()),
            "n_creatinine_defines_onset": len(cr),
            "n_onset_creatinine_missing_storetime": int(cr.storetime_max.isna().sum()),
            "n_onset_creatinine_pending_at_t0": int(cr.onset_creatinine_pending_at_t0.fillna(False).sum()),
            "pct_onset_creatinine_pending_at_t0": 100 * cr.onset_creatinine_pending_at_t0.fillna(False).mean() if len(cr) else float("nan"),
            "delay_median_min": delay.median(), "delay_q1_min": delay.quantile(0.25), "delay_q3_min": delay.quantile(0.75),
            "delay_p90_min": delay.quantile(0.90), "delay_p95_min": delay.quantile(0.95),
            "pct_delay_gt_30min": 100 * (delay > 30).mean() if len(cr) else float("nan"),
            "pct_delay_gt_1h": 100 * (delay > 60).mean() if len(cr) else float("nan"),
            "pct_delay_gt_2h": 100 * (delay > 120).mean() if len(cr) else float("nan"),
            "n_matched_t0_creatinine_last_value_changed": int(frame.matched_t0_creatinine_last_value_changed.fillna(False).sum()),
        })
    ac.write(pd.DataFrame(summary_rows), "onset_creatinine_summary.csv")

    diff_rows = []
    for (mode, cohort, timepoint), frame in flags.groupby(["mode", "cohort", "monitoring_timepoint"]):
        aki, non = frame[frame.grp.eq("AKI")], frame[frame.grp.eq("non-AKI")]
        for metric, column in DIFFERENTIAL_METRICS.items():
            k1, n1, k0, n0 = int(aki[column].sum()), len(aki), int(non[column].sum()), len(non)
            p = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1] if n1 and n0 else float("nan")
            diff_rows.append({
                "mode": mode, "cohort": cohort, "monitoring_timepoint": timepoint, "metric": metric,
                "aki_n": n1, "aki_k": k1, "aki_pct": 100 * k1 / n1 if n1 else float("nan"),
                "non_aki_n": n0, "non_aki_k": k0, "non_aki_pct": 100 * k0 / n0 if n0 else float("nan"),
                "diff_pp_aki_minus_non_aki": (100 * k1 / n1 if n1 else float("nan")) - (100 * k0 / n0 if n0 else float("nan")),
                "fisher_p": p,
            })
    diff = pd.DataFrame(diff_rows)
    ac.write(diff, "differential_availability_aki_vs_non_aki.csv")
    print(pd.DataFrame(summary_rows).T.to_string())
    view = diff[diff["mode"].eq("matched") & diff.cohort.eq("all_eligible") & diff.monitoring_timepoint.isin([0, -12, -24])]
    print(view.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
