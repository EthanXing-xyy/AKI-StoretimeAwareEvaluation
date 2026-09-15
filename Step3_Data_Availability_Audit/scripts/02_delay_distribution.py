"""Summarize availability delay by laboratory feature and item identifier.

Scopes: rows of the frozen test cohort inside the ICU stay window (primary; each labevent_id counted
once even when an admission has two test stays), and every in-scope interim row (context).
Missing storetime, negative delays, and extreme delays are counted separately and never removed;
quantiles use every non-missing delay, including negative values.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402

STATS = """
    count(*) AS n_rows,
    count(delay_min) AS n_with_delay,
    count(*) - count(delay_min) AS n_missing_delay,
    quantile_cont(delay_min, 0.25) AS q1_min,
    quantile_cont(delay_min, 0.50) AS median_min,
    quantile_cont(delay_min, 0.75) AS q3_min,
    quantile_cont(delay_min, 0.75) - quantile_cont(delay_min, 0.25) AS iqr_min,
    quantile_cont(delay_min, 0.90) AS p90_min,
    quantile_cont(delay_min, 0.95) AS p95_min,
    100 * avg((delay_min > 30)::INT) AS pct_gt_30min,
    100 * avg((delay_min > 60)::INT) AS pct_gt_1h,
    100 * avg((delay_min > 120)::INT) AS pct_gt_2h
"""
ANOMALIES = """
    count(*) AS n_rows,
    sum(storetime_raw_missing::INT) AS n_storetime_missing_raw,
    sum((storetime IS NULL AND NOT coalesce(storetime_raw_missing, FALSE))::INT) AS n_storetime_unparseable,
    sum((storetime_raw_missing IS NULL)::INT) AS n_not_in_extract,
    sum((delay_min < 0)::INT) AS n_negative_delay,
    sum((delay_min = 0)::INT) AS n_zero_delay,
    sum((delay_min > 1440)::INT) AS n_gt_24h,
    sum((delay_min > 10080)::INT) AS n_gt_7d,
    min(delay_min) AS min_delay_min,
    max(delay_min) AS max_delay_min
"""


def grouped(con, source: str, keys: list[str], body: str, scope: str) -> pd.DataFrame:
    key_sql = ", ".join(keys)
    frame = con.execute(f"""
        SELECT {key_sql}, {body} FROM {source}
        GROUP BY GROUPING SETS (({key_sql}), ())
    """).df()
    for key in keys:
        frame[key] = frame[key].astype(object).where(frame[key].notna(), "ALL_LABS" if key == "feature" else "ALL")
    frame.insert(0, "scope", scope)
    return frame


def main() -> None:
    scope, used, no_source = ac.lab_feature_scope()
    test = ac.load_test_cohort()
    con = ac.connect()
    ac.register_test_labs(con, test, scope)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE test_rows AS
        SELECT DISTINCT labevent_id, itemid, feature, charttime, storetime, storetime_raw_missing, delay_min FROM test_labs
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE all_rows AS
        SELECT l.labevent_id, l.itemid, i.feature, l.charttime, r.storetime, r.storetime_raw_missing,
               date_diff('second', l.charttime, r.storetime) / 60.0 AS delay_min
        FROM read_parquet('{ac.posix(ac.INTERIM_LABS)}') l
        JOIN scope_items i ON l.itemid = i.itemid
        LEFT JOIN read_parquet('{ac.posix(ac.STORETIME)}') r ON l.labevent_id = r.labevent_id
    """)
    sources = {"test_cohort_icu_window": "test_rows", "all_interim_lab_rows": "all_rows"}

    by_feature = pd.concat([grouped(con, t, ["feature"], STATS, s) for s, t in sources.items()], ignore_index=True)
    by_itemid = pd.concat([grouped(con, t, ["feature", "itemid"], STATS, s) for s, t in sources.items()], ignore_index=True)
    by_itemid = by_itemid[~(by_itemid.feature.eq("ALL_LABS") ^ by_itemid.itemid.eq("ALL"))]
    anomalies = pd.concat([grouped(con, t, ["feature"], ANOMALIES, s) for s, t in sources.items()], ignore_index=True)
    examples = con.execute("""
        (SELECT 'negative_delay' AS kind, * FROM test_rows WHERE delay_min < 0 ORDER BY delay_min LIMIT 50)
        UNION ALL
        (SELECT 'largest_delay' AS kind, * FROM test_rows WHERE delay_min IS NOT NULL ORDER BY delay_min DESC LIMIT 25)
        UNION ALL
        (SELECT 'missing_storetime' AS kind, * FROM test_rows WHERE delay_min IS NULL LIMIT 50)
    """).df()
    con.close()

    order = by_feature[by_feature.scope.eq("test_cohort_icu_window") & ~by_feature.feature.eq("ALL_LABS")].sort_values("median_min", ascending=False).feature.tolist()
    rank = {f: i for i, f in enumerate(["ALL_LABS"] + order)}
    for frame in (by_feature, by_itemid, anomalies):
        frame.sort_values(["scope", "feature"], key=lambda s: s.map(rank) if s.name == "feature" else s, inplace=True, kind="stable")
    ac.write(by_feature, "lab_delay_by_feature.csv")
    ac.write(by_itemid, "lab_delay_by_itemid.csv")
    ac.write(anomalies, "delay_anomalies.csv")
    ac.write(examples, "delay_anomaly_examples.csv")
    ac.write(pd.DataFrame({"feature": used + no_source, "status": ["audited"] * len(used) + ["in_feature_list_no_lab_source"] * len(no_source)}), "lab_feature_scope.csv")
    show = by_feature[by_feature.scope.eq("test_cohort_icu_window")]
    print(show[["feature", "n_rows", "n_missing_delay", "median_min", "iqr_min", "p90_min", "p95_min", "pct_gt_30min", "pct_gt_1h", "pct_gt_2h"]].round(1).to_string(index=False))
    print(anomalies[anomalies.feature.eq("ALL_LABS")].to_string(index=False))


if __name__ == "__main__":
    main()
