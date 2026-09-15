# %%
import numpy as np
import os
import pandas as pd

from IPython.display import display

pd.set_option("display.max_columns", None)

# %%

def get_cohort(
    data_path:str,
    max_predict_time:int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_master = (
        pd.read_parquet(
            f"{data_path}/master.parquet"
        )
        .assign(
            los = lambda df: (df["discharge_dt"] - df["admission_dt"]) / pd.Timedelta(1, 'h'),
            loo = lambda df: (df["aki_dt"] - df["admission_dt"]) / pd.Timedelta(1, 'h'),
            observable_dt = lambda df: (
                np.where(
                    df["loo"].notnull(),
                    df["aki_dt"],
                    df["discharge_dt"]
                )
            )
        )
        .pipe(
            lambda df: (display(df) or df)
        )
    )
    
    least_input_seq_pos = max_predict_time + 24
    least_input_seq_neg = 24

    df_control = (
        df_master
        .query(
            f"loo.isnull() & los >= {least_input_seq_neg}"
        )
    )
    
    df_case = (
        df_master
        .query(
            f"loo.notnull() & loo >= {least_input_seq_pos}"
        )
    )
    
    df_excluded_control = (
        df_master
        .query(
            f"loo.isnull() & los < {least_input_seq_neg}"
        )
    )
    
    df_excluded_case = (
        df_master
        .query(
            f"loo.notnull() & loo < {least_input_seq_pos}"
        )
    )
    
    df_include = (
        pd.concat(
            [
                df_control,
                df_case
            ]
        )
        .sort_values(
            ["pid", "visit_id"]
        )
        .reset_index(drop=True)
    )
    
    df_exclude = (
        pd.concat(
            [
                df_excluded_control,
                df_excluded_case
            ]
        )
        .sort_values(
            ["pid", "visit_id"]
        )
        .reset_index(drop=True)
    )
    
    return (
        df_include, df_exclude
    )

# %%
if __name__ == "__main__":
    
    data_path = "../../../data/raw/ilsan"
    
    df_include, df_exclude = get_cohort(
        data_path=data_path,
        max_predict_time=72
    )

# %%
