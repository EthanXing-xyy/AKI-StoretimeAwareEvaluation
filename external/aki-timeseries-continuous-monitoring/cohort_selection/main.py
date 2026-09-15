# %%
import os
import pandas as pd
import argparse

from src.select_cohort import get_cohort

pd.set_option("display.max_columns", None)
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../../data")
    args = parser.parse_args()

    for hospital_name in [
        "ilsan",
        "cchlmc",
        "mimic-iv"
    ]:
        data_path = f"{args.root_path}/raw/{hospital_name}"
        save_path = f"{args.root_path}/processed/{hospital_name}/cohort"
        
        os.makedirs(save_path, exist_ok=True)
        
        df_include, df_exclude = get_cohort(
            data_path=data_path,
            max_predict_time=72
        )
        
        df_include.to_parquet(f"{save_path}/master_include.parquet")
        df_exclude.to_parquet(f"{save_path}/master_exclude.parquet")
# %%