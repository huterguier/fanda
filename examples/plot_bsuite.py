from functools import partial

import ast
import pandas as pd
import seaborn as sns
from scipy.stats import trim_mean

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend, save_fig

def filter_runs(df: pd.DataFrame) -> pd.DataFrame:
    latest_timestamps = (
        (
            df.groupby(
                [
                    "network",
                    "environment.env_id",
                    "seed",
                    "group",
                ]
            )["_timestamp"]
            .max()
            .reset_index()
        )
        .sort_values("_timestamp", ascending=False)
        .drop_duplicates(
            subset=[
                "network",
                "environment.env_id",
                "seed",
            ],
            keep="first",
        )
    )

    df = df[df["group"].isin(latest_timestamps["group"].unique())].copy()
    return df


def get_networks(df):

    def func(row):
        torso = str(row["algorithm.actor.torso._target_"])
        cell = str(row["algorithm.actor.torso.cell._target_"])
        pattern = str(row["algorithm.actor.torso.cell.pattern"])

        torso_name = torso.split(".")[-1]

        if torso_name == "RNN":
            if cell == "None" or pd.isna(cell):
                return torso_name

            cell_name = cell.split(".")[-1]

            if cell_name == "xLSTMCell":
                pattern = ast.literal_eval(pattern)
                cell_name = "".join(pattern) + "LSTM"
            return cell_name.replace("Cell", "")

        return torso_name

    df["network"] = df.apply(func, axis=1)
    return df


def main():
    (
        fetch_wandb("noahfarr", "benchmarks", filters={
            "config.environment.env_id": "MemoryChain-bsuite",
            "state": "finished",
            "created_at": {"$gte": "2025-11-11"},
        })
        .pipe(get_networks)
        .pipe(filter_runs)
        .pipe( 
            lineplot, 
            x="environment.env_params.memory_length", 
            y="evaluation/mean_episode_returns", 
            hue="network",
            palette="colorblind",
            marker="o",
            markeredgewidth=0,
            estimator=partial(trim_mean, proportiontocut=0.25),
            errorbar=("ci", 95),
            err_kws={"alpha": 0.2},
        )
        .pipe(
            annotate_axis, 
            xlabel="Memory Length",
            ylabel="IQM Episode Return",
            labelsize="xx-large",
        )
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(add_legend, column="network")
        .pipe(save_fig, name="plots/bsuite_memory_chain")
    )


if __name__ == "__main__":
    main()

