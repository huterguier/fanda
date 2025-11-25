from functools import partial

import ast
import pandas as pd
import seaborn as sns
from scipy.stats import trim_mean

from fanda.plot_utils import (
    add_legend,
    save_fig,
)
from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import lineplot

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
    df, fig, ax = (
        fetch_wandb("noahfarr", "benchmarks", filters={
            "config.environment.env_id": "MemoryChain-bsuite",
            "state": "finished",
            "created_at": {"$gte": "2025-11-11"},
        })
        .pipe(get_networks)
        .pipe(filter_runs)
        .pipe(transforms.remove_outliers, column="evaluation/mean_episode_returns")
        .pipe( 
            lineplot, 
            x="environment.env_params.memory_length", 
            y="evaluation/mean_episode_returns", 
            hue="network",
            xlabel="Memory Length",
            ylabel="IQM Episode Return",
            palette="colorblind",
            marker="o",
            markeredgewidth=0,
            estimator=partial(trim_mean, proportiontocut=0.25),
            errorbar=("ci", 95),
            err_kws={"alpha": 0.2},
        )
    )

    xlabels = df["network"].unique()
    colors = sns.color_palette("colorblind", len(xlabels))
    colors = dict(zip(xlabels, colors))
    ax = add_legend(
        ax,
        labels=xlabels,
        colors=colors,
    )
    save_fig(fig, "plots/bsuite_memory_chain_piped")

if __name__ == "__main__":
    main()

