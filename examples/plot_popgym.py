from functools import partial

import ast
import pandas as pd
import seaborn as sns
from scipy.stats import trim_mean

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, pointplot, add_legend, save_fig

DIFFICULTIES = ["Medium", "Hard"]

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

def restore_seeds(df):
    df["seed"] = df.groupby(["network", "environment.env_id", "_step"]).cumcount()
    return df


def main(difficulty):
    (
        fetch_wandb("noahfarr", "benchmarks", filters={
            "config.environment.env_id": {"$regex": difficulty},
            "state": "finished",
            "created_at": {"$gte": "2025-11-11"},
        })
        .pipe(get_networks)
        .pipe(filter_runs)
        .pipe(restore_seeds)
        .pipe(transforms.normalize, column="evaluation/mmer", groupby=["environment.env_id"])
        .pipe(lambda df: df.groupby(["network", "seed", "_step"])["evaluation/mmer"].mean().reset_index())
        .pipe(lambda df: df.groupby(["network", "seed"])["evaluation/mmer"].max().reset_index())
        .sort_values("evaluation/mmer", ascending=False)
        .pipe( 
            pointplot, 
            x="evaluation/mmer", 
            y="network", 
            hue="network",
            palette="colorblind",
            capsize=0.2,
            dodge=True,
            estimator=partial(trim_mean, proportiontocut=0.25),
        )
        .pipe(annotate_axis, xlabel="Normalized MMER", title="IQM", grid_alpha=0.25)
        .pipe(decorate_axis, ticklabelsize="xx-large", wrect=5, spines=["bottom"])
        .pipe(save_fig, name=f"plots/popgym_{difficulty.lower()}")

    )

if __name__ == "__main__":
    for difficulty in DIFFICULTIES:
        main(difficulty)

