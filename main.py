from functools import partial

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import trim_mean
import ast

from fanda.wandb_client import fetch_history
from fanda.plot_utils import (
    plot_learning_curves,
    plot_interval_estimates,
    add_legend,
    save_fig,
)


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


def get_networks(row):
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


def load_data(difficulty):
    try:
        df = pd.read_parquet(f"data/popgym_{difficulty}.parquet")
    except FileNotFoundError:
        df = pd.read_json(f"data/popgym_{difficulty}.json")

    return df


def plot_popgym(difficulty=None):
    if difficulty is None:
        difficulties = ["easy", "medium", "hard"]
        dfs = []
        for diff in difficulties:
            dfs.append(load_data(diff))
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = load_data(difficulty)
        df = df[df["environment.env_id"].str.endswith(difficulty.capitalize())]

    metric = "evaluation/mmer"

    df["network"] = df.apply(get_networks, axis=1)

    df = filter_runs(df)

    df["seed"] = df.groupby(["network", "environment.env_id", "_step"]).cumcount()

    df["evaluation/mmer"] = df.groupby("environment.env_id")[metric].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    df = (
        df.groupby(["network", "seed", "_step"])[metric]
        .mean()
        .reset_index()
        .groupby(["network", "seed"])[metric]
        .max()
        .reset_index()
    )
    df.rename(columns={"evaluation/mmer": "MMER"}, inplace=True)

    df["network"] = df["network"].apply(lambda x: str(x).split(".")[-1])
    df = df.sort_values("MMER", ascending=False)

    xlabels = df["network"].unique().tolist()
    color_palette = sns.color_palette("colorblind", n_colors=len(xlabels))
    colors = dict(zip(xlabels, color_palette))

    fig, ax = plot_interval_estimates(
        df,
        x="MMER",
        y="network",
        hue="network",
        palette=colors,
        capsize=0.2,
        dodge=True,
        estimator=partial(trim_mean, proportiontocut=0.25),
        title="IQM",
        xlabel="Normalized MMER",
    )

    if difficulty is None:
        path = "plots/popgym"
    else:
        path = f"plots/popgym_{difficulty}"

    save_fig(fig, path)


def plot_bsuite():
    df = pd.read_parquet("data/bsuite_memory_chain.parquet")

    df["network"] = df.apply(get_networks, axis=1)

    df = filter_runs(df)

    color_palette = sns.color_palette("colorblind")
    xlabels = df["network"].unique().tolist()
    colors = dict(zip(xlabels, color_palette))

    fig, ax = plot_learning_curves(
        df,
        x="environment.env_params.memory_length",
        y="evaluation/mean_episode_returns",
        hue="network",
        xlabel="Memory Length",
        ylabel="IQM Episode Return",
        palette=colors,
        marker="o",
        markeredgewidth=0,
        estimator=partial(trim_mean, proportiontocut=0.25),
        errorbar=("ci", 95),
        err_kws={"alpha": 0.2},
    )

    ax = add_legend(
        ax,
        labels=xlabels,
        colors=colors,
    )

    # save_fig(fig, "plots/bsuite_memory_chain")

    lengths = [31, 63, 127, 255, 511]
    for length in lengths:
        fig, ax = plot_learning_curves(
            df[df["environment.env_params.memory_length"] == length],
            x="_step",
            y="evaluation/mean_episode_returns",
            hue="network",
            xlabel="Number of Frames (in millions)",
            ylabel="IQM Episode Return",
            palette=colors,
            estimator=partial(trim_mean, proportiontocut=0.25),
            errorbar=("ci", 95),
            err_kws={"alpha": 0.2},
        )
        ax = add_legend(
            ax,
            labels=xlabels,
            colors=colors,
        )
        save_fig(fig, f"plots/bsuite_memory_chain_{length}_step")


def get_data():
    api = wandb.Api()
    df = fetch_history(
        api,
        "noahfarr",
        "benchmarks",
        filters={
            "config.environment.env_id": {"$regex": "Hard"},
            "state": "finished",
        },
    )
    df.to_json("data/popgym_hard.json")


if __name__ == "__main__":
    # get_data()
    # plot_popgym("easy")
    # plot_popgym("medium")
    # plot_popgym("hard")
    # plot_popgym()
    plot_bsuite()
