import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker
from matplotlib import rcParams
from matplotlib import rc
import seaborn as sns

sns.set_style("white")

rcParams["legend.loc"] = "best"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"

rc("text", usetex=False)


def decorate_axis(ax, wrect=10, hrect=10, ticklabelsize="large"):
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))
    return ax


def annotate_and_decorate_axis(
    ax,
    labelsize="x-large",
    ticklabelsize="x-large",
    xticks=None,
    xticklabels=None,
    yticks=None,
    legend=False,
    grid_alpha=0.2,
    legendsize="x-large",
    xlabel="",
    ylabel="",
    wrect=10,
    hrect=10,
):
    """Annotates and decorates the plot."""
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, alpha=grid_alpha)
    ax = decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
    if legend:
        ax.legend(fontsize=legendsize)
    return ax


def add_legend(ax, labels, colors):
    fake_patches = [mpatches.Patch(color=colors[label], alpha=0.75) for label in labels]
    legend = ax.legend(
        fake_patches,
        labels,
        loc="upper center",
        fancybox=True,
        ncol=min(len(labels), 5),
        fontsize="x-large",
        bbox_to_anchor=(0.5, 1.2),
    )
    return legend


def lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize=(7, 5),
    xlabel="",
    ylabel="",
    **kwargs,
):

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.lineplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        **kwargs,
    )

    ax = annotate_and_decorate_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        labelsize="xx-large",
        ticklabelsize="xx-large",
        legend=True,
    )
    df.attrs["fig"] = fig
    df.attrs["ax"] = ax
    return df


def pointplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    figwidth=3.4,
    row_height=0.37,
    title="",
    xlabel="",
    **kwargs,
):
    figsize = (figwidth, row_height * len(df[y].unique()))
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.pointplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        **kwargs,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax = decorate_axis(
        ax,
        ticklabelsize="xx-large",
        wrect=5,
    )
    ax.set_title(title, fontsize="xx-large")
    ax.tick_params(axis="both", which="major")
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(
        x=0.5,
        y=-0.25,
        s=xlabel,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize="xx-large",
    )
    df.attrs["fig"] = fig
    df.attrs["ax"] = ax
    return df

def add_legend(df, column):
    labels = df[column].unique()
    colors = sns.color_palette("colorblind", len(labels))
    colors = dict(zip(labels, colors))
    fake_patches = [mpatches.Patch(color=colors[label], alpha=0.75) for label in labels]
    df.attrs["ax"].legend(
        fake_patches,
        labels,
        loc="upper center",
        fancybox=True,
        ncol=min(len(labels), 5),
        fontsize="x-large",
        bbox_to_anchor=(0.5, 1.2),
    )
    return df

def save_fig(df, name):
    file_name = "{}.pdf".format(name)
    df.attrs["fig"].savefig(file_name, format="pdf", bbox_inches="tight")
    return df

