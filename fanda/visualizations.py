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


def decorate_axis(df, wrect=10, hrect=10, ticklabelsize="large", spines=None):
    """Helper function for decorating plots."""
    spines = spines or ["bottom", "left"]
    for spine in ["top", "right", "bottom", "left"]:
        if spine not in spines:
            df.attrs["ax"].spines[spine].set_visible(False)
        else:
            df.attrs["ax"].spines[spine].set_linewidth(2)

    # Deal with ticks and the blank space at the origin
    df.attrs["ax"].tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    df.attrs["ax"].spines["left"].set_position(("outward", hrect))
    df.attrs["ax"].spines["bottom"].set_position(("outward", wrect))
    return df


def annotate_axis(
    df,
    labelsize="x-large",
    xticks=None,
    xticklabels=None,
    yticks=None,
    grid_alpha=0.2,
    xlabel="",
    ylabel="",
    title="",
):
    """Annotates and decorates the plot."""
    df.attrs["ax"].set_xlabel(xlabel, fontsize=labelsize)
    df.attrs["ax"].set_ylabel(ylabel, fontsize=labelsize)
    df.attrs["ax"].set_title(title, fontsize=labelsize)
    if xticks is not None:
        df.attrs["ax"].set_xticks(ticks=xticks)
        df.attrs["ax"].set_xticklabels(xticklabels)
    if yticks is not None:
        df.attrs["ax"].set_yticks(yticks)
    df.attrs["ax"].grid(True, alpha=grid_alpha)
    return df


def lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize=(7, 5),
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

    df.attrs["fig"] = fig
    df.attrs["ax"] = ax
    return df


def pointplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    figwidth=3.4,
    row_height=0.37,
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
    df.attrs["fig"] = fig
    df.attrs["ax"] = ax
    return df

def add_legend(df, column, fontsize="x-large"):
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
        fontsize=fontsize,
        bbox_to_anchor=(0.5, 1.2),
    )
    return df

def save_fig(df, name):
    file_name = "{}.pdf".format(name)
    df.attrs["fig"].savefig(file_name, format="pdf", bbox_inches="tight")
    return df

