import matplotlib.patches as mpatches

def save_fig(fig, name):

    file_name = "{}.pdf".format(name)

    fig.savefig(file_name, format="pdf", bbox_inches="tight")

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

