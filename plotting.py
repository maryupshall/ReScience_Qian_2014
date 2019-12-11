from os.path import join

import matplotlib.pyplot as plt


def init_figure(size, dpi=96):
    """Create a figure of set size

    :param size: (l,w) in cm of the figure
    :param dpi: Display resolution (defaults 96 dpi)
    :return: None
    """
    plt.figure(figsize=size, dpi=dpi)


def save_fig(name, extension="pdf", figure_path="figures", figure_prefix="figure"):
    """Clean figure and save

    :param name: Figure name suffix
    :param extension: Export format (default pdf)
    :param figure_path: Folder to save figure (default figures)
    :param figure_prefix: prefix of figure (default figure)
    :return: None
    """
    plt.tight_layout()

    save_path = join(figure_path, figure_prefix + "_" + name + "." + extension)
    plt.savefig(save_path, format=extension)
    plt.close("all")


def make_panel_title(title):
    """Add the panel title to the panel

    :param title: Panel title
    :return: None
    """
    # Plot it at top left corner
    plt.text(
        0,
        1,
        title,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
    )


def set_properties(
    title,
    x_label="",
    y_label="",
    x_tick=(),
    y_tick=(),
    x_limits=None,
    y_limits=None,
    y_ticklabel=None,
    x_ticklabel=None,
):
    """Set plot properties

    :param title: Panel title
    :param x_label: x axis label defaults to ""
    :param y_label: y axis label defaults to ""
    :param x_tick: x axis tick marks - array of ticks
    :param y_tick: y axis tick marks - array of ticks
    :param x_limits: x axis limits  - none for matplotlib defaults or array of [min, max]
    :param y_limits: y axis limits  - none for matplotlib defaults or array of [min, max]
    :param y_ticklabel: y axis tick labels  - none for matplotlib defaults or [] for blank
    :param x_ticklabel: x axis tick labels  - none for matplotlib defaults or [] for blank
    :return: None
    """
    # Set x axis features
    plt.xlabel(x_label)
    plt.xticks(x_tick)
    plt.xlim(x_limits)
    if x_ticklabel is not None:
        plt.gca().set_xticklabels(x_ticklabel)

    # Set y axis features
    plt.ylabel(y_label)
    plt.yticks(y_tick)
    plt.ylim(y_limits)
    if y_ticklabel is not None:
        plt.gca().set_yticklabels(y_ticklabel)

    make_panel_title(title)
