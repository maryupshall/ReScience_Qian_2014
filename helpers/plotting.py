from os.path import join

import matplotlib.pyplot as plt


def init_figure(size=(2, 2), dpi=96):
    plt.figure(figsize=size, dpi=dpi)


def save_fig(name, extension='pdf', figure_path='figures', figure_prefix='figure'):
    plt.tight_layout()

    plt.savefig(join(figure_path, figure_prefix + "_" + name + '.' + extension), format=extension)
    plt.close('all')


def make_legend(legend):
    plt.legend(legend, fontsize=8)


def set_properties(title, x_label="", y_label="", x_tick=None, y_tick=None, x_limits=None, y_limits=None,
                   y_ticklabel=None, x_ticklabel=None):
    plot_label_size = 12
    tick_label_size = 10

    plt.xlabel(x_label, size=plot_label_size)
    if x_ticklabel is not None:
        plt.gca().set_xticklabels(x_ticklabel)

    plt.ylabel(y_label, size=plot_label_size)
    if y_ticklabel is not None:
        plt.gca().set_yticklabels(y_ticklabel)

    if x_tick is None:
        x_tick = []
    if y_tick is None:
        y_tick = []

    plt.xticks(x_tick, size=tick_label_size)
    plt.yticks(y_tick, size=tick_label_size)

    plt.xlim(x_limits)
    plt.ylim(y_limits)

    plt.text(0, 1, title, horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.tight_layout()
