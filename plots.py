import matplotlib as mpl         # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def use_tex():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.serif": ["Computer Modern"]}
    )
    # for e.g. \text command
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def set_ax_info(ax, xlabel, ylabel, title=None, legend=True):
    """Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        xlabel (str): the desired label on the x-axis
        ylabel (str): the desired label on the y-axis
        title (str, optional): the desired title on the axis
            default: None
        legend (bool, optional): whether or not to add labels/legend
            default: True
    """
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    if title is not None:
        ax.set_title(title, fontsize=20)
    if legend:
        ax.legend(fontsize=15)
