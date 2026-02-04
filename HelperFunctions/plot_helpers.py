import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def style_ax(ax, title, xlabel=None, ylabel=None, xticks=None):
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(alpha=0.3)


def add_percent_labels(ax, threshold=0.03):
    for p in ax.patches:
        w = p.get_width()
        if w > threshold:
            ax.text(
                p.get_x() + w / 2,
                p.get_y() + p.get_height() / 2,
                f"{w*100:.1f}%",
                ha="center",
                va="center",
                fontsize=8
            )
