import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def info_dtypes_hist(df,title):
    """
    Plot a histogram of pandas DataFrame column dtypes.
    """
    dtype_counts = df.dtypes.value_counts()

    ax = dtype_counts.plot(kind="bar",figsize=(5,3))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

    ax.set_xlabel("Data type")
    ax.set_ylabel("Number of columns")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.show()
