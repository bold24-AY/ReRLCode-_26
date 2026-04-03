from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt


def save_data_and_plot(  # noqa: PLR0913
    data: Sequence[float | int],
    filename: str,
    xlabel: str,
    ylabel: str,
    out_folder: Path,
    dpi: int = 96,
) -> None:
    """Save numeric data to a text file and a corresponding line plot image.

    Args:
        data: Sequence of numeric values to plot and save.
        filename: Base name (without extension) for output files.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        out_folder: Directory where output files are written.
        dpi: Resolution in dots per inch for the saved plot image.
    """
    out_folder.mkdir(parents=True, exist_ok=True)

    plot_file = out_folder / f"plot_{filename}.png"
    data_file = out_folder / f"plot_{filename}_data.txt"

    min_val = min(data)
    max_val = max(data)
    margin_min = 0.05 * abs(min_val)
    margin_max = 0.05 * abs(max_val)

    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")
    except ImportError:
        pass # fallback to normal matplotlib if seaborn is missing

    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(data, label="Raw data", alpha=0.6, color="dodgerblue")
    
    # Add moving average for better visual trends
    if len(data) >= 10:
        window = max(5, len(data) // 20)
        import numpy as np
        if len(data) >= window:
            rolling_avg = np.convolve(data, np.ones(window)/window, mode='valid')
            # Pad with NaNs to align with original data
            padded_avg = np.concatenate((np.full(window - 1, np.nan), rolling_avg))
            ax.plot(padded_avg, label=f"Moving Avg ({window})", color="crimson", linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, len(data))
    ax.set_ylim(min_val - margin_min, max_val + margin_max)
    ax.set_title(f"{ylabel} vs {xlabel}", fontsize=22)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_file, dpi=dpi)
    plt.close(fig)

    data_text = "\n".join(str(value) for value in data) + "\n"
    data_file.write_text(data_text, encoding="utf-8")
