import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tls_sync.plotting import set_plot_format


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the .npz file produced by correlation_heatmap.py",
    )
    parser.add_argument(
        "corr_name",
        type=str,
        nargs="?",
        default=None,
        help="Correlation metric name (for example: pearson or plv)",
    )
    parser.add_argument(
        "--omega1",
        type=float,
        default=3.75,
        help="Value of omega1 used to draw the dashed reference lines",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path. Defaults to <corr_name>_heatmap.png",
    )
    return parser.parse_args(args)


def _load_heatmaps(path: str):
    with np.load(path, allow_pickle=True) as data:
        heatmaps = data["heatmaps"]
        if hasattr(heatmaps, "item"):
            try:
                heatmaps = heatmaps.item()
            except ValueError:
                pass

        if isinstance(heatmaps, np.ndarray) and heatmaps.dtype == object:
            if heatmaps.shape == ():
                heatmaps = heatmaps.item()
            elif heatmaps.size == 1:
                heatmaps = heatmaps[0]

        if isinstance(heatmaps, dict):
            return heatmaps, np.asarray(data["freq_list"])

        raise TypeError("The supplied .npz file does not contain heatmaps in the expected format")


def main(data_path: str, corr_name: str | None, omega1: float, output: str | None):
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    corr_name = (corr_name or data_path.stem.replace("_heatmap_data", "")).lower()
    heatmaps, freq_list = _load_heatmaps(str(data_path))

    set_plot_format(1.25, 1.25)
    fig, axes = plt.subplots(1, len(heatmaps), figsize=(12, 7), sharey=True)

    if len(heatmaps) == 1:
        axes = [axes]

    for ax, (solver_name, heatmap) in zip(axes, heatmaps.items()):
        data = np.asarray(np.vstack(heatmap))
        image = ax.imshow(
            data,
            origin="lower",
            extent=[freq_list[0], freq_list[-1], freq_list[0], freq_list[-1]],
            aspect="auto",
            cmap="inferno",
            vmin=-1 if corr_name == "pearson" else 0,
            vmax=1,
        )
        ax.set_title(solver_name)
        ax.set_xlabel(r"$\omega_d$", fontsize=22)
        ax.set_ylabel(r"$\omega_2$", fontsize=22)
        ax.vlines(
            x=omega1,
            color="c",
            ymin=freq_list[0],
            ymax=freq_list[-1],
            linestyle="--",
            linewidth=1,
        )
        ax.hlines(
            y=omega1,
            color="c",
            xmin=freq_list[0],
            xmax=freq_list[-1],
            linestyle="--",
            linewidth=1,
        )
        fig.colorbar(image, ax=ax, orientation="horizontal")

    fig.suptitle(
        f"{corr_name.capitalize() if corr_name == 'pearson' else corr_name.upper()} Correlation Coefficient Heatmaps"
    )

    plt.tight_layout()

    output_path = Path(output) if output is not None else Path(f"{corr_name}_heatmap.png")
    if not output_path.is_absolute():
        output_path = data_path.parent / output_path
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(
        data_path=args.data_path,
        corr_name=args.corr_name,
        omega1=args.omega1,
        output=args.output,
    )
