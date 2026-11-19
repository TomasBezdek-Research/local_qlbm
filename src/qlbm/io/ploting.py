from __future__ import annotations

from os import PathLike

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qlbm.config.roi import RegionOfInterest


def _plot_heatmap(
    numpy_results: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (8, 6),
    colorbar_orientation: str | None = "vertical",
    fontsize: int = 15,
    colorbar_only: bool = False,
) -> tuple[Figure, Axes]:
    """
    Create a heatmap or standalone colorbar.

    Args:
        numpy_results: Array containing the values to plot.
        vmin: Lower bound of the color scale.
        vmax: Upper bound of the color scale.
        figsize: Figure size in inches.
        colorbar_orientation: Colorbar orientation, or None to omit it.
        fontsize: Font size used for tick labels.
        colorbar_only: Whether to render only the colorbar.

    Returns:
        Figure and axes containing the generated plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if colorbar_only:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(
            cmap="coolwarm",
            norm=norm,
        )

        cbar = fig.colorbar(
            sm,
            cax=ax,
            orientation=colorbar_orientation,
        )
        cbar.ax.tick_params(labelsize=fontsize)

        return fig, ax

    # Transpose so dimension 0 maps to x and dimension 1 maps to y.
    data = numpy_results.T

    image = ax.imshow(
        data,
        cmap="coolwarm",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    if colorbar_orientation in ("vertical", "horizontal"):
        cbar = fig.colorbar(
            image,
            ax=ax,
            orientation=colorbar_orientation,
        )
        cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xticks(
        np.arange(data.shape[1]),
        labels=np.arange(data.shape[1]),
        fontsize=fontsize,
    )
    ax.set_yticks(
        np.arange(data.shape[0]),
        labels=np.arange(data.shape[0]),
        fontsize=fontsize,
    )

    # Draw cell boundaries at half-integer coordinates.
    ax.set_xticks(
        np.arange(0.5, data.shape[1] - 0.5, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(0.5, data.shape[0] - 0.5, 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    return fig, ax


def _highlight_roi(
    ax: Axes,
    roi: RegionOfInterest,
) -> None:
    """
    Highlight a two-dimensional region of interest on a heatmap.

    Args:
        ax: Axes containing the heatmap.
        roi: Region of interest to highlight.
    """
    if roi.is_subgrid:
        if len(roi.bounds) != 2:
            raise ValueError("ROI visualization requires a two-dimensional region.")

        x_low, x_high = roi.bounds[0]
        y_low, y_high = roi.bounds[1]

        # Slightly inset the outline so it remains visible inside the cell borders.
        rect = patches.Rectangle(
            (x_low - 0.45, y_low - 0.45),
            x_high - x_low - 0.1,
            y_high - y_low - 0.1,
            linewidth=4,
            edgecolor="red",
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(rect)

    elif roi.is_point_cloud:
        for point in roi.points_set:
            if len(point) != 2:
                raise ValueError("ROI visualization requires two-dimensional points.")

            x, y = point

            # Slightly inset the outline so it remains visible inside the cell borders.
            rect = patches.Rectangle(
                (x - 0.45, y - 0.45),
                0.9,
                0.9,
                linewidth=4,
                edgecolor="red",
                facecolor="none",
                zorder=10,
            )
            ax.add_patch(rect)


def plot_res(
    numpy_results: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (8, 6),
    colorbar_orientation: str | None = "vertical",
    save_path: str | PathLike[str] | None = None,
    fontsize: int = 15,
    colorbar_only: bool = False,
) -> None:
    """
    Plot simulation results as a heatmap.

    Args:
        numpy_results: Array containing the values to plot.
        vmin: Lower bound of the color scale.
        vmax: Upper bound of the color scale.
        figsize: Figure size in inches.
        colorbar_orientation: Colorbar orientation, or None to omit it.
        save_path: Optional path for saving the figure.
        fontsize: Font size used for tick labels.
        colorbar_only: Whether to render only the colorbar.
    """
    fig, _ = _plot_heatmap(
        numpy_results,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        colorbar_orientation=colorbar_orientation,
        fontsize=fontsize,
        colorbar_only=colorbar_only,
    )

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()


def plot_res_roi(
    numpy_results: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (8, 6),
    colorbar_orientation: str | None = "vertical",
    save_path: str | PathLike[str] | None = None,
    fontsize: int = 15,
    colorbar_only: bool = False,
    roi: RegionOfInterest | None = None,
) -> None:
    """
    Plot simulation results as a heatmap with an optional region of interest.

    Args:
        numpy_results: Array containing the values to plot.
        vmin: Lower bound of the color scale.
        vmax: Upper bound of the color scale.
        figsize: Figure size in inches.
        colorbar_orientation: Colorbar orientation, or None to omit it.
        save_path: Optional path for saving the figure.
        fontsize: Font size used for tick labels.
        colorbar_only: Whether to render only the colorbar.
        roi: Optional region of interest to highlight.
    """
    fig, ax = _plot_heatmap(
        numpy_results,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        colorbar_orientation=colorbar_orientation,
        fontsize=fontsize,
        colorbar_only=colorbar_only,
    )

    if roi is not None and not colorbar_only:
        _highlight_roi(ax, roi)

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()
