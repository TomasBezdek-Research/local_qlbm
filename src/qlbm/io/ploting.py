import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from qlbm.config.roi import RegionOfInterest


def plot_res(
    numpy_results,
    vmin=None,
    vmax=None,
    figsize=(8, 6),
    colorbar_orientation="vertical",
    save_path=None,
    fontsize: int = 15,
    colorbar_only=False,
):
    """Plot the results as a heatmap using matplotlib.

    Args:
        numpy_results (np.ndarray): The numpy array of results to plot.
        vmin (float, optional): The lower bound of the color scale.
        vmax (float, optional): The upper bound of the color scale.
        figsize (tuple): Width and height of the figure in inches.
        colorbar_orientation (str, optional): 'vertical', 'horizontal', or None/False to omit.
        save_path (str, optional): File path to save the image (e.g., 'plot.png').
        fontsize (int): Fontsize for ticks and legend.
        colorbar_only (bool, optional): If True, renders only the colorbar/legend instead of the heatmap.
    """
    if colorbar_only:
        fig, ax = plt.subplots(figsize=figsize)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("coolwarm")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax, orientation=colorbar_orientation)
        cbar.ax.tick_params(labelsize=fontsize)
    else:
        data = numpy_results.T

        # Create figure with the specified size
        plt.figure(figsize=figsize)

        # Pass vmin and vmax to imshow to control the color scaling
        im = plt.imshow(data, cmap="coolwarm", origin="lower", vmin=vmin, vmax=vmax)

        # Handle colorbar options
        if colorbar_orientation == "vertical":
            cbar = plt.colorbar(im, orientation="vertical")
            cbar.ax.tick_params(labelsize=fontsize)
        elif colorbar_orientation == "horizontal":
            cbar = plt.colorbar(im, orientation="horizontal")
            cbar.ax.tick_params(labelsize=fontsize)

        plt.xticks(
            ticks=np.arange(data.shape[1]),
            labels=np.arange(data.shape[1]),
            fontsize=fontsize,
        )
        plt.yticks(
            ticks=np.arange(data.shape[0]),
            labels=np.arange(data.shape[0]),
            fontsize=fontsize,
        )

        # Add grid lines dynamically at half-integer positions between the main ticks
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, data.shape[1] - 0.5, 1), minor=True)
        ax.set_yticks(np.arange(0.5, data.shape[0] - 0.5, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Save image if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def plot_res_roi(
    numpy_results,
    vmin=None,
    vmax=None,
    figsize=(8, 6),
    colorbar_orientation="vertical",
    save_path=None,
    fontsize: int = 15,
    colorbar_only=False,
    roi: RegionOfInterest | None = None,
):
    """Plot the results as a heatmap using matplotlib.

    Args:
        numpy_results (np.ndarray): The numpy array of results to plot.
        vmin (float, optional): The lower bound of the color scale.
        vmax (float, optional): The upper bound of the color scale.
        figsize (tuple): Width and height of the figure in inches.
        colorbar_orientation (str, optional): 'vertical', 'horizontal', or None/False to omit.
        save_path (str, optional): File path to save the image (e.g., 'plot.png').
        fontsize (int): Fontsize for ticks and legend.
        colorbar_only (bool, optional): If True, renders only the colorbar/legend instead of the heatmap.
        roi (RegionOfInterest, optional): Region of interest to highlight.
    """
    if colorbar_only:
        fig, ax = plt.subplots(figsize=figsize)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("coolwarm")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax, orientation=colorbar_orientation)
        cbar.ax.tick_params(labelsize=fontsize)
    else:
        data = numpy_results.T

        # Create figure with the specified size
        plt.figure(figsize=figsize)

        # Pass vmin and vmax to imshow to control the color scaling
        im = plt.imshow(data, cmap="coolwarm", origin="lower", vmin=vmin, vmax=vmax)

        # Handle colorbar options
        if colorbar_orientation == "vertical":
            cbar = plt.colorbar(im, orientation="vertical")
            cbar.ax.tick_params(labelsize=fontsize)
        elif colorbar_orientation == "horizontal":
            cbar = plt.colorbar(im, orientation="horizontal")
            cbar.ax.tick_params(labelsize=fontsize)

        plt.xticks(
            ticks=np.arange(data.shape[1]),
            labels=np.arange(data.shape[1]),
            fontsize=fontsize,
        )
        plt.yticks(
            ticks=np.arange(data.shape[0]),
            labels=np.arange(data.shape[0]),
            fontsize=fontsize,
        )

        # Add grid lines dynamically at half-integer positions between the main ticks
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, data.shape[1] - 0.5, 1), minor=True)
        ax.set_yticks(np.arange(0.5, data.shape[0] - 0.5, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Highlight Region of Interest (ROI) if provided
        if roi is not None:
            if roi.is_subgrid:
                # bounds is a list of (low, high) tuples for each dimension (transposed data mapping: dim 0 -> x, dim 1 -> y)
                x_low, x_high = roi.bounds[0]
                y_low, y_high = roi.bounds[1]
                width = x_high - x_low
                height = y_high - y_low

                # Since origin="lower" and cells are centered on integers,
                # a cell at index `i` spans from `i - 0.5` to `i + 0.5`
                rect = patches.Rectangle(
                    (x_low - 0.45, y_low - 0.45),
                    width - 0.1,
                    height - 0.1,
                    linewidth=4,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                print("subgrid")

            elif roi.is_point_cloud:
                for pt in roi.points_set:
                    x, y = pt[0], pt[1]
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
                print("Cloud")

    # Save image if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
