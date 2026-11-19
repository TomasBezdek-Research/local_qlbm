from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
from pyevtk.hl import gridToVTK


def export_density_to_vtk(
    filename: PathLike[str] | str,
    data: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """
    Export a scalar density field as a VTK structured grid.

    One-, two-, and three-dimensional density arrays are supported.
    Lower-dimensional data is expanded to three dimensions as required
    by the VTK representation.

    Args:
        filename: Output filename without the VTK extension.
        data: Density values defined on the grid cells.
        spacing: Grid spacing along the x, y, and z directions.
        origin: Coordinates of the grid-cell centers at index zero.
    """
    shape = data.shape
    dims = len(shape)

    # Expand to 3D if needed (VTK requirement)
    full_shape = list(shape) + [1] * (3 - dims)
    nx, ny, nz = full_shape

    dx, dy, dz = spacing
    x0, y0, z0 = origin

    def get_coords(n: int, delta: float, start: float) -> np.ndarray:
        """Return cell-boundary coordinates for one grid dimension."""
        if n > 1:
            return np.arange(n + 1) * delta + (start - 0.5 * delta)
        return np.array([start])

    x = get_coords(nx, dx, x0)
    y = get_coords(ny, dy, y0)
    z = get_coords(nz, dz, z0)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    data_3d = data.reshape(full_shape)

    gridToVTK(
        str(filename),
        X,
        Y,
        Z,
        cellData={"Density": data_3d},
    )


def clear_vtk_series(
    filename: PathLike[str] | str,
) -> None:
    """
    Remove VTK files belonging to an existing output series.

    All `.vts` files in the target directory whose names start with the
    provided base filename are removed.

    Args:
        filename: Base filename of the VTK output series, without extension.
    """

    full_path = Path(filename)
    target_dir = full_path.parent
    base_name = full_path.name

    search_pattern = f"{base_name}*.vts"

    for file in target_dir.glob(search_pattern):
        file.unlink()
