from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from pyevtk.hl import gridToVTK

from ._analyzer import LBMAnalyzer
from ._engine import LBMEngine
from ._state import LBMState
from .types import BoundaryCondition, RegionOfInterest, VelocitySet


class LBMGrid:
    """
    Class representing a quantum LBM grid.

    Parameters
    ----------
    grid_size : ArrayLike
        The size of the grid.
    velocity_set : VelocitySet
        The set of velocities to use.
    boundary_conditions : list[BoundaryCondition] | BoundaryCondition
        The boundary conditions to use. If a single condition is provided, it will be
        applied to all dimensions.
    name : str, optional
        The name of the grid.
    """

    def __init__(
        self,
        grid_size: ArrayLike,
        velocity_set: VelocitySet,
        boundary_conditions: list[BoundaryCondition] | BoundaryCondition,
        name=None,
    ):
        # Initialize internal components
        self._state = LBMState(grid_size, velocity_set, boundary_conditions, name)
        self._engine = LBMEngine(self._state)
        self._analyzer = LBMAnalyzer(self._state, self._engine)

    def set_initial_state(
        self, bubble_roi: RegionOfInterest, velocity: ArrayLike
    ) -> None:
        """User sets the config; stored in state.

        Args:
            bubble_roi (RegionOfInterest): The region of interest for the bubble.
            velocity (ArrayLike): The velocity of the bubble.
        """
        self._state.set_initial_state(bubble_roi, velocity)

    ############################
    #   Low level simulation   #
    ############################
    def run_simulation(self, iterations=1, init=True):
        """Runs the simulation for a given number of iterations.

        Args:
            iterations (int, optional): The number of iterations to run. Defaults to 1.
            init (bool, optional): Whether to initialize the simulation. Defaults to True.
        """
        self._engine.run_simulation(iterations, init)

    def apply_conditions(self):
        """Sets initial conditions for the simulation"""
        self._engine.apply_initialization()

    def stream(self):
        """One stream step in simulation"""
        self._engine.stream()

    def apply_boundary_conditions(self):
        """Applies boundary conditions to the simulation"""
        self._engine.apply_boundary_conditions()

    #########################################
    #   Run simulation with visualization   #
    #########################################
    def run_simulation_with_visualization(
        self,
        filename,
        iterations=1,
        init=True,
        time_steps_per_frame=1,
    ):
        """Runs the simulation for a given number of iterations with visualization."""
        full_path = Path(filename)
        target_dir = full_path.parent
        base_name = full_path.name
        search_pattern = f"{base_name}*.vts"
        for file in target_dir.glob(search_pattern):
            file.unlink()
            print(f"Deleted: {file}")

        if init:
            self._engine.apply_initialization()
        dens = self.measure_density()
        dens_grid = self._state.physical_density_to_numpy(dens)
        self._export_to_vtk(f"{filename}_{0:03d}", dens_grid)

        for i in range(iterations):
            for _ in range(time_steps_per_frame):
                self._engine.stream()
                self._engine.apply_boundary_conditions()
            dens = self.measure_density()
            dens_grid = self._state.physical_density_to_numpy(dens)
            self._export_to_vtk(
                f"{filename}_{(time_steps_per_frame * (i + 1)):03d}", dens_grid
            )

    def _export_to_vtk(
        self, filename, data, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
    ):
        shape = data.shape
        dims = len(shape)

        # Expand to 3D shape
        full_shape = list(shape) + [1] * (3 - dims)
        nx, ny, nz = full_shape
        dx, dy, dz = spacing
        x0, y0, z0 = origin

        # Logic: Only add a boundary (+1) if the dimension > 1
        # This prevents 2D from becoming a "thin 3D box"
        def get_coords(n, delta, start):
            if n > 1:
                # Multi-cell dimension: Offset for cell centers
                return np.arange(n + 1) * delta + (start - 0.5 * delta)
            else:
                # Flat dimension: Keep exactly at the origin point
                return np.array([start])

        x = get_coords(nx, dx, x0)
        y = get_coords(ny, dy, y0)
        z = get_coords(nz, dz, z0)

        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Reshape data
        data_3d = data.reshape(full_shape)

        # Export
        gridToVTK(filename, X, Y, Z, cellData={"Density": data_3d})

        print(f"Saved: {filename}.vts")

    ###################
    #   Measurement   #
    ###################
    def measure_density(self) -> dict:
        """Measures the simulation"""
        return self._state.get_physical_density()

    #############################
    #   Local region analysis   #
    #############################
    def analyze_region(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        eps: float = 0.01,
        alpha: float = 0.05,
        shots: int | None = None,
        backend=None,
    ) -> dict:
        """
        1. Estimates total density (f) in ROI using IQAE.
        2. Amplifies ROI to make internal distribution measurable.
        3. Measures cells and rescales so their sum equals f.

        Args:
            roi: RegionOfInterest object.
            simulation_iterations: Number of iterations to run the simulation.
            eps: The desired accuracy of the estimate.
            shots: Number of shots to measure.

        Returns:
            A dict with the following keys:
                total_density: The total density of the region.
                cells: A dict of the cells in the region.
        """
        return self._analyzer.analyze_region(
            roi=roi,
            simulation_iterations=simulation_iterations,
            eps=eps,
            alpha=alpha,
            shots=shots,
            backend=backend,
        )

    def amplify_region(
        self,
        roi: RegionOfInterest,
        amplification_iterations: int = 1,
        simulation_iterations: int = 1,
    ) -> None:
        """
        Amplifies the region to make internal distribution measurable.

        Args:
            roi: RegionOfInterest object.
            amplification_iterations: Number of iterations to run the amplification.
            simulation_iterations: Number of iterations to run the simulation.
        """
        self._analyzer.amplify_region(
            roi=roi,
            amplification_iterations=amplification_iterations,
            simulation_iterations=simulation_iterations,
        )

    def estimate_region_density(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        eps: float = 0.01,
        alpha: float = 0.05,
        shots: int | None = None,
        backend=None,
    ) -> float:
        """
        1. Estimates total density (f) in ROI using IQAE.
        2. Amplifies ROI to make internal distribution measurable.

        Args:
            roi: RegionOfInterest object.
            eps: The desired accuracy of the estimate.
            alpha: The desired confidence level of the estimate.
            shots: Number of shots to measure.
            backend: The backend to use for the simulation.

        Returns:
            The estimated density of the region.
        """
        return float(
            self._analyzer.estimate_region_density(
                roi=roi,
                simulation_iterations=simulation_iterations,
                eps=eps,
                alpha=alpha,
                shots=shots,
                backend=backend,
            )
        )

    def fast_region_density_estimate(self, roi: RegionOfInterest):
        return self._analyzer.fast_region_density_estimate(roi)

    ####################
    #    Properties    #
    ####################
    @property
    def name(self):
        return self._state.name

    @property
    def grid_size(self):
        return self._state.grid_size

    @property
    def velocity_set(self):
        return self._state.velocity_set

    @property
    def dimensions(self):
        return self._state.dimensions

    @property
    def position_register_sizes(self):
        return self._state.position_register_sizes

    @property
    def positions_register(self):
        return self._state.positions_register

    @property
    def velocity_qubits_per_dim(self):
        return self._state.velocity_qubits_per_dim

    @property
    def velocity_register(self):
        return self._state.velocity_register

    @property
    def sign_register(self):
        return self._state.sign_register

    @property
    def total_size(self):
        return self._state.total_size

    ######################
    #   Visualization    #
    ######################
    def results_to_numpy(self, simulation_results):
        return self._state.results_to_numpy(simulation_results)

    def physical_density_to_numpy(self, physical_density):
        """
        Converts the 'cells' dict from analyze_region into a dense NumPy array.

        Args:
            physical_density: The dictionary of physical density.

        Returns:
            A NumPy array with the same shape as the grid.
        """
        return self._state.physical_density_to_numpy(physical_density)

    ################
    #   Display    #
    ################
    def __repr__(self):
        """Developer-friendly representation."""
        # We use !r to get the repr of the velocity_set (e.g., VelocitySet.D2Q9)
        return (
            f"LBMGrid(grid_size={self._state.grid_size.tolist()}, "
            f"velocity_set={self._state.velocity_set!r}, "
            f"dimensions={self._state.dimensions})"
        )

    def __str__(self):
        """User-friendly summary."""
        # This provides a clean 'dashboard' view of the grid
        return (
            f"--- Quantum LBM Grid: {self._state.name or 'Unnamed'} ---\n"
            f"Dimensions : {self._state.dimensions}D\n"
            f"Grid Size  : {' x '.join(map(str, self._state.grid_size))}\n"
            f"Velocity   : {self._state.velocity_set.name}\n"
            f"Total Size : {self._state.total_size} qubits\n"
            f"--------------------------------------"
        )
