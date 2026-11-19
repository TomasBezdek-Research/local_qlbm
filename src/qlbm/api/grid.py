from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from qrisp import QuantumBool, QuantumFloat, QuantumModulus
from qrisp.interface.measurement_result import MultiMeasurementResult

from local_measurement.region_measurement import RegionEstimationInput
from qlbm.config.enums import BoundaryCondition, VelocitySet
from qlbm.config.model import LBMConfig
from qlbm.config.roi import RegionOfInterest
from qlbm.io.ploting import plot_res, plot_res_roi
from qlbm.io.vtk import export_density_to_vtk
from qlbm.kernels.boundary import apply_boundary_conditions
from qlbm.kernels.engine_jasp import (
    initialize_and_run,
    initialize_only,
    run_simulation,
    simulation_step,
)
from qlbm.kernels.initialization import initialize_state
from qlbm.kernels.streaming import streaming_step
from qlbm.measurement.measurement import full_measurement, measure_density
from qlbm.region_oracle.oracle import get_phase_oracle, get_region_oracle


class LBMGrid:
    """
    User-facing interface for configuring and running quantum LBM simulations.

    The grid stores the simulation configuration and manages the quantum
    registers used during initialization, simulation, measurement, and
    region-based analysis.

    The underlying quantum operations are implemented in `qlbm.kernels`.
    """

    def __init__(
        self,
        grid_size: ArrayLike,
        velocity_set: VelocitySet,
        boundary_conditions: list[BoundaryCondition],
        name: str | None = None,
    ) -> None:
        """
        Create a quantum LBM grid.

        Args:
            grid_size: Number of lattice cells along each spatial dimension.
            velocity_set: Discrete velocity set used by the simulation.
            boundary_conditions: Boundary conditions applied to the domain.
            name: Optional name of the simulation.
        """
        self._config = LBMConfig(
            _grid_size_input=np.asarray(grid_size, dtype=int),
            velocity_set=velocity_set,
            _boundary_conditions_input=boundary_conditions,
            name=name,
        )

        self._position_register: list[QuantumModulus] | None = None
        self._sign_register: list[QuantumBool] | None = None
        self._velocity_register: list[QuantumFloat] | None = None
        self._boundary_ancillas: list[QuantumBool] | None = None
        self._oracle_ancillas: list[QuantumBool] | None = None

    # ======================================================
    # INITIAL CONDITIONS
    # ======================================================

    def set_initial_state(
        self,
        bubble_roi: RegionOfInterest,
        velocity: ArrayLike,
    ) -> None:
        """
        Set the initial region and velocity of the simulation.

        Args:
            bubble_roi: Region of interest defining the initially occupied region.
            velocity: Initial velocity associated with the region.
        """
        self._config.set_initial_state(
            bubble=bubble_roi,
            velocity=np.asarray(velocity, dtype=int),
        )

    # ======================================================
    # INTERNAL HELPERS
    # ======================================================

    def _require_initialized(self) -> None:
        """Raise an error if the simulation registers have not been initialized."""
        if (
            self._position_register is None
            or self._sign_register is None
            or self._velocity_register is None
            or self._boundary_ancillas is None
            or self._oracle_ancillas is None
        ):
            raise RuntimeError(
                "Registers are not initialized. "
                "Call initialize(), run_simulation(init=True), or run_simulation()."
            )

    def _store_registers(
        self,
        position_register: list[QuantumModulus],
        sign_register: list[QuantumBool],
        velocity_register: list[QuantumFloat],
        boundary_ancillas: list[QuantumBool],
        oracle_ancillas: list[QuantumBool],
    ) -> None:
        """Store the quantum registers associated with the current simulation state."""
        self._position_register = position_register
        self._sign_register = sign_register
        self._velocity_register = velocity_register
        self._boundary_ancillas = boundary_ancillas
        self._oracle_ancillas = oracle_ancillas

    # ======================================================
    # LOW-LEVEL SIMULATION API
    # ======================================================

    def initialize(self, create_oracle_ancillas: bool = False) -> None:
        """
        Initialize the quantum registers and prepare the simulation state.

        Args:
            create_oracle_ancillas: Whether to allocate ancilla qubits required
                by region oracles.
        """
        (
            position_register,
            sign_register,
            velocity_register,
            boundary_ancillas,
            oracle_ancillas,
        ) = initialize_only(
            self._config,
            create_oracle_ancillas=create_oracle_ancillas,
        )

        self._store_registers(
            position_register,
            sign_register,
            velocity_register,
            boundary_ancillas,
            oracle_ancillas,
        )

    def run_simulation(self, iterations: int = 1, init: bool = True) -> None:
        """
        Run the simulation for a specified number of iterations.

        A new state is initialized when `init` is True or when no current
        simulation state exists. Otherwise, the existing state is advanced.

        Args:
            iterations: Number of simulation steps to perform.
            init: Whether to force initialization of a new simulation state.
        """
        if init or self._position_register is None:
            (
                position_register,
                sign_register,
                velocity_register,
                boundary_ancillas,
                oracle_ancillas,
            ) = initialize_and_run(
                self._config,
                iterations=iterations,
                create_oracle_ancillas=False,
            )

            self._store_registers(
                position_register,
                sign_register,
                velocity_register,
                boundary_ancillas,
                oracle_ancillas,
            )
            return

        self._require_initialized()

        run_simulation(
            self._config,
            self.position_register,
            self.velocity_register,
            self.sign_register,
            self.boundary_ancillas,
            iterations=iterations,
        )

    def step(self) -> None:
        """Perform one complete simulation step: streaming and boundary conditions."""
        self._require_initialized()

        simulation_step(
            self._config,
            self.position_register,
            self.velocity_register,
            self.sign_register,
            self.boundary_ancillas,
        )

    def stream(self) -> None:
        """Perform one streaming step without applying boundary conditions."""
        self._require_initialized()

        streaming_step(
            self._config,
            self.position_register,
            self.velocity_register,
            self.sign_register,
        )

    def apply_boundary_conditions(self) -> None:
        """Apply the configured boundary conditions to the current state."""
        self._require_initialized()

        apply_boundary_conditions(
            self._config,
            self.position_register,
            self.velocity_register,
            self.sign_register,
            self.boundary_ancillas,
        )

    # ======================================================
    # MEASUREMENT
    # ======================================================

    def measure_density(
        self,
        shots: int | None = None,
        backend: object | None = None,
    ) -> dict[tuple[int, ...], float]:
        """
        Measure the spatial density distribution in physical grid coordinates.

        Ghost-cell coordinates used internally by the quantum representation
        are removed from the returned result.

        Args:
            shots: Number of measurement shots. If None, use the backend default.
            backend: Backend used to execute the measurement.

        Returns:
            Mapping from physical grid coordinates to measured density.
        """

        self._require_initialized()

        if self._position_register is None:
            raise ValueError("Position register is not initialized")

        return measure_density(
            self._config,
            self._position_register,
            shots=shots,
            backend=backend,
        )

    def full_measurement(
        self,
        shots: int | None = None,
        backend: object | None = None,
    ) -> MultiMeasurementResult:
        """
        Jointly measure the position, sign, and velocity registers.

        Args:
            shots: Number of measurement shots. If None, use the backend default.
            backend: Backend used to execute the measurement.

        Returns:
            Joint measurement result for the position, sign, and velocity registers.
        """
        self._require_initialized()

        if self._position_register is None:
            raise ValueError("Position register is not initialized")
        if self._sign_register is None:
            raise ValueError("Sign register is not initialized")
        if self._velocity_register is None:
            raise ValueError("Velocity register is not initialized")

        return full_measurement(
            self._position_register,
            self._sign_register,
            self._velocity_register,
            shots=shots,
            backend=backend,
        )

    # ======================================================
    # CALLABLE RETURN UTILITIES
    # ======================================================
    def state_function(
        self,
        simulation_iterations: int = 1,
    ) -> Callable[..., None]:
        """
        Create a callable that prepares and evolves a simulation state.

        The returned callable expects flattened position, sign, velocity,
        and boundary-ancilla registers in this order.

        Args:
            simulation_iterations: Number of simulation iterations performed
                by the returned callable.

        Returns:
            Callable that initializes the supplied registers and advances
            the simulation.
        """

        def wrapper(*flat_args):
            d = self.dimensions

            pos = list(flat_args[0:d])
            sign = list(flat_args[d : 2 * d])
            vel = list(flat_args[2 * d : 3 * d])
            anc = list(flat_args[3 * d : 4 * d])

            initialize_state(self.config, pos, vel, sign)

            run_simulation(
                self.config,
                pos,
                vel,
                sign,
                anc,
                iterations=simulation_iterations,
            )

        return wrapper

    def qargs_factory(
        self,
        create_oracle_ancillas: bool = True,
    ) -> Callable[[], tuple[Any, ...]]:
        """
        Create a factory that allocates the quantum arguments for the simulation.

        The returned factory allocates position, sign, velocity, boundary-ancilla,
        and optionally oracle-ancilla registers, followed by a target qubit.

        Args:
            create_oracle_ancillas: Whether to allocate ancillas required by
                region oracles.

        Returns:
            Callable that allocates and returns the flattened quantum arguments.
        """

        def factory():
            from qlbm.kernels.registers import allocate_registers

            pos, sign, vel, anc, oracle = allocate_registers(
                self.config,
                create_oracle_ancillas=create_oracle_ancillas,
            )

            target = QuantumBool()

            return (
                *pos,
                *sign,
                *vel,
                *anc,
                *oracle,
                target,
            )

        return factory

    def flag_oracle(
        self,
        roi: RegionOfInterest,
    ) -> Callable[..., None]:
        """
        Create a bit-flip oracle that flags positions inside a region of interest.

        Args:
            roi: Region of interest to encode in the oracle.

        Returns:
            Callable implementing the region-membership oracle.
        """
        encoded_roi = self.config.encode_roi(roi)

        def oracle(*flat_args):
            d = self.dimensions

            pos = list(flat_args[0:d])
            oracle_ancillas = list(flat_args[-1 - d : -1])
            target = flat_args[-1]

            get_region_oracle(
                config=self.config,
                roi=encoded_roi,
                position_register=pos,
                oracle_ancillas=oracle_ancillas,
                target_qubit=target,
            )

        return oracle

    def phase_oracle(
        self,
        roi: RegionOfInterest,
    ) -> Callable[..., None]:
        """
        Create a phase oracle for positions inside a region of interest.

        Args:
            roi: Region of interest to encode in the oracle.

        Returns:
            Callable implementing the region phase oracle.
        """

        encoded_roi = self.config.encode_roi(roi)

        def oracle(*flat_args):
            d = self.dimensions

            pos = list(flat_args[0:d])
            oracle_ancillas = list(flat_args[-1 - d : -1])

            target = flat_args[-1]

            get_phase_oracle(
                config=self.config,
                roi=encoded_roi,
                position_register=pos,
                oracle_ancillas=oracle_ancillas,
                target_qubit=target,
            )

        return oracle

    def classical_classifier(
        self,
        roi: RegionOfInterest,
    ) -> Callable[..., bool]:
        """
        Create a classical classifier for membership in a region of interest.

        Args:
            roi: Region against which points are tested.

        Returns:
            Callable returning whether a point lies inside the region.
        """

        def wrapper(point) -> bool:
            return roi.contains(point)

        return wrapper

    def region_estimation_problem(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
    ) -> RegionEstimationInput:
        """
        Construct a region-estimation problem for the given region of interest.

        Args:
            roi: Region whose probability or density is to be estimated.
            simulation_iterations: Number of simulation iterations included
                in the state preparation.

        Returns:
            Region-estimation input containing the quantum arguments, state
            preparation function, and region oracle.
        """

        return RegionEstimationInput(
            qargs=self.qargs_factory(),
            state_function=self.state_function(
                simulation_iterations=simulation_iterations,
            ),
            flag_oracle=self.flag_oracle(roi),
        )

    def sampling_measurement_function(
        self,
    ) -> Callable[..., dict[tuple[int, ...], float]]:
        """
        Create a callable for measuring density from supplied quantum registers.

        Returns:
            Callable that measures the position registers and returns the
            corresponding physical density distribution.
        """

        def wrapper(
            *flat_args, shots: int | None = None, backend: object | None = None
        ) -> dict[tuple[int, ...], float]:
            pos = list(flat_args[: self.dimensions])
            return measure_density(self._config, pos, shots, backend)

        return wrapper

    # ======================================================
    # NUMPY / EXPORT UTILITIES
    # ======================================================

    def physical_density_to_numpy(
        self,
        physical_density: dict[tuple[int, ...], float],
    ) -> np.ndarray:
        """
        Convert a sparse physical-density mapping to a dense NumPy array.

        Args:
            physical_density: Mapping from physical grid coordinates to density values.

        Returns:
            Dense array with shape equal to the physical grid size.
        """
        grid = np.zeros(tuple(self._config.grid_size.tolist()), dtype=float)

        for coords, val in physical_density.items():
            grid[coords] = val

        return grid

    def results_to_numpy(
        self,
        analysis_results: dict,
    ) -> np.ndarray:
        """
        Convert analysis results containing cell values to a dense NumPy array.

        The input dictionary is expected to contain a `cells` mapping from
        grid coordinates to values.

        Args:
            analysis_results: Analysis result dictionary containing a `cells` mapping.

        Returns:
            Dense array with shape equal to the physical grid size.
        """
        grid = np.zeros(tuple(self._config.grid_size.tolist()), dtype=float)

        cells = analysis_results.get("cells", {})
        for coords, val in cells.items():
            grid[coords] = val

        return grid

    def plot_numpy_results(
        self,
        numpy_results: np.ndarray,
        vmin: float | None = None,
        vmax: float | None = None,
        figsize=(8, 6),
        colorbar_orientation="vertical",
        save_path=None,
        fontsize: int = 15,
        colorbar_only: bool = False,
    ) -> None:
        """
        Plot simulation results as a heatmap.

        Args:
            numpy_results: Array of values to plot.
            vmin: Lower bound of the color scale.
            vmax: Upper bound of the color scale.
            figsize: Figure size in inches.
            colorbar_orientation: Orientation of the colorbar, or None/False to omit it.
            save_path: Optional path to save the figure.
            fontsize: Font size used for ticks and labels.
            colorbar_only: Whether to render only the colorbar.
        """

        plot_res(
            numpy_results,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            colorbar_orientation=colorbar_orientation,
            save_path=save_path,
            fontsize=fontsize,
            colorbar_only=colorbar_only,
        )

    def plot_numpy_results_roi(
        self,
        numpy_results: np.ndarray,
        vmin: float | None = None,
        vmax: float | None = None,
        figsize=(8, 6),
        colorbar_orientation="vertical",
        save_path=None,
        fontsize: int = 15,
        colorbar_only: bool = False,
        roi: RegionOfInterest | None = None,
    ) -> None:
        """
        Plot simulation results as a heatmap with an optional region of interest.

        Args:
            numpy_results: Array of values to plot.
            vmin: Lower bound of the color scale.
            vmax: Upper bound of the color scale.
            figsize: Figure size in inches.
            colorbar_orientation: Orientation of the colorbar, or None/False to omit it.
            save_path: Optional path to save the figure.
            fontsize: Font size used for ticks and labels.
            colorbar_only: Whether to render only the colorbar.
            roi: Optional region of interest to highlight.
        """
        plot_res_roi(
            numpy_results,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            colorbar_orientation=colorbar_orientation,
            save_path=save_path,
            fontsize=fontsize,
            colorbar_only=colorbar_only,
            roi=roi,
        )

    def run_simulation_with_visualization(
        self,
        filename: PathLike[str] | str,
        iterations: int = 1,
        init: bool = True,
        time_steps_per_frame: int = 1,
    ) -> None:
        """
        Run the simulation while periodically exporting density snapshots as VTK files.

        This method is intended for visualization rather than the standard simulation
        workflow. The quantum state is measured after every `time_steps_per_frame`
        simulation steps, and each measured density field is exported as a VTK file.

        Existing VTK files matching the output filename prefix are removed before
        the simulation starts.

        Args:
            filename: Base path used for the generated VTK files.
            iterations: Number of density snapshots to generate after the initial state.
            init: Whether to initialize the simulation before generating snapshots.
            time_steps_per_frame: Number of simulation steps between consecutive snapshots.
        """
        full_path = Path(filename)
        target_dir = full_path.parent
        base_name = full_path.name

        search_pattern = f"{base_name}*.vts"
        for file in target_dir.glob(search_pattern):
            file.unlink()

        if init:
            self.initialize()

        density = self.measure_density()
        density_grid = self.physical_density_to_numpy(density)
        export_density_to_vtk(f"{filename}_{0:03d}", density_grid)

        for i in range(iterations):
            for _ in range(time_steps_per_frame):
                self.step()

            density = self.measure_density()
            density_grid = self.physical_density_to_numpy(density)
            export_density_to_vtk(
                f"{filename}_{(time_steps_per_frame * (i + 1)):03d}",
                density_grid,
            )

    # ======================================================
    # PROPERTIES
    # ======================================================

    @property
    def name(self) -> str | None:
        """Return the optional name of the simulation."""
        return self._config.name

    @property
    def grid_size(self) -> np.ndarray:
        """Return the physical grid size."""
        return self._config.grid_size

    @property
    def velocity_set(self) -> VelocitySet:
        """Return the discrete velocity set used by the simulation."""
        return self._config.velocity_set

    @property
    def dimensions(self) -> int:
        """Return the number of spatial dimensions."""
        return self._config.dimensions

    @property
    def velocity_qubits_per_dim(self) -> int:
        """Return the number of velocity qubits used per spatial dimension."""
        return self._config.velocity_qubits_per_dim

    @property
    def positions_register_shape(self) -> tuple[int, ...]:
        """Return the shape of the position quantum registers."""
        return self._config.positions_register_shape

    @property
    def total_size(self) -> int:
        """Return the total number of qubits required by the simulation."""
        return self._config.total_size

    @property
    def position_register(self) -> list[QuantumModulus]:
        """Return the initialized position registers."""
        assert self._position_register is not None, "Registers not initialized"
        return self._position_register

    @property
    def velocity_register(self) -> list[QuantumFloat]:
        """Return the initialized velocity registers."""
        assert self._velocity_register is not None, "Registers not initialized"
        return self._velocity_register

    @property
    def sign_register(self) -> list[QuantumBool]:
        """Return the initialized sign registers."""
        assert self._sign_register is not None, "Registers not initialized"
        return self._sign_register

    @property
    def boundary_ancillas(self) -> list[QuantumBool]:
        """Return the ancilla qubits used for boundary handling."""
        assert self._boundary_ancillas is not None, "Registers not initialized"
        return self._boundary_ancillas

    @property
    def oracle_ancillas(self) -> list[QuantumBool]:
        """Return the ancilla qubits used by region oracles."""
        assert self._oracle_ancillas is not None, "Registers not initialized"
        return self._oracle_ancillas

    @property
    def config(self) -> LBMConfig:
        """Return the underlying simulation configuration."""
        return self._config

    # ======================================================
    # DISPLAY
    # ======================================================

    def __repr__(self) -> str:
        """Return a concise developer-oriented representation of the grid."""
        return (
            f"LBMGrid(grid_size={self._config.grid_size.tolist()}, "
            f"velocity_set={self._config.velocity_set!r}, "
            f"dimensions={self._config.dimensions})"
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the grid configuration."""
        return (
            f"--- Quantum LBM Grid: {self._config.name or 'Unnamed'} ---\n"
            f"Dimensions : {self._config.dimensions}D\n"
            f"Grid Size  : {' x '.join(map(str, self._config.grid_size))}\n"
            f"Velocity   : {self._config.velocity_set.name}\n"
            f"Total Size : {self._config.total_size} qubits\n"
            f"--------------------------------------"
        )
