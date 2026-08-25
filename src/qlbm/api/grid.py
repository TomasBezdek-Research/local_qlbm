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
    User-facing API for the quantum LBM solver.

    This class stores:
    - static simulation configuration (`LBMConfig`)
    - the currently active quantum registers (after initialization/run)

    The actual quantum evolution logic lives in `local_qlbm.kernels.*`.
    """

    def __init__(
        self,
        grid_size: ArrayLike,
        velocity_set: VelocitySet,
        boundary_conditions: list[BoundaryCondition],
        name: str | None = None,
    ) -> None:
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
        self._config.set_initial_state(
            bubble=bubble_roi,
            velocity=np.asarray(velocity, dtype=int),
        )

    # ======================================================
    # INTERNAL HELPERS
    # ======================================================

    def _require_initialized(self) -> None:
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
        Allocate registers and prepare the initial state, but do not
        advance the simulation in time.
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
        Run the simulation for a given number of iterations.

        If `init=True`, registers are freshly allocated and initialized.
        If `init=False`, the simulation continues from the current state.
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
        """
        Perform one full simulation step:
        streaming + boundary conditions.
        """
        self._require_initialized()

        simulation_step(
            self._config,
            self.position_register,
            self.velocity_register,
            self.sign_register,
            self.boundary_ancillas,
        )

    def stream(self) -> None:
        """
        Perform one streaming step only.
        """
        self._require_initialized()

        streaming_step(
            self._config,
            self.position_register,
            self.velocity_register,
            self.sign_register,
        )

    def apply_boundary_conditions(self) -> None:
        """
        Apply boundary conditions only.
        """
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
        Measure position density and convert it back to physical coordinates
        (ghost cells removed).

        Args:
            shots (int | None, optional): Number of shots to run. Defaults to None.
            backend (object | None, optional): Backend to run the simulation on. Defaults to None.

        Returns:
            dict[tuple[int, ...], float]: Dictionary of physical coordinates and their corresponding density.
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
        Measure positions + signs + velocity registers together.
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
        def wrapper(point) -> bool:
            return roi.contains(point)

        return wrapper

    def region_estimation_problem(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
    ) -> RegionEstimationInput:

        return RegionEstimationInput(
            qargs=self.qargs_factory(),
            state_function=self.state_function(
                simulation_iterations=simulation_iterations,
            ),
            flag_oracle=self.flag_oracle(roi),
        )

    def sampling_measurement_function(
        self,
        shots: int | None = None,
        backend: object | None = None,
    ) -> Callable[..., dict[tuple[int, ...], float]]:
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
        Convert sparse physical density dictionary to dense NumPy array.
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
        Convert analysis result dict with key 'cells' to dense NumPy array.
        """
        grid = np.zeros(tuple(self._config.grid_size.tolist()), dtype=float)

        cells = analysis_results.get("cells", {})
        for coords, val in cells.items():
            grid[coords] = val

        return grid

    def run_simulation_with_visualization(
        self,
        filename: PathLike[str] | str,
        iterations: int = 1,
        init: bool = True,
        time_steps_per_frame: int = 1,
    ) -> None:
        """
        Run the simulation and export density snapshots as VTK files.

        It works just as a visualization tool, and does not reflect the actual simulation workflow, because
        it does the measurement after each time step, not at the end of the simulation.
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
        return self._config.name

    @property
    def grid_size(self) -> np.ndarray:
        return self._config.grid_size

    @property
    def velocity_set(self) -> VelocitySet:
        return self._config.velocity_set

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    @property
    def velocity_qubits_per_dim(self) -> int:
        return self._config.velocity_qubits_per_dim

    @property
    def positions_register_shape(self) -> tuple[int, ...]:
        return self._config.positions_register_shape

    @property
    def total_size(self) -> int:
        return self._config.total_size

    @property
    def position_register(self) -> list[QuantumModulus]:
        assert self._position_register is not None, "Registers not initialized"
        return self._position_register

    @property
    def velocity_register(self) -> list[QuantumFloat]:
        assert self._velocity_register is not None, "Registers not initialized"
        return self._velocity_register

    @property
    def sign_register(self) -> list[QuantumBool]:
        assert self._sign_register is not None, "Registers not initialized"
        return self._sign_register

    @property
    def boundary_ancillas(self) -> list[QuantumBool]:
        assert self._boundary_ancillas is not None, "Registers not initialized"
        return self._boundary_ancillas

    @property
    def oracle_ancillas(self) -> list[QuantumBool]:
        assert self._oracle_ancillas is not None, "Registers not initialized"
        return self._oracle_ancillas

    @property
    def config(self) -> LBMConfig:
        return self._config

    # ======================================================
    # DISPLAY
    # ======================================================

    def __repr__(self) -> str:
        return (
            f"LBMGrid(grid_size={self._config.grid_size.tolist()}, "
            f"velocity_set={self._config.velocity_set!r}, "
            f"dimensions={self._config.dimensions})"
        )

    def __str__(self) -> str:
        return (
            f"--- Quantum LBM Grid: {self._config.name or 'Unnamed'} ---\n"
            f"Dimensions : {self._config.dimensions}D\n"
            f"Grid Size  : {' x '.join(map(str, self._config.grid_size))}\n"
            f"Velocity   : {self._config.velocity_set.name}\n"
            f"Total Size : {self._config.total_size} qubits\n"
            f"--------------------------------------"
        )
