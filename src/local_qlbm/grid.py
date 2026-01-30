from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike
from qrisp import (
    IQAE,
    QuantumFloat,
    QuantumModulus,
    amplitude_amplification,
    auto_uncompute,
    multi_measurement,
    prepare,
    z,
)
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_carry_path import (
    QuantumBool,
)

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
    name : str, optional
        The name of the grid.
    """

    def __init__(
        self,
        grid_size: ArrayLike,
        velocity_set: VelocitySet,
        boundary_conditions: list[BoundaryCondition],
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

    ###################
    #   Measurement   #
    ###################
    def measure_density(self):
        """Measures the simulation"""
        self._state.get_physical_density()

    #############################
    #   Local region analysis   #
    #############################
    def analyze_region(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        eps: float = 0.01,
        shots: int = 1000,
        backend=None,
    ) -> Dict:
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
            roi, simulation_iterations, eps, shots, backend
        )

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
