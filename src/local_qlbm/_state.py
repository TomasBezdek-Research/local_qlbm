from typing import Dict

import numpy as np
from numpy.typing import ArrayLike
from qrisp import (
    QuantumFloat,
    QuantumModulus,
    multi_measurement,
)
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_carry_path import (
    QuantumBool,
)

from local_qlbm.types import BoundaryCondition, RegionOfInterest, VelocitySet


class LBMState:
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
        self._name = name
        self._grid_size = np.array(grid_size, dtype=np.int32)
        self._velocity_set = velocity_set
        self._dimensions = self._grid_size.size
        if not self._dimensions == velocity_set.dimensions:
            raise ValueError(
                f"Grid dimensions {self._dimensions} must match velocity set dimensions {velocity_set.dimensions}"
            )

        if len(boundary_conditions) != self._dimensions:
            raise ValueError(
                f"Boundary conditions must be specified for each dimension. Got {len(boundary_conditions)} conditions for {self._dimensions} dimensions."
            )
        self._boundary_conditions = boundary_conditions
        self._boundary_layers = np.array(
            [layer.aditional_layers for layer in boundary_conditions], np.int32
        )

        # Initialize the grid
        # Positions
        self._positions_register_shape = self._grid_size + self._boundary_layers
        self._positions_register = [
            QuantumModulus(size) for size in self._positions_register_shape
        ]
        self._position_register_sizes = np.array(
            [mod.size for mod in self._positions_register], dtype=np.int32
        )

        # Velocities
        self._init_velocity_register()

        self._total_size = (
            np.sum(self._position_register_sizes)
            + (self._velocity_qubits_per_dim + 1) * self._dimensions
        )

    def _init_velocity_register(self):
        # sign register - 0/False-negative direction, 1/True-positive direction
        self._sign_register = [QuantumBool() for _ in range(self._dimensions)]
        num_points = np.ceil(len(self._velocity_set.unique_coordinates) / 2)
        self._velocity_qubits_per_dim = int(np.ceil(np.log2(num_points)))

        if self._velocity_qubits_per_dim < 1:
            self._velocity_qubits_per_dim = 1

        # Initialize the velocity register
        self._velocity_register = [
            QuantumFloat(self._velocity_qubits_per_dim, signed=False)
            for _ in range(self._dimensions)
        ]

    #############################
    #   Measurement Functions   #
    #############################
    def _measure_density(self, shots=None, backend=None) -> Dict:
        """
        Measure the density of the grid.
        """
        measurement = multi_measurement(
            self._positions_register, shots=shots, backend=backend
        )
        return measurement

    def full_measurement(self, shots=None, backend=None) -> Dict:
        """
        Measure the full state of the grid.
        """
        measurement = multi_measurement(
            self._positions_register + self._sign_register + self._velocity_register,
            shots=shots,
            backend=backend,
        )
        return measurement

    def get_physical_density(self, shots=None, backend=None) -> Dict:
        """
        Measures density and strips away boundary layers/ghost cells.
        Returns a dict with user-space coordinate tuples as keys.

        Args:
            shots: Number of shots to measure.
            backend: Backend to use for measurement.

        Returns:
            A dict with the following keys:
                total_density: The total density of the region.
                cells: A dict of the cells in the region.
        """
        raw_data = self._measure_density(shots=shots, backend=backend)
        physical_data = {}

        for state_tuple, prob in raw_data.items():
            # Convert each dimension in the tuple
            user_coords = tuple(
                self._encoded_to_user_coord(state_tuple[d], d)
                for d in range(self._dimensions)
            )

            # Filter out ghost cells (coords < 0 or coords >= grid_size)
            is_inside = all(
                0 <= user_coords[d] < self._grid_size[d]
                for d in range(self._dimensions)
            )

            if is_inside:
                physical_data[user_coords] = physical_data.get(user_coords, 0) + prob

        return physical_data

    ####################
    #    Properties    #
    ####################
    @property
    def name(self):
        return self._name

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def velocity_set(self):
        return self._velocity_set

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def position_register_sizes(self):
        return self._position_register_sizes

    @property
    def positions_register(self):
        return self._positions_register

    @property
    def velocity_qubits_per_dim(self):
        return self._velocity_qubits_per_dim

    @property
    def velocity_register(self):
        return self._velocity_register

    @property
    def sign_register(self):
        return self._sign_register

    @property
    def total_size(self):
        return self._total_size

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    ##################
    #   Utilities    #
    ##################
    def results_to_numpy(self, analysis_results: Dict) -> np.ndarray:
        """Converts the 'cells' dict from analyze_region into a dense NumPy array.

        Args:
            analysis_results: The output of analyze_region.

        Returns:
            A NumPy array with the same shape as the grid.
        """
        grid = np.zeros(self._grid_size)
        cells = analysis_results.get("cells", {})
        for coords, val in cells.items():
            grid[coords] = val
        return grid

    ###################
    #   Coordinates   #
    ###################
    def _user_to_encoded_coord(self, user_coord, dimension):
        """Converts user 0-indexed coord to register index."""
        return user_coord + self._boundary_conditions[dimension].offset

    def _encoded_to_user_coord(self, encoded_coord, dimension):
        """Converts register index back to user 0-indexed coord."""
        return encoded_coord - self._boundary_conditions[dimension].offset

    def _get_encoded_bounds(self, user_bounds):
        """Converts list of user (min, max) tuples to encoded indices."""
        return [
            (self._user_to_encoded_coord(low, d), self._user_to_encoded_coord(high, d))
            for d, (low, high) in enumerate(user_bounds)
        ]

    def _get_encoded_point(self, user_point):
        """Converts user [x, y, ...] list to encoded indices."""
        return [
            self._user_to_encoded_coord(coord, d) for d, coord in enumerate(user_point)
        ]

    def _get_encoded_points(self, user_points):
        """Converts list of user [x, y, ...] tuples to encoded indices."""
        return [self._get_encoded_point(point) for point in user_points]

    def _encode_roi(self, user_roi: RegionOfInterest) -> RegionOfInterest:
        """Converts user-space ROI to register-space (encoded) ROI."""
        if user_roi.is_subgrid:
            encoded_bounds = self._get_encoded_bounds(user_bounds=user_roi.bounds)
            return RegionOfInterest(bounds=encoded_bounds)

        else:
            encoded_points = self._get_encoded_points(user_points=user_roi.points_list)
            return RegionOfInterest(points_list=encoded_points)
