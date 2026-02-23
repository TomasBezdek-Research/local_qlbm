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

from .types import BoundaryCondition, RegionOfInterest, VelocitySet


class LBMState:
    """
    Class representing a quantum LBM state.

    Parameters
    ----------
    grid_size : ArrayLike
        The size of the grid.
    velocity_set : VelocitySet
        The set of velocities to use.
    boundary_conditions : list[BoundaryCondition] | BoundaryCondition
        The boundary conditions to use.
    name : str, optional
        The name of the grid.
    """

    _initial_bubble: RegionOfInterest
    _initial_velocity: ArrayLike

    def __init__(
        self,
        grid_size: ArrayLike,
        velocity_set: VelocitySet,
        boundary_conditions: list[BoundaryCondition] | BoundaryCondition,
        name=None,
    ):
        self._name = name
        self._grid_size = np.array(grid_size, dtype=int)
        self._velocity_set = velocity_set
        self._dimensions = self._grid_size.size
        if not self._dimensions == velocity_set.dimensions:
            raise ValueError(
                f"Grid dimensions {self._dimensions} must match velocity set dimensions {velocity_set.dimensions}"
            )

        if isinstance(boundary_conditions, BoundaryCondition):
            boundary_conditions = [boundary_conditions] * self._dimensions
        if len(boundary_conditions) != self._dimensions:
            raise ValueError(
                f"Boundary conditions must be specified for each dimension. Got {len(boundary_conditions)} conditions for {self._dimensions} dimensions."
            )
        self._boundary_conditions = boundary_conditions
        self._boundary_layers = [
            self._velocity_set.aditional_layers if bc.need_offset else 0
            for bc in self._boundary_conditions
        ]
        self._boundary_offsets = [
            self._velocity_set.offset if bc.need_offset else 0
            for bc in self._boundary_conditions
        ]

        # Initialize the ancilla indexes
        self._boundary_ancilla_indexes = {}
        idx = 0
        for i, bc in enumerate(self._boundary_conditions):
            if (
                bc == BoundaryCondition.BOUNCE_BACK
                or bc == BoundaryCondition.SIMPLE_SLIP
            ):
                self._boundary_ancilla_indexes[i] = idx
                idx += 1

        # Initialize the grid
        # Positions
        self._positions_register_shape = self._grid_size + self._boundary_layers

        # Velocities
        self._init_velocity_properties()

        # Initialize the registers
        self._init_registers()

        self._position_register_sizes = np.array(
            [mod.size for mod in self._positions_register], dtype=int
        )
        self._total_size = (
            np.sum(self._position_register_sizes)
            + (self._velocity_qubits_per_dim + 1) * self._dimensions
        )

    def _init_velocity_properties(self):

        num_points = np.ceil(len(self._velocity_set.unique_coordinates) / 2)
        self._velocity_qubits_per_dim = int(np.ceil(np.log2(num_points)))

        if self._velocity_qubits_per_dim < 1:
            self._velocity_qubits_per_dim = 1

    def _init_registers(self):
        self._boundary_ancillas = []
        for _ in self._boundary_ancilla_indexes.keys():
            self._boundary_ancillas.append(QuantumBool())

        self._positions_register = [
            QuantumModulus(int(size)) for size in self._positions_register_shape
        ]
        # sign register - 0/False-negative direction, 1/True-positive direction
        self._sign_register = [QuantumBool() for _ in range(self._dimensions)]
        # Initialize the velocity register
        self._velocity_register = [
            QuantumFloat(self._velocity_qubits_per_dim, signed=False)
            for _ in range(self._dimensions)
        ]

    def set_registers(
        self,
        position_register: list,
        velocity_register: list,
        sign_register: list,
        boundary_ancillas: list,
    ):
        self._positions_register = position_register
        self._velocity_register = velocity_register
        self._sign_register = sign_register
        self._boundary_ancillas = boundary_ancillas

    ################################
    #   Initialization Functions   #
    ################################
    def set_initial_state(self, bubble_roi: RegionOfInterest, velocity: ArrayLike):
        """
        Stores the initial configuration.
        Does NOT apply gates yet.
        """
        self._initial_bubble = bubble_roi
        self._initial_velocity = velocity

    #############################
    #   Measurement Functions   #
    #############################
    def _measure_density(self, shots=None, backend=None) -> Dict:
        """
        Measure the density of the grid.

        Args:
            shots: Number of shots to measure.
            backend: Backend to use for measurement.

        Returns:
            Measurement results.
        """
        measurement = multi_measurement(
            self._positions_register, shots=shots, backend=backend
        )
        return measurement

    def full_measurement(self, shots=None, backend=None) -> Dict:
        """
        Measure the full state of the grid.

        Args:
            shots: Number of shots to measure.
            backend: Backend to use for measurement.

        Returns:
            Measurement results.
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
                self.encoded_to_user_coord(state_tuple[d], d)
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
    def boundary_ancillas(self):
        return self._boundary_ancillas

    @property
    def boundary_ancila_indexes(self):
        return self._boundary_ancilla_indexes

    @property
    def total_size(self):
        return self._total_size

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @property
    def boundary_offsets(self):
        return self._boundary_offsets

    @property
    def initial_bubble(self):
        return self._initial_bubble

    @property
    def initial_velocity(self):
        return self._initial_velocity

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

    def copy(self):
        """
        Returns a copy of the current state.
        """
        lbm_state = LBMState(
            self._grid_size, self._velocity_set, self._boundary_conditions
        )
        if self._initial_bubble is not None:
            lbm_state.set_initial_state(self._initial_bubble, self._initial_velocity)
        return lbm_state

    def get_clean_registers(self):
        """
        Returns freshly initialized registers tuple of the
        position, sign, velocity, and boundary ancilla registers.
        """
        vel = []
        sign = []
        anc = []
        anc = []
        for _ in self._boundary_ancilla_indexes.keys():
            anc.append(QuantumBool())

        pos = [QuantumModulus(int(size)) for size in self._positions_register_shape]
        # sign register - 0/False-negative direction, 1/True-positive direction
        sign = [QuantumBool() for _ in range(self._dimensions)]
        # Initialize the velocity register
        vel = [
            QuantumFloat(self._velocity_qubits_per_dim, signed=False)
            for _ in range(self._dimensions)
        ]
        return (
            pos,
            sign,
            vel,
            anc,
        )

    ###################
    #   Coordinates   #
    ###################
    def user_to_encoded_coord(self, user_coord, dimension):
        """Converts user 0-indexed coord to register index.

        Args:
            user_coord: The user 0-indexed coordinate.
            dimension: The dimension of the coordinate.

        Returns:
            Coordinates transformed to register index.
        """
        return user_coord + self._boundary_conditions[dimension].offset

    def encoded_to_user_coord(self, encoded_coord, dimension):
        """Converts register index back to user 0-indexed coord.

        Args:
            encoded_coord: The register index.
            dimension: The dimension of the coordinate.

        Returns:
            Coordinates transformed to user 0-indexed coordinate.
        """
        return encoded_coord - self._boundary_conditions[dimension].offset

    def get_encoded_bounds(self, user_bounds):
        """Converts list of user (min, max) tuples to encoded indices.

        Args:
            user_bounds: List of user (min, max) tuples.

        Returns:
            List of encoded (min, max) tuples.
        """
        return [
            (self.user_to_encoded_coord(low, d), self.user_to_encoded_coord(high, d))
            for d, (low, high) in enumerate(user_bounds)
        ]

    def get_encoded_point(self, user_point):
        """Converts user [x, y, ...] list to encoded indices.

        Args:
            user_point: List of user coordinates.

        Returns:
            List of encoded coordinates.
        """
        return [
            self.user_to_encoded_coord(coord, d) for d, coord in enumerate(user_point)
        ]

    def get_encoded_points(self, user_points):
        """Converts list of user [x, y, ...] tuples to encoded indices.

        Args:
            user_points: List of user [x, y, ...] tuples.

        Returns:
            List of encoded [x, y, ...] tuples.
        """
        return [self.get_encoded_point(point) for point in user_points]

    def encode_roi(self, user_roi: RegionOfInterest) -> RegionOfInterest:
        """Converts user-space ROI to register-space (encoded) ROI.

        Args:
            user_roi: User-space ROI.

        Returns:
            Register-space (encoded) ROI.
        """
        if user_roi.is_subgrid:
            encoded_bounds = self.get_encoded_bounds(user_bounds=user_roi.bounds)
            return RegionOfInterest(bounds=encoded_bounds)

        else:
            encoded_points = self.get_encoded_points(user_points=user_roi.points_list)
            return RegionOfInterest(points_list=encoded_points)
