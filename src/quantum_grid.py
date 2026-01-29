import enum

import numpy as np
from mpmath.functions.functions import re
from numpy.typing import ArrayLike
from qrisp import (
    IQAE,
    QuantumFloat,
    QuantumModulus,
    amplitude_amplification,
    auto_uncompute,
    multi_measurement,
    z,
)
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_carry_path import (
    QuantumBool,
)
from typing_extensions import Dict


class VelocitySet(enum.Enum):
    D1Q3 = (1, 3, [-1, 0, 1])
    D2Q5 = (2, 5, [-1, 0, 1])
    D2Q9 = (2, 9, [-1, 0, 1])
    D3Q15 = (3, 15, [-1, 0, 1])
    D3Q19 = (3, 19, [-1, 0, 1])
    D3Q27 = (3, 27, [-1, 0, 1])

    D2Q21 = (2, 21, [-2, -1, 0, 1, 2])

    @property
    def dimensions(self) -> int:
        return self.value[0]

    @property
    def velocity_set_size(self) -> int:
        return self.value[1]

    @property
    def unique_coordinates(self) -> list:
        return self.value[2]


class BoundaryCondition(enum.Enum):
    PERIODIC = 0
    BOUNCE_BACK = 1

    @property
    def aditional_layers(self) -> int:
        return 2 * self.value


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
        self._positions_register_layers = self._grid_size + self._boundary_layers
        self._positions_register = [
            QuantumModulus(size) for size in self._positions_register_layers
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

    def run_simulation(self, iterations=1):
        for _ in range(iterations):
            self.stream()
            self.apply_boundary_conditions()

    def stream(self):
        """
        Do one streaming step.
        """
        for d in range(self._dimensions):
            with self._sign_register[d]:
                self._positions_register[d] += self._velocity_register[d]

            with ~self._sign_register[d]:
                self._positions_register[d] -= self._velocity_register[d]

    @auto_uncompute
    def apply_boundary_conditions(self):
        """
        Check if the positions are within the grid boundaries.
        Flip the sign register if the position is out of bounds.
        """
        for d in range(self._dimensions):
            if self._boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue
            with (self._positions_register[d] == 0) | (
                self._positions_register[d] == self._grid_size[d] + 1
            ):
                self._sign_register[d] = ~self._sign_register[d]
                with self._sign_register[d]:
                    self._positions_register[d] += self._velocity_register[d]
                with ~self._sign_register[d]:
                    self._positions_register[d] -= self._velocity_register[d]

    @auto_uncompute
    def boundary_check(self):
        """
        Check if the positions are within the grid boundaries.
        Flip the sign register if the position is out of bounds.
        """
        for d in range(self._dimensions):
            if self._boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue
            with (self._positions_register[d] == 0) | (
                self._positions_register[d] == self._grid_size[d] + 1
            ):
                self._sign_register[d] = ~self._sign_register[d]

    #############################
    #   Measurement Functions   #
    #############################
    def measure_density(self, shots=None, backend=None) -> Dict:
        """
        Measure the density of the grid.
        """
        # TODO: Implement coordinates recalculation based on boundary conditions
        measurement = multi_measurement(
            self._positions_register, shots=shots, backend=backend
        )
        return measurement

    def full_measurement(self, shots=None, backend=None) -> Dict:
        """
        Measure the full state of the grid.
        """
        # TODO: Implement coordinates recalculation based on boundary conditions
        measurement = multi_measurement(
            self._positions_register + self._sign_register + self._velocity_register,
            shots=shots,
            backend=backend,
        )
        return measurement

    def estimate_region_density(
        self,
        bounds=None,
        points_list=None,
        simulation_iterations=1,
        eps=0.01,
        alpha=0.05,
    ):
        """
        Uses IQAE to estimate the probability mass (density) within
        the subgrid defined by 'bounds'.
        """
        # 1. Create a fresh indicator qubit (the 'good state' marker)
        target_qubit = QuantumBool()

        # 2. Gather all registers involved.
        # IQAE will treat the LAST element as the target indicator.
        iqae_vars = self._positions_register + [target_qubit]

        # 3. Define the IQAE state preparation wrapper
        def iqae_wrapper(*args):
            # Extract the target bit (the last argument)
            target = args[-1]

            # Step A: Run your simulation logic (Streaming + BC)
            self.run_simulation(iterations=simulation_iterations)

            # Step B: Apply the oracle logic
            # We compute the condition and XOR it onto the target qubit.

            region_oracle = self._get_region_oracle(
                bounds=bounds, points_list=points_list
            )
            target ^= region_oracle

        # 4. Execute IQAE
        # This returns a float representing the density in the subgrid
        return IQAE(iqae_vars, iqae_wrapper, eps=eps, alpha=alpha)

    def boost_region(
        self, region_density, bounds=None, points_list=None, simulation_iterations=1
    ):
        if region_density <= 0:
            print("Density is 0, nothing to amplify.")
            return

        # 2. Calculate optimal iterations m
        m = self.get_optimal_m(region_density)

        # 3. Handle the 'No Boost Needed' case
        if m == 0:
            if region_density >= 0.5:
                print(
                    f"Density is already high ({region_density:.4f}). Skipping amplification to avoid over-rotation."
                )
            else:
                print(
                    f"Density ({region_density:.4f}) is too low to amplify effectively with integer steps."
                )
            return

        print(f"Current density: {region_density:.4f}. Amplifying with {m} iterations.")

        # 4. Perform the amplification
        # Note: ensure simulation_iterations matches the state that created region_density!
        self.amplify_region(
            bounds=bounds,
            points_list=points_list,
            amplification_iterations=m,
            simulation_iterations=simulation_iterations,
        )

    def amplify_region(
        self,
        bounds=None,
        points_list=None,
        amplification_iterations=1,
        simulation_iterations=1,
    ):
        """
        Amplifies the density in a specific region of the grid.
        """
        # args tells AA which registers are being modified
        args = self._positions_register

        # A operator: Prepares the state (Physics)
        def state_function(*pos_regs):
            self.run_simulation(iterations=simulation_iterations)

        # O operator: Tags the state (Logic)
        def oracle_function(*pos_regs):
            # We re-evaluate the oracle based on the NEW positions
            # created by state_function
            condition = self._get_region_oracle(bounds=bounds, points_list=points_list)

            # Apply the phase flip required for AA
            z(condition)
            # Because of @auto_uncompute on the oracle methods,
            # 'condition' is cleaned up perfectly here.

        # Perform the amplification
        amplitude_amplification(
            args=args,
            state_function=state_function,
            oracle_function=oracle_function,
            iter=amplification_iterations,
        )

    @auto_uncompute
    def _get_region_oracle(self, bounds=None, points_list=None) -> QuantumBool:
        """
        Returns the oracle for the specified region.

        Args:
            bounds: List of tuples [(min_d0, max_d0), (min_d1, max_d1), ...]
            Example for 2D: [(2, 5), (1, 3)]
            points_list: List of coordinate lists [[x1, y1,...], [x2, y2,...], ...]
            Example for 2D: [[2, 3], [5, 7]]

        Returns:
            A QuantumBool representing the region.
        """
        if bounds is not None and points_list is not None:
            raise ValueError(
                "Either 'bounds' or 'points_list' must be specified, but not both."
            )
        if bounds is None and points_list is None:
            raise ValueError("Either 'bounds' or 'points_list' must be specified.")

        if bounds is not None:
            return self._get_subgrid_oracle(bounds)
        else:
            return self._get_multi_point_oracle(points_list)

    @auto_uncompute
    def _get_subgrid_oracle(self, bounds) -> QuantumBool:
        """
        Returns the oracle for the specified subgrid.

        Args:
            bounds: List of tuples [(min_d0, max_d0), (min_d1, max_d1), ...]
            Example for 2D: [(2, 5), (1, 3)]

        Returns:
            A QuantumBool representing the subgrid.
        """
        # Start with a condition for the first dimension
        res = (self._positions_register[0] >= bounds[0][0]) & (
            self._positions_register[0] <= bounds[0][1]
        )

        # Chain the other dimensions
        for d in range(1, self._dimensions):
            res &= self._positions_register[d] >= bounds[d][0]
            res &= self._positions_register[d] <= bounds[d][1]

        return res

    @auto_uncompute
    def _get_multi_point_oracle(self, points_list) -> QuantumBool:
        """
        Returns the oracle for the specified region.

        Args:
            points_list: List of coordinate lists [[x1, y1,...], [x2, y2,...], ...]
            Example for 2D: [[2, 3], [5, 7]]

        Returns:
            A QuantumBool representing the region.
        """

        def check_single_point(coords):
            res = self._positions_register[0] == coords[0]
            for d in range(1, self._dimensions):
                res &= self._positions_register[d] == coords[d]
            return res

        # Combine all points with OR logic
        final_oracle = check_single_point(points_list[0])
        for point in points_list[1:]:
            final_oracle |= check_single_point(point)

        return final_oracle

    def __repr__(self):
        return f"LBMGrid({self._grid_size}, {self._velocity_set})"

    def __str__(self):
        return f"LBMGrid({self._grid_size}, {self._velocity_set})"

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

    ##################
    #   Utilities    #
    ##################
    @staticmethod
    def get_optimal_m(a):
        """
        Calculates the optimal number of iterations for amplification.

        Args:
            a: The initial probability of the region.

        Returns:
            The optimal number of iterations.
        """
        # If the initial probability is already low,
        if a >= 1.0:
            return 0
        if a <= 0.0:
            return 0

        # If the initial probability is already high,
        if a >= 0.5:
            return 0

        theta = np.arcsin(np.sqrt(a))
        # Using floor ensures we don't rotate PAST the 100% mark
        m = int(np.floor(np.pi / (4 * theta)))

        return m

    @staticmethod
    def get_optimal_m_rounded(a):
        """
        Calculates the optimal number of iterations for amplification.

        Args:
            a: The initial probability of the region.

        Returns:
            The optimal number of iterations.
        """
        if a >= 0.5 or a <= 0.0:
            return 0
        theta = np.arcsin(np.sqrt(a))
        # This centers the rotation around the 90-degree (pi/2) target
        return max(0, int(np.round(np.pi / (4 * theta) - 0.5)))
