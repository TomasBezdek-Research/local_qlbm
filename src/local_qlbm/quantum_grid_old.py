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
from src.types import BoundaryCondition, RegionOfInterest, VelocitySet


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

    ########################
    #   Complex Workflow   #
    ########################
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
        # 1. Encode ROI for internal ghost-layer logic
        encoded_roi = self._encode_roi(roi)

        # 2. Get the high-precision "Mass" of the region
        print("Phase 1: Estimating total density (f)...")
        f = self.estimate_region_density(
            roi=encoded_roi,
            simulation_iterations=simulation_iterations,
            eps=eps,
            shots=shots,
            backend=backend,
        )

        if f <= 0:
            print("Density is zero; skipping amplification.")
            return {"total_density": 0, "cells": {}}

        # 3. Amplify so the shots actually land in the ROI
        print(f"Phase 2: Amplifying region (f={f:.4f})...")
        self.boost_region(
            region_density=f,
            roi=encoded_roi,
            simulation_iterations=simulation_iterations,
        )

        # 4. Measure physical density
        # Note: This dict only contains cells that were actually measured (shots > 0)
        counts = self.get_physical_density(shots=shots, backend=backend)

        # 5. Filter counts to only include the ROI
        # (in case some probability leaked or noise occurred)
        roi_counts = {}
        total_measured_prob = 0

        for coords, prob in counts.items():
            if self._is_in_roi(coords, roi):
                roi_counts[coords] = prob
                total_measured_prob += prob

        # 6. Rescale: (Measured Cell Prob / Total Measured Prob) * f
        # This makes the sum of roi_rescaled exactly equal to f.
        roi_rescaled = {}
        if total_measured_prob > 0:
            scale_factor = f / total_measured_prob
            for coords, prob in roi_counts.items():
                roi_rescaled[coords] = prob * scale_factor

        return {"total_density": f, "cells": roi_rescaled}

    ################################
    #   Initialization Functions   #
    ################################

    def set_initial_state(self, bubble_roi: RegionOfInterest, velocity: List[int]):
        """
        Stores the initial configuration.
        Does NOT apply gates yet.
        """
        self._initial_bubble = bubble_roi
        self._initial_velocity = velocity

    def apply_initialization(self):
        """
        Applies the gates to set the registers to t=0.
        This is called internally by IQAE/AA.
        """
        # 1. Prepare Positions
        encoded_bounds = self._get_encoded_bounds(self._initial_bubble.bounds)
        for d in range(self._dimensions):
            num_states = 2 ** self._positions_register[d].size
            statevector = np.zeros(num_states)
            start, end = encoded_bounds[d]
            statevector[start : end + 1] = 1.0
            statevector /= np.linalg.norm(statevector)
            prepare(self._positions_register[d], statevector)

        # 2. Prepare Velocity
        self._set_global_velocity(self._initial_velocity)

    def _set_global_velocity(self, velocity_vector):
        """
        Sets the velocity u for the transport equation.
        velocity_vector: list of floats/ints like [1, -1]

        Args:
            velocity_vector: The velocity vector.
        """
        for d in range(self._dimensions):
            u_val = velocity_vector[d]

            # Encode magnitude
            self._velocity_register[d].encode(abs(u_val))

            # Encode sign: 1 for positive, 0 for negative
            # (Note: In your logic, True/1 is positive, False/0 is negative)
            if u_val >= 0:
                self._sign_register[d].encode(1)
            else:
                self._sign_register[d].encode(0)

    ###############################
    #   Simulation Functions    #
    ###############################
    def run_simulation(self, iterations=1, init=True):
        if init:
            self.apply_initialization()
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
            # Create a clear condition for the boundary
            at_lower = self._positions_register[d] == 0
            at_upper = self._positions_register[d] == self._grid_size[d] + 1
            is_at_boundary = at_lower | at_upper
            with is_at_boundary:
                # Flip the sign register if the position is out of bounds
                self._sign_register[d] = ~self._sign_register[d]
                # Reflect the position (streaming correction step)
                with self._sign_register[d]:
                    self._positions_register[d] += self._velocity_register[d]
                with ~self._sign_register[d]:
                    self._positions_register[d] -= self._velocity_register[d]

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

    def estimate_region_density(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        eps: float = 0.01,
        alpha: float = 0.05,
        shots=None,
        backend=None,
    ):
        """
        Uses IQAE to estimate the probability mass (density) within
        the region defined by 'roi'.

        Args:
            roi: RegionOfInterest object.
            simulation_iterations: Number of iterations to run the simulation.
            eps: The desired accuracy of the estimate.
            alpha: The desired confidence level of the estimate.
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

            # Step A: Run your simulation logic (Initialization + Streaming + BC)
            self.run_simulation(iterations=simulation_iterations)

            # Step B: Apply the oracle logic
            # We compute the condition and XOR it onto the target qubit.

            region_oracle = self._get_region_oracle(roi=roi)
            target ^= region_oracle

        # 4. Execute IQAE
        # This returns a float representing the density in the subgrid
        mes_kwargs = {"backend": backend}
        if shots is not None:
            mes_kwargs["shots"] = shots
        return IQAE(
            iqae_vars,
            iqae_wrapper,
            eps=eps,
            alpha=alpha,
            mes_kwargs=mes_kwargs,
        )

    def boost_region(
        self, region_density, roi: RegionOfInterest, simulation_iterations: int = 1
    ):
        """
        Amplifies the density in a specific region of the grid.

        Args:
            region_density: The initial probability of the region.
            roi: RegionOfInterest object.
            simulation_iterations: Number of iterations to run the simulation.
        """
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
            roi=roi,
            amplification_iterations=m,
            simulation_iterations=simulation_iterations,
        )

    def amplify_region(
        self,
        roi: RegionOfInterest,
        amplification_iterations: int = 1,
        simulation_iterations: int = 1,
    ):
        """
        Amplifies the density in a specific region of the grid.

        Args:
            roi: RegionOfInterest object.
            amplification_iterations: Number of iterations to amplify.
            simulation_iterations: Number of iterations to run the simulation.
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
            condition = self._get_region_oracle(roi=roi)

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
    def _get_region_oracle(self, roi: RegionOfInterest) -> QuantumBool:
        """
        Returns the oracle for the specified region.

        Args:
            roi: RegionOfInterest object.

        Returns:
            A QuantumBool representing the region.
        """
        if roi.is_subgrid:
            return self._get_subgrid_oracle(roi.bounds)
        return self._get_multi_point_oracle(roi.points_list)

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

    def _is_in_roi(self, coords: tuple, roi: RegionOfInterest) -> bool:
        """Helper to check if user-space coords are inside the user-space ROI.

        Args:
            coords: A tuple of user-space coordinates.
            roi: A RegionOfInterest object.
        """
        if roi.is_subgrid:
            return all(low <= c <= high for c, (low, high) in zip(coords, roi.bounds))
        else:
            return list(coords) in roi.points_list

    ################
    #   Display    #
    ################
    def __repr__(self):
        """Developer-friendly representation."""
        # We use !r to get the repr of the velocity_set (e.g., VelocitySet.D2Q9)
        return (
            f"LBMGrid(grid_size={self._grid_size.tolist()}, "
            f"velocity_set={self._velocity_set!r}, "
            f"dimensions={self._dimensions})"
        )

    def __str__(self):
        """User-friendly summary."""
        # This provides a clean 'dashboard' view of the grid
        return (
            f"--- Quantum LBM Grid: {self._name or 'Unnamed'} ---\n"
            f"Dimensions : {self._dimensions}D\n"
            f"Grid Size  : {' x '.join(map(str, self._grid_size))}\n"
            f"Velocity   : {self._velocity_set.name}\n"
            f"Total Size : {self._total_size} qubits\n"
            f"--------------------------------------"
        )
