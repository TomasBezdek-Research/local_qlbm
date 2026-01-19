import enum

import numpy as np
from numpy.typing import ArrayLike
from qrisp import QuantumFloat, QuantumModulus, multi_measurement
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_carry_path import (
    QuantumBool,
)
from sympy.printing.pretty.pretty_symbology import B
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

    def stream(self):
        """
        Do one streaming step.
        """
        for d in range(self._dimensions):
            with self._sign_register[d]:
                self._positions_register[d] += self._velocity_register[d]

            with ~self._sign_register[d]:
                self._positions_register[d] -= self._velocity_register[d]

    def boundary_check(self):
        """
        Check if the positions are within the grid boundaries.
        Flip the sign register if the position is out of bounds.
        """
        for d in range(self._dimensions):
            if self._boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue
            with (self._positions_register[d]) == 0 | (
                self._positions_register[d] == self._grid_size[d] + 1
            ):
                self._sign_register[d] = ~self._sign_register[d]

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
