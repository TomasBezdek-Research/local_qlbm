import enum
from networkx.algorithms import boundary
from qiskit.circuit.library.n_local.evolved_operator_ansatz import QuantumRegister
from qrisp import QuantumFloat
import numpy as np
from numpy.typing import ArrayLike
from sympy.printing.pretty.pretty_symbology import B

class VelocitySet(enum.Enum):
    D1Q3 = (1, 3, [-1,0,1])
    D2Q5 = (2,5, [-1,0,1])
    D2Q9 = (2,9, [-1,0,1])
    D3Q15 = (3,15, [-1,0,1])
    D3Q19 = (3,19, [-1,0,1])
    D3Q27 = (3,27, [-1,0,1])

    D2Q21 = (2,21, [-2, -1,0,1,2])

    @property
    def dimensions(self)->int:
        return self.value[0]

    @property
    def velocity_set_size(self)->int:
        return self.value[1]

    @property
    def unique_coordinates(self)->list:
        return self.value[2]


class BoundaryCondition(enum.Enum):
    PERIODIC = 0
    BOUNCE_BACK = 1


class LBMGrid():
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
    def __init__(self, grid_size: ArrayLike, velocity_set: VelocitySet, boundary_conditions: list[BoundaryCondition], name= None):
        self._name = name
        self._grid_size = np.array(grid_size, dtype=np.int32)
        self._velocity_set = velocity_set
        self._dimensions = self._grid_size.size
        if not self._dimensions == velocity_set.dimensions:
            raise ValueError(f"Grid dimensions {self._dimensions} must match velocity set dimensions {velocity_set.dimensions}")

        if len(boundary_conditions) != self._dimensions:
            raise ValueError(f"Boundary conditions must be specified for each dimension. Got {len(boundary_conditions)} conditions for {self._dimensions} dimensions.")
        self._boundary_conditions = boundary_conditions

        # Initialize the grid
        # Positions
        self._position_register_sizes = np.array(np.ceil(np.log2(grid_size)), dtype=np.int32)
        self._positions_register = [QuantumFloat(size, signed=False) for size in self._position_register_sizes]

        # Velocities
        self._init_velocity_register()

        self._total_size = np.sum(self._position_register_sizes) + self._velocity_qubits_per_dim * self._dimensions

    def _init_velocity_register(self):
        num_points = len(self._velocity_set.unique_coordinates)
        self._velocity_qubits_per_dim = int(np.ceil(np.log2(num_points)))

        # Ensure direction bit logic
        if self._velocity_qubits_per_dim < 2:
            self._velocity_qubits_per_dim = 2

        # Initialize the velocity register
        self._velocity_register = [QuantumFloat(self._velocity_qubits_per_dim, signed=True) for _ in range(self._dimensions)]


    def stream(self):
        """
        Do one streaming step.
        """
        for d in range(self._dimensions):
            self._positions_register[d] += self._velocity_register[d]




    def __repr__(self):
        return f"LBMGrid({self._grid_size}, {self._velocity_set})"

    def __str__(self):
        return f"LBMGrid({self._grid_size}, {self._velocity_set})"

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
    def total_size(self):
        return self._total_size
