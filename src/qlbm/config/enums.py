import enum


class VelocitySet(enum.Enum):
    """
    Discrete velocity sets supported by the LBM implementation.

    Each member stores the spatial dimensionality, number of discrete
    velocities, and the unique velocity coordinates used along each dimension.
    """

    D1Q3 = (1, 3, [-1, 0, 1])
    D2Q5 = (2, 5, [-1, 0, 1])
    D2Q9 = (2, 9, [-1, 0, 1])
    D3Q15 = (3, 15, [-1, 0, 1])
    D3Q19 = (3, 19, [-1, 0, 1])
    D3Q27 = (3, 27, [-1, 0, 1])

    D2Q21 = (2, 21, [-2, -1, 0, 1, 2])

    @property
    def dimensions(self) -> int:
        """Return the spatial dimensionality of the velocity set."""
        return self.value[0]

    @property
    def velocity_set_size(self) -> int:
        """Return the number of discrete velocities in the velocity set."""
        return self.value[1]

    @property
    def unique_coordinates(self) -> list[int]:
        """Return the unique velocity coordinates used along each dimension."""
        return self.value[2]

    @property
    def offset(self) -> int:
        """Return the the maximal amplitude of the velocity set."""
        return self.value[2][-1]

    @property
    def additional_layers(self) -> int:
        """Return the total number of additional grid layers required by the velocity offset."""
        return 2 * self.value[2][-1]


class BoundaryCondition(enum.Enum):
    """Boundary conditions supported by the LBM implementation."""

    PERIODIC = 0
    BOUNCE_BACK = 1
    SIMPLE_SLIP = 2

    @property
    def requires_offset(self) -> bool:
        """Return whether the boundary condition requires a grid offset."""
        return self.value > 0


class BCCombination(enum.Enum):
    """Combinations of boundary-condition types present in a simulation."""

    PERIODIC = enum.auto()
    BOUNCE_BACK = enum.auto()
    SIMPLE_SLIP = enum.auto()
    BOUNCE_BACK_AND_SIMPLE_SLIP = enum.auto()
