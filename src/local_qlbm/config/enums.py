import enum


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

    @property
    def offset(self) -> int:
        return self.value[2][-1]

    @property
    def aditional_layers(self) -> int:
        return 2 * self.value[2][-1]


class BoundaryCondition(enum.Enum):
    PERIODIC = 0
    BOUNCE_BACK = 1
    SIMPLE_SLIP = 2

    @property
    def need_offset(self) -> bool:
        return self.value > 0


class BCCombination(enum.Enum):
    PERIODIC = enum.auto()
    BB = enum.auto()
    SLIP = enum.auto()
    BB_AND_SLIP = enum.auto()
