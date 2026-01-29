import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple


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

    @property
    def offset(self) -> int:
        return self.value


@dataclass(frozen=True)
class RegionOfInterest:
    bounds: Optional[List[Tuple[int, int]]] = None
    points_list: Optional[List[List[int]]] = None

    def __post_init__(self):
        """Validation logic runs after initialization."""
        if (self.bounds is not None) == (self.points_list is not None):
            raise ValueError(
                "Exactly one of 'bounds' or 'points_list' must be specified."
            )

    @classmethod
    def from_bounds(cls, bounds):
        return cls(bounds=bounds)

    @classmethod
    def from_points(cls, points):
        return cls(points_list=points)

    @property
    def is_subgrid(self) -> bool:
        return self.bounds is not None

    @property
    def is_point_cloud(self) -> bool:
        return self.points_list is not None
