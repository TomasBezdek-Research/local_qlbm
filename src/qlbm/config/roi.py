from dataclasses import dataclass


@dataclass(frozen=True)
class RegionOfInterest:
    _bounds: list[tuple[int, int]] | None = None
    _points_set: set[tuple[int, ...]] | None = None

    def __post_init__(self):
        """Validation logic runs after initialization."""
        if (self._bounds is not None) == (self._points_set is not None):
            raise ValueError(
                "Exactly one of 'bounds' or 'points_list' must be specified."
            )

    @property
    def bounds(self) -> list[tuple[int, int]]:
        if self._bounds is None:
            raise AttributeError("This ROI is a point cloud, not a subgrid.")
        return self._bounds

    @property
    def points_set(self) -> set[tuple[int, ...]]:
        if self._points_set is None:
            raise AttributeError("This ROI is a subgrid, not a point cloud.")
        return self._points_set

    @classmethod
    def from_bounds(cls, bounds: list[tuple[int, int]]):
        """
        Creates a RegionOfInterest from a list of tuples representing the bounds.

        Args:
            bounds (list): A list of tuples representing the bounds. Example: [(2, 5), (1, 3)].
            Lower bound is inclusive, Upper bound is exclusive.

        Returns:
            RegionOfInterest: A RegionOfInterest object.
        """
        return cls(_bounds=bounds)

    @classmethod
    def from_points(cls, points: set[tuple[int, ...]]):
        """
        Creates a RegionOfInterest from a list of lists representing the points.

        Args:
            points (list): A list of lists representing the points. Example: [[2, 3], [5, 7]]

        Returns:
            RegionOfInterest: A RegionOfInterest object.
        """
        return cls(_points_set=points)

    @property
    def is_subgrid(self) -> bool:
        return self._bounds is not None

    @property
    def is_point_cloud(self) -> bool:
        return self._points_set is not None

    def contains(self, coords: tuple[int, ...]) -> bool:
        """
        Checks if the given user-space coordinates are inside this ROI.

        Args:
            coords (tuple): The user-space coordinates to check.

        Returns:
            bool: True if the coordinates are inside the ROI, False otherwise.
        """
        if self.is_subgrid:
            # Check if coords are within the n-dimensional bounding box
            return all(low <= c < high for c, (low, high) in zip(coords, self.bounds))
        else:
            # Check if the coordinate exists in the discrete point list
            # We convert to list if points_list is a list of lists
            return coords in self.points_set
