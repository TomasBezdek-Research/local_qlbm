from dataclasses import dataclass


@dataclass(frozen=True)
class RegionOfInterest:
    """
    Region of interest defined either by rectangular bounds or by discrete points.

    Exactly one representation must be provided.
    """

    _bounds: list[tuple[int, int]] | None = None
    _points_set: set[tuple[int, ...]] | None = None

    def __post_init__(self):
        """Validate that exactly one ROI representation is specified."""
        if (self._bounds is not None) == (self._points_set is not None):
            raise ValueError("Exactly one of bounds or points must be specified.")

    @property
    def bounds(self) -> list[tuple[int, int]]:
        """Return the bounds defining the rectangular subgrid."""
        if self._bounds is None:
            raise AttributeError("This ROI is a point cloud, not a subgrid.")
        return self._bounds

    @property
    def points_set(self) -> set[tuple[int, ...]]:
        """Return the discrete points defining the point-cloud ROI."""
        if self._points_set is None:
            raise AttributeError("This ROI is a subgrid, not a point cloud.")
        return self._points_set

    @classmethod
    def from_bounds(cls, bounds: list[tuple[int, int]]):
        """
        Create a region of interest from rectangular bounds.

        Each `(lower, upper)` pair defines the interval along one dimension,
        with the lower bound inclusive and the upper bound exclusive.

        Args:
            bounds: Bounds for each spatial dimension.

        Returns:
            Region of interest defined by the given bounds.
        """
        return cls(_bounds=bounds)

    @classmethod
    def from_points(cls, points: set[tuple[int, ...]]):
        """
        Create a region of interest from a set of discrete points.

        Args:
            points: Coordinates belonging to the region, for example
                `{(2, 3), (5, 7)}`.

        Returns:
            Region of interest defined by the given points.
        """
        return cls(_points_set=points)

    @property
    def is_subgrid(self) -> bool:
        """Return whether the region is defined by rectangular bounds."""
        return self._bounds is not None

    @property
    def is_point_cloud(self) -> bool:
        """Return whether the region is defined by discrete points."""
        return self._points_set is not None

    def contains(self, coords: tuple[int, ...]) -> bool:
        """
        Check whether the given coordinates belong to this region.

        Args:
            coords: Coordinates in user-space representation.

        Returns:
            True if the coordinates are inside the region, otherwise False.
        """
        if self.is_subgrid:
            # Check if coords are within the n-dimensional bounding box
            return all(low <= c < high for c, (low, high) in zip(coords, self.bounds))
        else:
            return coords in self.points_set
