from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .enums import BCCombination, BoundaryCondition, VelocitySet
from .roi import RegionOfInterest


@dataclass
class LBMConfig:
    """
    Classical configuration of a quantum LBM simulation.

    Stores the static simulation parameters, initial conditions, and derived
    quantities required to construct the quantum registers and kernels.
    No quantum registers are stored in this class.
    """

    # ======== INPUT PARAMETERS ========

    _grid_size_input: np.ndarray | list[int]
    grid_size: np.ndarray = field(init=False)
    velocity_set: VelocitySet
    boundary_conditions: list[BoundaryCondition] = field(init=False)
    _boundary_conditions_input: list[BoundaryCondition] | BoundaryCondition

    name: str | None = None

    # Initial conditions (set later)
    initial_bubble: RegionOfInterest | None = None
    initial_velocity: np.ndarray | None = None

    # ======== DERIVED PARAMETERS ========

    dimensions: int = field(init=False)
    velocity_qubits_per_dim: int = field(init=False)
    boundary_offsets: tuple[int, ...] = field(init=False)
    boundary_layers: tuple[int, ...] = field(init=False)

    positions_register_shape: tuple[int, ...] = field(init=False)

    bc_combination: BCCombination = field(init=False)

    boundary_ancilla_indexes: dict[int, int] = field(init=False)
    bb_boundary_ancilla_indexes: dict[int, int] = field(init=False)

    total_size: int = field(init=False)

    # ======== INITIALIZATION ========

    def __post_init__(self) -> None:
        """Validate the input configuration and initialize all derived parameters."""
        self.grid_size = np.asarray(self._grid_size_input, dtype=int)
        self.dimensions = self.grid_size.size

        if self.dimensions != self.velocity_set.dimensions:
            raise ValueError(
                f"Grid dimensions {self.dimensions} must match "
                f"velocity set dimensions {self.velocity_set.dimensions}"
            )

        if not isinstance(self._boundary_conditions_input, list):
            bc_list = [self._boundary_conditions_input] * self.dimensions
        else:
            bc_list = self._boundary_conditions_input.copy()

        if len(bc_list) != self.dimensions:
            raise ValueError(
                f"Boundary conditions must match grid dimensions "
                f"({self.dimensions}), got {len(bc_list)}"
            )

        self.boundary_conditions = bc_list

        self._init_velocity_properties()
        self._init_boundary_properties()
        self._init_bc_combination()
        self._init_register_shapes()
        self._init_boundary_ancillas()
        self._compute_total_size()

    # ======== VELOCITY ========

    def _init_velocity_properties(self) -> None:
        """Determine the number of velocity-magnitude qubits required per dimension."""
        unique_coords = self.velocity_set.unique_coordinates

        num_points = int(np.ceil(len(unique_coords) / 2))
        qubits = int(np.ceil(np.log2(num_points)))

        self.velocity_qubits_per_dim = max(qubits, 1)

    # ======== BOUNDARIES ========

    def _init_boundary_properties(self) -> None:
        """Compute the coordinate offsets and additional layers for each dimension."""
        offsets: list[int] = []
        layers: list[int] = []

        for bc in self.boundary_conditions:
            if bc.requires_offset:
                offsets.append(self.velocity_set.offset)
                layers.append(self.velocity_set.additional_layers)
            else:
                offsets.append(0)
                layers.append(0)

        self.boundary_offsets = tuple(offsets)
        self.boundary_layers = tuple(layers)

    def _init_bc_combination(self) -> None:
        """Determine which combination of non-periodic boundary conditions is present."""

        active_bcs = {
            bc for bc in self.boundary_conditions if bc != BoundaryCondition.PERIODIC
        }

        has_bb = BoundaryCondition.BOUNCE_BACK in active_bcs
        has_slip = BoundaryCondition.SIMPLE_SLIP in active_bcs

        if has_bb and has_slip:
            self.bc_combination = BCCombination.BOUNCE_BACK_AND_SIMPLE_SLIP
        elif has_bb:
            self.bc_combination = BCCombination.BOUNCE_BACK
        elif has_slip:
            self.bc_combination = BCCombination.SIMPLE_SLIP
        else:
            self.bc_combination = BCCombination.PERIODIC

    # ======== REGISTER SHAPES ========

    def _init_register_shapes(self) -> None:
        """Compute the position-register shape including additional boundary layers."""

        self.positions_register_shape = tuple(
            (self.grid_size + np.array(self.boundary_layers)).tolist()
        )

    # ======== ANCILLA INDEXING ========

    def _init_boundary_ancillas(self) -> None:
        """
        Assign boundary ancilla indices to dimensions requiring boundary handling.

        `boundary_ancilla_indexes` contains all non-periodic boundaries, while
        `bb_boundary_ancilla_indexes` contains only bounce-back boundaries.
        """
        self.boundary_ancilla_indexes = {}
        self.bb_boundary_ancilla_indexes = {}

        idx = 0

        for d, bc in enumerate(self.boundary_conditions):
            if bc == BoundaryCondition.BOUNCE_BACK:
                self.bb_boundary_ancilla_indexes[d] = idx

            if bc in (BoundaryCondition.BOUNCE_BACK, BoundaryCondition.SIMPLE_SLIP):
                self.boundary_ancilla_indexes[d] = idx
                idx += 1

    # ======== SIZE ========

    def _compute_total_size(self) -> None:
        """Compute the total number of position, velocity, and sign qubits."""
        position_qubits = sum(
            int(np.ceil(np.log2(size))) for size in self.positions_register_shape
        )

        velocity_qubits = (self.velocity_qubits_per_dim + 1) * self.dimensions

        self.total_size = position_qubits + velocity_qubits

    # ======== INITIAL CONDITIONS ========

    def set_initial_state(
        self,
        bubble: RegionOfInterest,
        velocity: np.ndarray | list[int],
    ) -> None:
        """
        Set the initial region and velocity of the simulation.

        Args:
            bubble: Region defining the initial bubble.
            velocity: Initial velocity vector.
        """
        self.initial_bubble = bubble
        self.initial_velocity = np.array(velocity, dtype=int)

    # ======== ENCODING ========

    def user_to_encoded_coord(self, coord: int, dim: int) -> int:
        """
        Convert a physical-grid coordinate to its internally encoded coordinate.

        Args:
            coord: Coordinate in the physical grid.
            dim: Spatial dimension of the coordinate.

        Returns:
            Corresponding encoded coordinate.
        """
        return coord + self.boundary_offsets[dim]

    def encoded_to_user_coord(self, coord: int, dim: int) -> int:
        """
        Convert an internally encoded coordinate to its physical-grid coordinate.

        Args:
            coord: Encoded coordinate.
            dim: Spatial dimension of the coordinate.

        Returns:
            Corresponding physical-grid coordinate.
        """
        return coord - self.boundary_offsets[dim]

    def get_encoded_bounds(
        self,
        bounds: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        Convert physical-grid bounds to encoded bounds.

        Args:
            bounds: Bounds for each spatial dimension.

        Returns:
            Bounds shifted according to the configured boundary offsets.
        """
        return [
            (
                self.user_to_encoded_coord(low, d),
                self.user_to_encoded_coord(high, d),
            )
            for d, (low, high) in enumerate(bounds)
        ]

    def get_encoded_point(self, point: tuple[int, ...]) -> tuple[int, ...]:
        """
        Convert a physical-grid point to its encoded coordinates.

        Args:
            point: Point in physical-grid coordinates.

        Returns:
            Point shifted according to the configured boundary offsets.
        """
        return tuple(
            self.user_to_encoded_coord(coord, d) for d, coord in enumerate(point)
        )

    def get_encoded_points(
        self,
        points: set[tuple[int, ...]],
    ) -> set[tuple[int, ...]]:
        """
        Convert a set of physical-grid points to encoded coordinates.

        Args:
            points: Points in physical-grid coordinates.

        Returns:
            Points shifted according to the configured boundary offsets.
        """
        return {self.get_encoded_point(p) for p in points}

    def encode_roi(self, roi: RegionOfInterest) -> RegionOfInterest:
        """
        Convert a region of interest from physical-grid to encoded coordinates.

        For a subgrid ROI, the bounds are encoded. For a point-cloud ROI,
        each individual point is encoded.

        Args:
            roi: Region of interest in physical-grid coordinates.

        Returns:
            Equivalent region of interest in encoded coordinates.
        """
        if roi.is_subgrid:
            return RegionOfInterest.from_bounds(self.get_encoded_bounds(roi.bounds))
        else:
            return RegionOfInterest.from_points(self.get_encoded_points(roi.points_set))

    # ======== DISPLAY ========

    def __repr__(self) -> str:
        """Return a concise developer-oriented representation of the configuration."""
        return (
            f"LBMConfig(grid_size={self.grid_size.tolist()}, "
            f"velocity_set={self.velocity_set}, "
            f"dimensions={self.dimensions})"
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the simulation configuration."""
        return (
            f"--- LBM Config: {self.name or 'Unnamed'} ---\n"
            f"Dimensions : {self.dimensions}D\n"
            f"Grid Size  : {' x '.join(map(str, self.grid_size))}\n"
            f"Velocity   : {self.velocity_set.name}\n"
            f"Total size : {self.total_size} qubits\n"
            f"--------------------------------------"
        )
