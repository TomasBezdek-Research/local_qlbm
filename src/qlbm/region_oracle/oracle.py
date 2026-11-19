from __future__ import annotations

from qrisp import (
    QuantumBool,
    QuantumModulus,
    auto_uncompute,
    mcx,
    z,
)

from qlbm.config.model import LBMConfig
from qlbm.config.roi import RegionOfInterest


def get_phase_oracle(
    config: LBMConfig,
    roi: RegionOfInterest,
    position_register: list[QuantumModulus],
    oracle_ancillas: list[QuantumBool],
    target_qubit: QuantumBool,
) -> None:
    """
    Apply a phase flip to position states inside a region of interest.

    The region flag oracle is applied before and after a Z gate on the
    target qubit, converting the bit-flip oracle into a phase oracle.

    Args:
        config: Simulation configuration.
        roi: Encoded region of interest to mark.
        position_register: Position registers encoding the current coordinates.
        oracle_ancillas: Ancilla qubits used to evaluate region membership.
        target_qubit: Target qubit used by the oracle.
    """

    get_region_oracle(
        config=config,
        roi=roi,
        position_register=position_register,
        oracle_ancillas=oracle_ancillas,
        target_qubit=target_qubit,
    )

    z(target_qubit)

    get_region_oracle(
        config=config,
        roi=roi,
        position_register=position_register,
        oracle_ancillas=oracle_ancillas,
        target_qubit=target_qubit,
    )


def get_region_oracle(
    config: LBMConfig,
    roi: RegionOfInterest,
    position_register: list[QuantumModulus],
    oracle_ancillas: list[QuantumBool],
    target_qubit: QuantumBool,
) -> None:
    """
    Apply a flag oracle for the specified region of interest.

    The target qubit is flipped when the encoded position lies inside
    the region. The appropriate oracle implementation is selected based
    on whether the region is a subgrid or a point cloud.

    Args:
        config: Simulation configuration.
        roi: Encoded region of interest to mark.
        position_register: Position registers encoding the current coordinates.
        oracle_ancillas: Ancilla qubits used to evaluate region membership.
        target_qubit: Qubit flipped for positions inside the region.
    """
    if roi.is_subgrid:
        get_subgrid_oracle(
            config, roi.bounds, position_register, oracle_ancillas, target_qubit
        )
    elif roi.is_point_cloud:
        get_point_cloud_oracle(
            config, roi.points_set, position_register, oracle_ancillas, target_qubit
        )


def get_subgrid_oracle(
    config: LBMConfig,
    bounds: list[tuple[int, int]],
    position_register: list[QuantumModulus],
    oracle_ancillas: list[QuantumBool],
    target_qubit: QuantumBool,
) -> None:
    """
    Flag position states lying inside a rectangular subgrid.

    For each dimension, an ancilla marks whether the coordinate lies
    within the corresponding lower-inclusive, upper-exclusive interval.
    The target qubit is flipped when all dimensions satisfy their bounds.

    Args:
        config: Simulation configuration defining the number of dimensions.
        bounds: Encoded bounds for each dimension as `(lower, upper)` pairs.
        position_register: Position registers encoding the current coordinates.
        oracle_ancillas: Ancilla qubits used for the per-dimension checks.
        target_qubit: Qubit flipped when the position lies inside the subgrid.
    """

    for d in range(config.dimensions):
        with position_register[d] < bounds[d][0]:
            oracle_ancillas[d].flip()
        with position_register[d] >= bounds[d][1]:
            oracle_ancillas[d].flip()
        oracle_ancillas[d].flip()

    mcx(oracle_ancillas, target_qubit, method="gray")

    for d in range(config.dimensions):
        oracle_ancillas[d].flip()
        with position_register[d] < bounds[d][0]:
            oracle_ancillas[d].flip()
        with position_register[d] >= bounds[d][1]:
            oracle_ancillas[d].flip()


@auto_uncompute
def get_point_cloud_oracle(
    config: LBMConfig,
    points_set: set[tuple[int, ...]],
    position_register: list[QuantumModulus],
    oracle_ancillas: list[QuantumBool],
    target_qubit: QuantumBool,
) -> None:
    """
    Flag position states matching any point in a discrete point-cloud region.

    Each point is matched against all position dimensions. If all coordinates
    match a point, the target qubit is flipped.

    Args:
        config: Simulation configuration defining the number of dimensions.
        points_set: Set of encoded coordinates belonging to the region.
        position_register: Position registers encoding the current coordinates.
        oracle_ancillas: Ancilla qubits used for coordinate comparisons.
        target_qubit: Qubit flipped when the position matches a region point.
    """

    def check_single_point(coords: tuple[int, ...]):
        for d in range(config.dimensions):
            with position_register[d] == coords[d]:
                oracle_ancillas[d].flip()

        mcx(oracle_ancillas, target_qubit, method="gray")

        for d in range(config.dimensions):
            with position_register[d] == coords[d]:
                oracle_ancillas[d].flip()

    for point in points_set:
        check_single_point(point)
