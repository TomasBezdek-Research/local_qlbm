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

# ======================================================
# ORACLE
# ======================================================


def get_phase_oracle(
    config: LBMConfig,
    roi: RegionOfInterest,
    position_register: list[QuantumModulus],
    oracle_ancillas: list[QuantumBool],
    target_qubit: QuantumBool,
) -> None:
    """
    Converts the flag oracle into a phase oracle.

    Applies a phase flip to all states inside the ROI.
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
    Constructs the oracle for the region of interest.

    Args:
        roi (RegionOfInterest): The region of interest to construct the oracle for.
        position_register: The position register to use for the oracle.
        target_qubit (QuantumBool): The target qubit to use for the oracle.
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
    Constructs the oracle for the subgrid region of interest.

    Args:
        bounds (list[tuple[int, int]]): The bounds of the subgrid.
        position_register: The position register to use for the oracle.
        target_qubit (QuantumBool): The target qubit to use for the oracle.
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
    Returns the oracle for the specified region.

    Args:
        points_list: List of coordinate lists [[x1, y1,...], [x2, y2,...], ...]
        Example for 2D: [[2, 3], [5, 7]]

    Returns:
        A QuantumBool representing the region.
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
