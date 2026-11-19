from __future__ import annotations

from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumModulus,
    auto_uncompute,
    mcx,
)

from qlbm.config.enums import BCCombination, BoundaryCondition
from qlbm.config.model import LBMConfig


def apply_boundary_conditions(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
) -> None:
    """
    Apply the configured boundary conditions after a streaming step.

    The boundary-condition implementation is selected from the combination
    of periodic, simple-slip, and bounce-back boundaries present in the domain.

    Args:
        config: LBM configuration containing the boundary setup.
        position_register: Position registers after streaming.
        velocity_register: Velocity-magnitude registers.
        sign_register: Qubits encoding the velocity direction.
        boundary_ancillas: Ancillas used for boundary detection.

    Raises:
        ValueError: If the configured boundary-condition combination is unsupported.
    """
    if config.bc_combination == BCCombination.PERIODIC:
        return

    if config.bc_combination == BCCombination.SIMPLE_SLIP:
        _apply_bc_slip(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
        )
        return

    if config.bc_combination == BCCombination.BOUNCE_BACK:
        _apply_bc_bb(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
        )
        return

    if config.bc_combination == BCCombination.BOUNCE_BACK_AND_SIMPLE_SLIP:
        _apply_bc_bb_slip(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
        )
        return

    raise ValueError(f"Unsupported BC combination: {config.bc_combination}")


# ======================================================
# BOUNDARY APPLICATION MODES
# ======================================================


@auto_uncompute
def _apply_bc_slip(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
) -> None:
    """
    Apply simple-slip boundary conditions.

    For each non-periodic dimension, states outside the physical domain are
    reflected by reversing the velocity component normal to that boundary.
    Periodic dimensions remain unchanged.

    Args:
        config: LBM configuration containing the boundary setup.
        position_register: Position registers after streaming.
        velocity_register: Velocity-magnitude registers.
        sign_register: Qubits encoding the velocity direction.
        boundary_ancillas: Ancillas used for boundary detection.
    """
    for d in range(config.dimensions):
        if config.boundary_conditions[d] == BoundaryCondition.PERIODIC:
            continue

        _is_out_boundary(
            config,
            position_register,
            boundary_ancillas,
            d,
        )

        anc = boundary_ancillas[config.boundary_ancilla_indexes[d]]

        with anc:
            # Reverse the velocity component normal to the boundary.
            sign_register[d].flip()

            # Move the streamed state back into the physical domain.
            with sign_register[d]:
                position_register[d] += velocity_register[d]
            with ~sign_register[d]:
                position_register[d] -= velocity_register[d]

        _comes_from_boundary(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
            d,
        )


@auto_uncompute
def _apply_bc_bb(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
) -> None:
    """
    Apply bounce-back boundary conditions.

    After detecting out-of-domain states, the velocity direction is reversed
    in every spatial dimension and the streamed position is reflected back
    into the physical domain.

    Args:
        config: LBM configuration containing the boundary setup.
        position_register: Position registers after streaming.
        velocity_register: Velocity-magnitude registers.
        sign_register: Qubits encoding the velocity direction.
        boundary_ancillas: Ancillas used for boundary detection.
    """
    for d in range(config.dimensions):
        if config.boundary_conditions[d] == BoundaryCondition.PERIODIC:
            continue

        _is_out_boundary(
            config,
            position_register,
            boundary_ancillas,
            d,
        )

    apply_bb = QuantumBool()
    apply_bb.flip()
    # Flag states requiring a global bounce-back reflection.
    mcx(
        [anc for anc in boundary_ancillas],
        apply_bb,
        method="gray",
        ctrl_state=0,
    )

    with apply_bb:
        for d in range(config.dimensions):
            sign_register[d].flip()

            with sign_register[d]:
                position_register[d] += velocity_register[d]
            with ~sign_register[d]:
                position_register[d] -= velocity_register[d]

    apply_bb.uncompute()

    for d in range(config.dimensions):
        if config.boundary_conditions[d] == BoundaryCondition.PERIODIC:
            continue

        _comes_from_boundary(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
            d,
        )


@auto_uncompute
def _apply_bc_bb_slip(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
) -> None:
    """
    Apply a mixed combination of bounce-back and simple-slip boundaries.

    Bounce-back states have all velocity components reversed. For states that
    do not require bounce-back, a simple-slip boundary reverses only the
    velocity component normal to the corresponding boundary. Periodic
    boundaries require no correction.

    Args:
        config: LBM configuration containing the boundary setup.
        position_register: Position registers after streaming.
        velocity_register: Velocity-magnitude registers.
        sign_register: Qubits encoding the velocity direction.
        boundary_ancillas: Ancillas used for boundary detection.
    """
    for d in range(config.dimensions):
        if config.boundary_conditions[d] == BoundaryCondition.PERIODIC:
            continue

        _is_out_boundary(
            config,
            position_register,
            boundary_ancillas,
            d,
        )

    # Detect whether the state requires bounce-back correction.
    apply_bb = QuantumBool()
    apply_bb.flip()

    bb_controls = [
        boundary_ancillas[config.bb_boundary_ancilla_indexes[d]]
        for d in config.bb_boundary_ancilla_indexes
    ]

    mcx(
        bb_controls,
        apply_bb,
        method="gray",
        ctrl_state=0,
    )

    for d in range(config.dimensions):
        bc = config.boundary_conditions[d]

        if bc in (
            BoundaryCondition.PERIODIC,
            BoundaryCondition.BOUNCE_BACK,
        ):
            with apply_bb:
                sign_register[d].flip()

                with sign_register[d]:
                    position_register[d] += velocity_register[d]
                with ~sign_register[d]:
                    position_register[d] -= velocity_register[d]

        # Apply slip only when the state is not already handled by bounce-back.
        if bc == BoundaryCondition.SIMPLE_SLIP:
            apply_slip = QuantumBool()
            apply_slip.flip()

            mcx(
                [
                    apply_bb,
                    boundary_ancillas[config.boundary_ancilla_indexes[d]],
                ],
                apply_slip,
                method="gray",
                ctrl_state=0,
            )

            with apply_slip:
                sign_register[d].flip()

                with sign_register[d]:
                    position_register[d] += velocity_register[d]
                with ~sign_register[d]:
                    position_register[d] -= velocity_register[d]

            apply_slip.uncompute()

    apply_bb.uncompute()

    for d in range(config.dimensions):
        if config.boundary_conditions[d] == BoundaryCondition.PERIODIC:
            continue

        _comes_from_boundary(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
            d,
        )


# ======================================================
# BOUNDARY ANCILLA HELPERS
# ======================================================


@auto_uncompute
def _is_out_boundary(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    boundary_ancillas: list[QuantumBool],
    dim: int,
) -> None:
    """
    Flag states lying outside the physical domain along one dimension.

    The boundary ancilla associated with `dim` is flipped when the encoded
    position lies in a boundary layer outside the physical grid.

    Args:
        config: LBM configuration containing the encoded boundary locations.
        position_register: Position registers to inspect.
        boundary_ancillas: Ancillas used for boundary detection.
        dim: Spatial dimension to check.
    """
    if dim not in config.boundary_ancilla_indexes:
        return

    if config.boundary_offsets[dim] > 1:
        _is_out_boundary_multi_layer(
            config,
            position_register,
            boundary_ancillas,
            dim,
        )
    else:
        _is_out_boundary_single_layer(
            config,
            position_register,
            boundary_ancillas,
            dim,
        )


@auto_uncompute
def _is_out_boundary_single_layer(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    boundary_ancillas: list[QuantumBool],
    dim: int,
) -> None:
    """
    Flag states in either ghost cell of a single-layer boundary.

    Args:
        config: LBM configuration containing the encoded boundary locations.
        position_register: Position registers to inspect.
        boundary_ancillas: Ancillas used for boundary detection.
        dim: Spatial dimension to check.
    """
    anc = boundary_ancillas[config.boundary_ancilla_indexes[dim]]

    # Lower ghost cell.
    with position_register[dim] == (config.boundary_offsets[dim] - 1):
        anc.flip()

    # Upper ghost cell.
    with position_register[dim] == int(
        config.grid_size[dim] + config.boundary_offsets[dim]
    ):
        anc.flip()


@auto_uncompute
def _is_out_boundary_multi_layer(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    boundary_ancillas: list[QuantumBool],
    dim: int,
) -> None:
    """
    Flag states in the ghost region of a multi-layer boundary.

    The ancilla is flipped when the encoded position lies below the physical
    domain or at or above its upper bound.

    Args:
        config: LBM configuration containing the encoded boundary locations.
        position_register: Position registers to inspect.
        boundary_ancillas: Ancillas used for boundary detection.
        dim: Spatial dimension to check.
    """
    anc = boundary_ancillas[config.boundary_ancilla_indexes[dim]]

    with position_register[dim] < config.boundary_offsets[dim]:
        anc.flip()

    with position_register[dim] >= int(
        config.grid_size[dim] + config.boundary_offsets[dim]
    ):
        anc.flip()


@auto_uncompute
def _comes_from_boundary(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
    dim: int,
) -> None:
    """
    Update the boundary ancilla after applying a boundary correction.

    The ancilla is flipped for states in the near-boundary region whose
    velocity points away from the corresponding boundary. This restores
    the ancilla state after the reflected position and velocity have been updated.

    Args:
        config: LBM configuration containing the encoded boundary locations.
        position_register: Corrected position registers.
        velocity_register: Velocity-magnitude registers.
        sign_register: Qubits encoding the corrected velocity direction.
        boundary_ancillas: Ancillas used for boundary detection.
        dim: Spatial dimension being corrected.
    """
    anc = boundary_ancillas[config.boundary_ancilla_indexes[dim]]

    # State is near the lower boundary and moving away from it.
    with position_register[dim] <= velocity_register[dim]:
        with sign_register[dim]:
            anc.flip()

    # State is near the upper boundary and moving away from it.
    upper_bound = int(config.grid_size[dim] + config.boundary_offsets[dim])

    for i in range(1, config.boundary_offsets[dim] + 1):
        with position_register[dim] == upper_bound - i:
            with velocity_register[dim] >= i:
                with ~sign_register[dim]:
                    anc.flip()
