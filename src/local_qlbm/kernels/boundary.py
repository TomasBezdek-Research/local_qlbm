from __future__ import annotations

from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumModulus,
    auto_uncompute,
    mcx,
)

from local_qlbm.config.enums import BCCombination, BoundaryCondition
from local_qlbm.config.model import LBMConfig


def apply_boundary_conditions(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
) -> None:
    """
    Apply configured boundary conditions after one streaming step.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.

    Returns:
        None
    """
    if config.bc_combination == BCCombination.PERIODIC:
        return

    if config.bc_combination == BCCombination.SLIP:
        _apply_bc_slip(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
        )
        return

    if config.bc_combination == BCCombination.BB:
        _apply_bc_bb(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
        )
        return

    if config.bc_combination == BCCombination.BB_AND_SLIP:
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
    Apply slip boundary conditions.
    Used when all active BCs are SIMPLE_SLIP (and/or periodic).

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
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
            # Flip direction
            sign_register[d].flip()

            # Reflect position
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
    Used when all active BCs are BOUNCE_BACK (and/or periodic).

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
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
    Apply mixed boundary conditions:
    some dimensions bounce-back, some slip, some periodic.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
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
    Flip the boundary ancilla for a dimension if position is outside
    the physical domain.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
        dim (int): The dimension index.
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
    Single-layer ghost-cell check.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
        dim (int): The dimension index.
    """
    anc = boundary_ancillas[config.boundary_ancilla_indexes[dim]]

    with position_register[dim] == (config.boundary_offsets[dim] - 1):
        anc.flip()

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
    Multi-layer ghost-cell check.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
        dim (int): The dimension index.
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
    Flip the boundary ancilla if the particle is currently located in the
    near-boundary region and its velocity points away from the boundary.

    This mirrors your original post-correction ancilla logic.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
        dim (int): The dimension index.
    """
    anc = boundary_ancillas[config.boundary_ancilla_indexes[dim]]

    # Lower boundary and moving away from it
    with position_register[dim] <= velocity_register[dim]:
        with sign_register[dim]:
            anc.flip()

    # Upper boundary and moving away from it
    upper_bound = int(config.grid_size[dim] + config.boundary_offsets[dim])

    for i in range(1, config.boundary_offsets[dim] + 1):
        with position_register[dim] == upper_bound - i:
            with velocity_register[dim] >= i:
                with ~sign_register[dim]:
                    anc.flip()
