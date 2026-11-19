from __future__ import annotations

from qrisp import QuantumBool, QuantumFloat, QuantumModulus

from qlbm.config.model import LBMConfig


def allocate_registers(
    config: LBMConfig,
    create_oracle_ancillas: bool = False,
) -> tuple[
    list[QuantumModulus],
    list[QuantumBool],
    list[QuantumFloat],
    list[QuantumBool],
    list[QuantumBool],
]:
    """
    Allocate the quantum registers required by an LBM simulation.

    Position, sign, velocity, and boundary-ancilla registers are allocated
    according to the simulation configuration. Oracle ancillas are allocated
    only when requested.

    Args:
        config: Simulation configuration defining the register sizes.
        create_oracle_ancillas: Whether to allocate ancillas used by region oracles.

    Returns:
        Position registers, sign registers, velocity registers,
        boundary ancillas, and oracle ancillas, in that order.
    """

    # ======== POSITION REGISTERS ========

    position_register = [
        QuantumModulus(int(size)) for size in config.positions_register_shape
    ]

    # ======== SIGN REGISTER ========

    sign_register = [QuantumBool() for _ in range(config.dimensions)]

    # ======== VELOCITY REGISTER ========

    velocity_register = [
        QuantumFloat(config.velocity_qubits_per_dim, signed=False)
        for _ in range(config.dimensions)
    ]

    # ======== BOUNDARY ANCILLAS ========

    boundary_ancillas = [QuantumBool() for _ in config.boundary_ancilla_indexes]

    # ======== ORACLE ANCILLAS ========

    if create_oracle_ancillas:
        oracle_ancillas = [QuantumBool() for _ in range(config.dimensions)]
    else:
        oracle_ancillas = []

    return (
        position_register,
        sign_register,
        velocity_register,
        boundary_ancillas,
        oracle_ancillas,
    )
