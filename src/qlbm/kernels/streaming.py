from __future__ import annotations

from qrisp import QuantumBool, QuantumFloat, QuantumModulus

from qlbm.config.model import LBMConfig


def streaming_step(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
) -> None:
    """
    Perform one LBM streaming step.

    Each position register is shifted by the corresponding velocity magnitude
    in the direction encoded by its sign qubit.

    Args:
        config: LBM configuration defining the spatial dimensions.
        position_register: Position registers to update.
        velocity_register: Velocity-magnitude registers for each dimension.
        sign_register: Qubits encoding the velocity direction in each dimension.
    """

    for d in range(config.dimensions):
        # Positive direction
        with sign_register[d]:
            position_register[d] += velocity_register[d]

        # Negative direction
        with ~sign_register[d]:
            position_register[d] -= velocity_register[d]
