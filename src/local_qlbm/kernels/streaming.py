from __future__ import annotations

from qrisp import QuantumBool, QuantumFloat, QuantumModulus

from local_qlbm.config.model import LBMConfig


def streaming_step(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
) -> None:
    """
    Perform one LBM streaming (advection) step.

    Updates positions according to velocity and direction.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
    """

    for d in range(config.dimensions):
        # Positive direction
        with sign_register[d]:
            position_register[d] += velocity_register[d]

        # Negative direction
        with ~sign_register[d]:
            position_register[d] -= velocity_register[d]
