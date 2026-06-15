from __future__ import annotations

from qrisp import QuantumBool, QuantumFloat, QuantumModulus

from local_qlbm.config.model import LBMConfig
from local_qlbm.kernels.boundary import apply_boundary_conditions
from local_qlbm.kernels.initialization import initialize_state
from local_qlbm.kernels.registers import allocate_registers
from local_qlbm.kernels.streaming import streaming_step


def simulation_step(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
) -> None:
    """
    Perform one full simulation step:
    1. streaming
    2. boundary-condition correction

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
    """
    streaming_step(
        config,
        position_register,
        velocity_register,
        sign_register,
    )

    apply_boundary_conditions(
        config,
        position_register,
        velocity_register,
        sign_register,
        boundary_ancillas,
    )


def run_simulation(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
    iterations: int = 1,
) -> None:
    """
    Run the simulation on already-allocated and already-initialized registers.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
        boundary_ancillas (list[QuantumBool]): The boundary ancillas.
        iterations (int): The number of iterations to run.
    """
    for _ in range(iterations):
        simulation_step(
            config,
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
        )


def initialize_and_run(
    config: LBMConfig,
    iterations: int = 1,
    create_oracle_ancillas: bool = False,
) -> tuple[
    list[QuantumModulus],
    list[QuantumBool],
    list[QuantumFloat],
    list[QuantumBool],
    list[QuantumBool],
]:
    """
    Allocate registers, initialize the quantum state, and run the simulation.

    Args:
        config (LBMConfig): The LBM configuration.
        iterations (int): The number of iterations to run.
        create_oracle_ancillas (bool): Whether to create oracle ancillas.

    Returns:
        (
            position_register,
            sign_register,
            velocity_register,
            boundary_ancillas,
            oracle_ancillas,
        )
    """
    (
        position_register,
        sign_register,
        velocity_register,
        boundary_ancillas,
        oracle_ancillas,
    ) = allocate_registers(
        config,
        create_oracle_ancillas=create_oracle_ancillas,
    )

    initialize_state(
        config,
        position_register,
        velocity_register,
        sign_register,
    )

    run_simulation(
        config,
        position_register,
        velocity_register,
        sign_register,
        boundary_ancillas,
        iterations=iterations,
    )

    return (
        position_register,
        sign_register,
        velocity_register,
        boundary_ancillas,
        oracle_ancillas,
    )


def initialize_only(
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
    Allocate registers and initialize the state, but do not run any time steps.

    Args:
        config (LBMConfig): The LBM configuration.
        create_oracle_ancillas (bool): Whether to create oracle ancillas.

    Returns:
        (
            position_register,
            sign_register,
            velocity_register,
            boundary_ancillas,
            oracle_ancillas,
        )
    """
    (
        position_register,
        sign_register,
        velocity_register,
        boundary_ancillas,
        oracle_ancillas,
    ) = allocate_registers(
        config,
        create_oracle_ancillas=create_oracle_ancillas,
    )

    initialize_state(
        config,
        position_register,
        velocity_register,
        sign_register,
    )

    return (
        position_register,
        sign_register,
        velocity_register,
        boundary_ancillas,
        oracle_ancillas,
    )
