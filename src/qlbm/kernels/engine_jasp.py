from __future__ import annotations

from qrisp import QuantumBool, QuantumFloat, QuantumModulus

from qlbm.config.model import LBMConfig
from qlbm.kernels.boundary import apply_boundary_conditions
from qlbm.kernels.initialization import initialize_state
from qlbm.kernels.registers import allocate_registers
from qlbm.kernels.streaming import streaming_step


def simulation_step(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    boundary_ancillas: list[QuantumBool],
) -> None:
    """
    Perform one complete LBM simulation step.

    The position registers are first updated by the streaming operation,
    after which the configured boundary conditions are applied.

    Args:
        config: LBM configuration defining the simulation setup.
        position_register: Position registers to update.
        velocity_register: Velocity-magnitude registers.
        sign_register: Qubits encoding the velocity direction.
        boundary_ancillas: Ancillas used for boundary handling.
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
    Advance an initialized LBM state for a specified number of iterations.

    The supplied registers are assumed to be already allocated and initialized.
    Each iteration consists of streaming followed by boundary-condition handling.

    Args:
        config: LBM configuration defining the simulation setup.
        position_register: Initialized position registers.
        velocity_register: Initialized velocity-magnitude registers.
        sign_register: Qubits encoding the velocity direction.
        boundary_ancillas: Ancillas used for boundary handling.
        iterations: Number of simulation steps to perform.
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
    Allocate registers, prepare the initial state, and run the simulation.

    Args:
        config: LBM configuration containing the simulation setup and
            initial conditions.
        iterations: Number of simulation steps to perform.
        create_oracle_ancillas: Whether to allocate ancillas used by
            region oracles.

    Returns:
        Position registers, sign registers, velocity registers,
        boundary ancillas, and oracle ancillas, in that order.
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
    Allocate registers and prepare the initial state without advancing the simulation.

    Args:
        config: LBM configuration containing the simulation setup and
            initial conditions.
        create_oracle_ancillas: Whether to allocate ancillas used by
            region oracles.

    Returns:
        Position registers, sign registers, velocity registers,
        boundary ancillas, and oracle ancillas, in that order.
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
