from __future__ import annotations

import numpy as np
from qrisp import QuantumBool, QuantumFloat, QuantumModulus, prepare

from qlbm.config.model import LBMConfig
from qlbm.config.roi import RegionOfInterest


def initialize_state(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
) -> None:
    """
    Prepare the initial position and velocity state.

    The position registers are initialized from the configured region of interest,
    while the velocity magnitude and direction are encoded in separate registers.

    Args:
        config: LBM configuration containing the initial conditions.
        position_register: Position registers to initialize.
        velocity_register: Velocity-magnitude registers to initialize.
        sign_register: Qubits encoding the velocity direction.

    Raises:
        ValueError: If the initial region or velocity has not been configured.
        NotImplementedError: If point-cloud initialization is requested.
    """

    if config.initial_bubble is None:
        raise ValueError("Initial bubble is not set.")

    if config.initial_velocity is None:
        raise ValueError("Initial velocity is not set.")

    if config.initial_bubble.is_subgrid:
        _initialize_subgrid(
            config,
            position_register,
            config.initial_bubble,
        )
    else:
        raise NotImplementedError("Point-cloud initialization is not implemented.")

    _initialize_velocity(
        config,
        velocity_register,
        sign_register,
        config.initial_velocity,
    )


# ======================================================
# POSITION INITIALIZATION
# ======================================================


def _initialize_subgrid(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    roi: RegionOfInterest,
) -> None:
    """
    Prepare a uniform superposition over a rectangular region of interest.

    Each position dimension is initialized independently to a uniform
    superposition over its encoded interval.

    Args:
        config: LBM configuration defining the coordinate encoding.
        position_register: Position registers to initialize.
        roi: Subgrid region defining the initial position state.

    Raises:
        ValueError: If the initialization region is empty.
    """

    encoded_bounds = config.get_encoded_bounds(roi.bounds)

    for d in range(config.dimensions):
        num_states = 2 ** position_register[d].size
        statevector = np.zeros(num_states)

        start, end = encoded_bounds[d]
        statevector[start:end] = 1.0

        # Normalize
        norm = np.linalg.norm(statevector)
        if norm == 0:
            raise ValueError("Initialization region is empty.")
        statevector /= norm

        prepare(position_register[d], statevector)


# ======================================================
# VELOCITY INITIALIZATION
# ======================================================


def _initialize_velocity(
    config: LBMConfig,
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
    velocity: np.ndarray,
) -> None:
    """
    Prepare the velocity-magnitude and direction registers.

    For each spatial dimension, the absolute velocity is encoded in the
    velocity register and its direction is encoded in the corresponding sign qubit.

    Args:
        config: LBM configuration defining the spatial dimensions.
        velocity_register: Registers encoding the velocity magnitudes.
        sign_register: Qubits encoding the velocity directions.
        velocity: Initial velocity vector.
    """

    for d in range(config.dimensions):
        u_val = int(velocity[d])

        # ======== VELOCITY (magnitude) ========

        num_states = 2 ** velocity_register[d].size
        statevector = np.zeros(num_states)

        statevector[abs(u_val)] = 1.0
        prepare(velocity_register[d], statevector)

        # ======== SIGN ========

        if u_val >= 0:
            # positive direction
            prepare(sign_register[d], np.array([0.0, 1.0]))
        else:
            # negative direction
            prepare(sign_register[d], np.array([1.0, 0.0]))
