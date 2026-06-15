from __future__ import annotations

import numpy as np
from qrisp import QuantumBool, QuantumFloat, QuantumModulus, prepare

from local_qlbm.config.model import LBMConfig
from local_qlbm.config.roi import RegionOfInterest


def initialize_state(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    velocity_register: list[QuantumFloat],
    sign_register: list[QuantumBool],
) -> None:
    """
    Initialize full quantum state (positions + velocity).

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
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
    Prepare uniform superposition over ROI subgrid.

    Args:
        config (LBMConfig): The LBM configuration.
        position_register (list[QuantumModulus]): The position register.
        roi (RegionOfInterest): The region of interest.
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
    Prepare velocity magnitude + sign registers.

    Args:
        config (LBMConfig): The LBM configuration.
        velocity_register (list[QuantumFloat]): The velocity register.
        sign_register (list[QuantumBool]): The sign register.
        velocity (np.ndarray): The velocity values.
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
