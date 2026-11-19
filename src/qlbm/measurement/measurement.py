from qrisp import QuantumBool, QuantumFloat, QuantumModulus, multi_measurement
from qrisp.interface.measurement_result import MultiMeasurementResult

from qlbm.config.model import LBMConfig


def measure_density(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    shots: int | None = None,
    backend: object | None = None,
) -> dict[tuple[int, ...], float]:
    """
    Measure the spatial density distribution in physical grid coordinates.

    The measured encoded coordinates are converted back to physical coordinates.
    States outside the physical domain, such as additional boundary layers, are
    excluded from the result.

    Args:
        config: Simulation configuration defining the coordinate encoding.
        position_register: Position registers to measure.
        shots: Number of measurement shots.
        backend: Backend used for measurement.

    Returns:
        Mapping from physical grid coordinates to their measured probabilities.
    """

    raw_data = multi_measurement(
        position_register,
        shots=shots,
        backend=backend,
    )

    physical_data: dict[tuple[int, ...], float] = {}

    for state_tuple, prob in raw_data.items():
        user_coords = tuple(
            config.encoded_to_user_coord(state_tuple[d], d)
            for d in range(config.dimensions)
        )

        is_inside = all(
            0 <= user_coords[d] < config.grid_size[d] for d in range(config.dimensions)
        )

        if is_inside:
            physical_data[user_coords] = physical_data.get(user_coords, 0.0) + prob

    return physical_data


def full_measurement(
    position_register: list[QuantumModulus],
    sign_register: list[QuantumBool],
    velocity_register: list[QuantumFloat],
    shots: int | None = None,
    backend: object | None = None,
) -> MultiMeasurementResult:
    """
    Jointly measure the position, sign, and velocity registers.

    Args:
        position_register: Position registers to measure.
        sign_register: Velocity-sign registers to measure.
        velocity_register: Velocity-magnitude registers to measure.
        shots: Number of measurement shots.
        backend: Backend used for measurement.

    Returns:
        Joint measurement result for all supplied quantum registers.
    """

    return multi_measurement(
        position_register + sign_register + velocity_register,
        shots=shots,
        backend=backend,
    )
