from qrisp import QuantumBool, QuantumFloat, QuantumModulus, multi_measurement

from local_qlbm.config.model import LBMConfig


def measure_density(
    config: LBMConfig,
    position_register: list[QuantumModulus],
    shots: int | None = None,
    backend: object | None = None,
) -> dict[tuple[int, ...], float]:
    """
    Measure position density and convert it back to physical coordinates
    (ghost cells removed).

    Args:
        shots (int | None, optional): Number of shots to run. Defaults to None.
        backend (object | None, optional): Backend to run the simulation on. Defaults to None.
        position_register (list[QuantumModulus]): Position register to measure.

    Returns:
        dict[tuple[int, ...], float]: Dictionary of physical coordinates and their corresponding density.
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
) -> dict[tuple[object, ...], float]:
    """
    Measure positions + signs + velocity registers together.
    """

    return multi_measurement(
        position_register + sign_register + velocity_register,
        shots=shots,
        backend=backend,
    )
