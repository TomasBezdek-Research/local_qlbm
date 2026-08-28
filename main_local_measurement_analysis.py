from collections.abc import Sequence

from local_measurement.region_measurement import (
    EstimationOptions,
    RegionEstimationInput,
    RegionMeasurementMethod,
)
from local_measurement.region_measurement_analysis import (
    estimate_region_probability_analysis,
)
from paper_experiments.local_measurement.circut_analysis import analyze_circuit
from qlbm.api.grid import LBMGrid
from qlbm.config.enums import BoundaryCondition, VelocitySet
from qlbm.config.roi import RegionOfInterest


def run_analysis_all(iterations: int = 1):
    grid = LBMGrid(
        grid_size=[4, 4],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC],
    )
    init_roi = RegionOfInterest.from_bounds(bounds=[(0, 2), (0, 2)])
    measure_roi = RegionOfInterest.from_points(points={(1, 1), (2, 2), (1, 2), (2, 1)})
    init_velocity = [1, 1]
    eps = 0.05
    alpha = 0.1
    shots = 1000

    for method in RegionMeasurementMethod:
        run_analysis(
            grid=grid,
            method=method,
            init_roi=init_roi,
            measure_roi=measure_roi,
            init_velocity=init_velocity,
            eps=eps,
            alpha=alpha,
            shots=shots,
            iterations=0,
        )


def run_analysis(
    grid: LBMGrid,
    method: RegionMeasurementMethod,
    init_roi: RegionOfInterest,
    measure_roi: RegionOfInterest,
    init_velocity: list,
    eps: float,
    alpha: float,
    shots: int | None = None,
    iterations: int = 1,
):
    grid = LBMGrid(
        grid_size=[4, 4],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC],
    )

    grid.set_initial_state(bubble_roi=init_roi, velocity=init_velocity)

    region_estimation_input = RegionEstimationInput(
        qargs=grid.qargs_factory(create_oracle_ancillas=True),
        state_function=grid.state_function(simulation_iterations=iterations),
        flag_oracle=grid.flag_oracle(measure_roi),
        phase_oracle=grid.phase_oracle(measure_roi),
        classical_classifier=grid.classical_classifier(measure_roi),
        sampling_measurement=grid.sampling_measurement_function(shots=shots),
    )

    estimation_options = EstimationOptions(shots=shots, eps=eps, alpha=alpha)

    print("#" * 50)
    print(f"Estimation For method: {method.value}")
    print("-" * 50)
    print()

    res, quantum_sessions, shot_counts = estimate_region_probability_analysis(
        problem=region_estimation_input,
        estimation_method=method,
        options=estimation_options,
    )
    if not isinstance(quantum_sessions, Sequence):
        quantum_sessions = [quantum_sessions]
    if not isinstance(shot_counts, Sequence):
        shot_counts = [shot_counts]

    print(f"Method: {method}, Result: {res}\n")
    print("-" * 50)
    print()
    for quantum_session, shot_count in zip(quantum_sessions, shot_counts):
        print(f"SHOTS: {shot_count}\n")
        _ = analyze_circuit(quantum_session)
        print("-" * 50)
        print()
    print()
    print("-" * 50)
    print("-" * 50)
    print()


if __name__ == "__main__":
    # run_analysis_all(iterations=0)

    grid = LBMGrid(
        grid_size=[4, 4],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC],
    )
    init_roi = RegionOfInterest.from_bounds(bounds=[(0, 2), (0, 2)])
    measure_roi = RegionOfInterest.from_points(points={(1, 1), (2, 2), (1, 2), (2, 1)})
    init_velocity = [1, 1]
    eps = 0.02
    alpha = 0.1
    shots = 1000
    method = RegionMeasurementMethod.IQAE

    # run_analysis_all()

    run_analysis(
        grid=grid,
        method=method,
        init_roi=init_roi,
        measure_roi=measure_roi,
        init_velocity=init_velocity,
        eps=eps,
        alpha=alpha,
        shots=shots,
        iterations=1,
    )
