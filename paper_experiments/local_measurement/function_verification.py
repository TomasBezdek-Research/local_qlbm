import csv
from collections.abc import Sequence

from local_measurement.region_measurement import (
    EstimationOptions,
    RegionEstimationInput,
    RegionMeasurementMethod,
    estimate_region_probability,
)
from paper_experiments.local_measurement.circut_analysis import analyze_circuit
from qlbm.api.grid import LBMGrid
from qlbm.config.enums import BoundaryCondition, VelocitySet
from qlbm.config.roi import RegionOfInterest

GRID = LBMGrid(
    grid_size=[32, 32],
    velocity_set=VelocitySet.D2Q9,
    boundary_conditions=[BoundaryCondition.SIMPLE_SLIP, BoundaryCondition.BOUNCE_BACK],
)
GRID = LBMGrid(
    grid_size=[8, 8],
    velocity_set=VelocitySet.D2Q9,
    boundary_conditions=[BoundaryCondition.SIMPLE_SLIP, BoundaryCondition.BOUNCE_BACK],
)
INIT_ROI = RegionOfInterest.from_bounds(bounds=[(0, 2), (0, 3)])
INIT_VELOCITY = [-1, -1]
ITERATIONS = 1


def run_analysis_all(iterations: int = 1):
    grid = LBMGrid(
        grid_size=[8, 8],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC],
    )
    init_roi = RegionOfInterest.from_bounds(bounds=[(0, 2), (0, 2)])
    measure_roi = RegionOfInterest.from_points(points={(1, 1), (2, 2), (1, 2), (2, 1)})
    init_velocity = [1, 1]
    eps = 0.05
    alpha = 0.1
    shots = 10

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
    qae_precision: int | None = None,
):
    grid.set_initial_state(bubble_roi=init_roi, velocity=init_velocity)

    region_estimation_input = RegionEstimationInput(
        qargs=grid.qargs_factory(create_oracle_ancillas=True),
        state_function=grid.state_function(simulation_iterations=iterations),
        flag_oracle=grid.flag_oracle(measure_roi),
        phase_oracle=grid.phase_oracle(measure_roi),
        classical_classifier=grid.classical_classifier(measure_roi),
        sampling_measurement=grid.sampling_measurement_function(shots=shots),
    )

    estimation_options = EstimationOptions(
        shots=shots, eps=eps, alpha=alpha, qae_precision=qae_precision
    )

    print("#" * 50)
    print(f"Estimation For method: {method.value}")
    print("-" * 50)
    print()

    res = estimate_region_probability(
        problem=region_estimation_input,
        estimation_method=method,
        options=estimation_options,
    )

    print("-" * 50)
    print(f"Method: {method}, Result: {res}\n")
    print("-" * 50)
    print("-" * 50)
    print()
    return res.probability


def verification(
    method: RegionMeasurementMethod,
    measure_roi: RegionOfInterest,
    eps: float,
    alpha: float,
    shots: int | None = None,
    qae_precision: int | None = None,
):
    return run_analysis(
        grid=GRID,
        method=method,
        init_roi=INIT_ROI,
        measure_roi=measure_roi,
        init_velocity=INIT_VELOCITY,
        eps=eps,
        alpha=alpha,
        shots=shots,
        iterations=ITERATIONS,
        qae_precision=qae_precision,
    )


if __name__ == "__main__":
    repetitions = 10  # 100
    roi_something = RegionOfInterest.from_points(points={(3, 3)})
    roi_zero = RegionOfInterest.from_points(points={(10, 10)})
    roi_interval = RegionOfInterest.from_bounds(bounds=[(1, 7), (0, 2)])
    roi_cloud = RegionOfInterest.from_points(
        points={(0, 2), (3, 0), (20, 20), (15, 12), (30, 29)}
    )

    out_name = "zero_qae"
    roi_mes = roi_zero

    results = {}
    for method in [
        RegionMeasurementMethod.QAE,
        # RegionMeasurementMethod.IQAE,
        # RegionMeasurementMethod.OBPA,
        # RegionMeasurementMethod.CLASSICAL_SAMPLING,
    ]:
        results[method] = []
        for i in range(repetitions):
            results[method].append(
                verification(
                    method=method,
                    measure_roi=roi_mes,
                    eps=2 * 1e-1,
                    alpha=0.05,
                    shots=None,
                    qae_precision=2,
                )
            )
        print(method.name, ":  ", results[method])

    with open(f"output_expanded_{out_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # Create dynamic headers based on the number of elements in the lists
        num_cols = len(next(iter(results.values())))
        header = ["Estimation Method"] + [f"Prob_{i + 1}" for i in range(num_cols)]
        writer.writerow(header)

        # Unpack the list elements directly into the row
        for key, values in results.items():
            writer.writerow([key.name] + values)

    # GRID.set_initial_state(bubble_roi=INIT_ROI, velocity=INIT_VELOCITY)
    # GRID.run_simulation(iterations=1)
    # res = GRID.measure_density()
    # print(res)
    # np_res = GRID.physical_density_to_numpy(res)
    # GRID.plot_numpy_results(np_res, vmin=0, vmax=0.1)
