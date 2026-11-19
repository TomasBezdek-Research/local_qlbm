import argparse
import csv
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

GRID = LBMGrid(
    grid_size=[4, 4],
    velocity_set=VelocitySet.D2Q9,
    boundary_conditions=[BoundaryCondition.SIMPLE_SLIP, BoundaryCondition.BOUNCE_BACK],
)
INIT_ROI = RegionOfInterest.from_bounds(bounds=[(0, 3), (0, 3)])
INIT_VELOCITY = [-1, -1]
ITERATIONS = 1


def run_analysis_detailed(
    grid: LBMGrid,
    method: RegionMeasurementMethod,
    init_roi: RegionOfInterest,
    measure_roi: RegionOfInterest,
    init_velocity: list,
    shots: int,
    run_idx: int,
    iterations: int = 1,
    eps: float | None = None,
    alpha: float | None = None,
    qae_precision: int | None = None,
):
    grid.set_initial_state(bubble_roi=init_roi, velocity=init_velocity)

    region_estimation_input = RegionEstimationInput(
        qargs=grid.qargs_factory(create_oracle_ancillas=True),
        state_function=grid.state_function(simulation_iterations=iterations),
        flag_oracle=grid.flag_oracle(measure_roi),
        phase_oracle=grid.phase_oracle(measure_roi),
        classical_classifier=grid.classical_classifier(measure_roi),
        sampling_measurement=grid.sampling_measurement_function(),
    )

    estimation_options = EstimationOptions(shots=shots)

    print("#" * 50)
    print(f"Estimation For method: {method.value} | Shots: {shots} | Run: {run_idx}")
    print("-" * 50)

    # Use the analysis version to get sessions and shot counts
    res, quantum_sessions, shot_counts = estimate_region_probability_analysis(
        problem=region_estimation_input,
        estimation_method=method,
        options=estimation_options,
    )

    print("\n\nESTIMATION RESULTS: ", res.probability)

    if not isinstance(quantum_sessions, Sequence):
        quantum_sessions = [quantum_sessions]
    if not isinstance(shot_counts, Sequence):
        shot_counts = [shot_counts]

    run_records = []

    # Loop through each iterative quantum session / circuit generated
    for idx, (quantum_session, shot_count) in enumerate(
        zip(quantum_sessions, shot_counts)
    ):
        print(f"--- Iteration/Session {idx + 1} (SHOTS: {shot_count}) ---")
        circuit_metrics = analyze_circuit(quantum_session)
        print("-" * 50)

        # Build a record combining setup parameters, run index, and circuit metrics
        record = {
            "Shots_Config": shots,
            "Run_Index": run_idx,
            "Session_Index": idx + 1,
            "Actual_Shots": shot_count,
            "Estimated_Probability": res.probability,
            "Lower_Bound": res.lower_bound,
            "Upper_Bound": res.upper_bound,
            **circuit_metrics,  # Unpacks num_qubits, depth, gate counts, etc.
        }
        run_records.append(record)

    return run_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run shot-based region measurement scaling experiments (SimpleSampling, OBPA)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="SimpleSampling",
        choices=["OBPA", "SimpleSampling"],
        help="Select the measurement method.",
    )
    parser.add_argument(
        "--oracle",
        type=str,
        default="point",
        choices=["point", "zero", "interval", "cloud"],
        help="Select the oracle / region of interest (ROI).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of repetitions per shot configuration to compute empirical error/confidence.",
    )
    args = parser.parse_args()

    # Mappings for methods and oracles
    method_map = {
        "IQAE": RegionMeasurementMethod.IQAE,
        "QAE": RegionMeasurementMethod.QAE,
        "OBPA": RegionMeasurementMethod.OBPA,
        "SimpleSampling": RegionMeasurementMethod.CLASSICAL_SAMPLING,
    }

    oracle_map = {
        "point": RegionOfInterest.from_points(points={(1, 1)}),
        "zero": RegionOfInterest.from_points(points={(2, 3)}),
        "interval": RegionOfInterest.from_bounds(bounds=[(1, 4), (0, 2)]),
        "cloud": RegionOfInterest.from_points(points={(0, 1), (2, 0), (2, 3), (3, 1)}),
    }

    selected_method = method_map[args.method]
    selected_roi = oracle_map[args.oracle]

    # Define shot configurations to sweep over instead of eps/alpha
    shots_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 10000]
    num_runs = args.runs

    out_name = f"scaling_output/shot_scaling_{args.method.lower()}_{args.oracle}"
    all_detailed_records = []

    print(
        f"Starting Shot-Based Scaling Experiment | Method: {args.method} | Oracle: {args.oracle} | Repetitions: {num_runs}..."
    )

    for shots in shots_list:
        for run_idx in range(1, num_runs + 1):
            records = run_analysis_detailed(
                grid=GRID,
                method=selected_method,
                init_roi=INIT_ROI,
                measure_roi=selected_roi,
                init_velocity=INIT_VELOCITY,
                shots=shots,
                run_idx=run_idx,
            )
            all_detailed_records.extend(records)

    # Write all iteration records into a single comprehensive CSV file
    csv_filename = f"{out_name}.csv"
    if all_detailed_records:
        fieldnames = list(all_detailed_records[0].keys())
        with open(csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detailed_records)

    print(
        f"\nExperiment completed! All per-run and per-iteration metrics saved to {csv_filename}"
    )
