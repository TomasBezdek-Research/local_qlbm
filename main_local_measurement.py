from local_measurement.region_measurement import (
    EstimationOptions,
    RegionEstimationInput,
    RegionMeasurementMethod,
    estimate_region_probability,
)
from qlbm.api.grid import LBMGrid
from qlbm.config.enums import BoundaryCondition, VelocitySet
from qlbm.config.roi import RegionOfInterest

if __name__ == "__main__":
    grid = LBMGrid(
        grid_size=[4, 4],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC],
    )
    initial_bubble = RegionOfInterest.from_bounds(bounds=[(0, 2), (0, 2)])
    initial_velocity = [1, 1]

    roi = RegionOfInterest.from_bounds(bounds=[(1, 3), (1, 3)])
    roi_2 = RegionOfInterest.from_points(points={(1, 1), (2, 2), (1, 2), (2, 1)})

    measure_roi = roi_2

    grid.set_initial_state(bubble_roi=initial_bubble, velocity=initial_velocity)

    region_estimation_input = RegionEstimationInput(
        qargs=grid.qargs_factory(create_oracle_ancillas=True),
        state_function=grid.state_function(simulation_iterations=0),
        flag_oracle=grid.flag_oracle(measure_roi),
        phase_oracle=grid.phase_oracle(measure_roi),
        classical_classifier=grid.classical_classifier(measure_roi),
        sampling_measurement=grid.sampling_measurement_function(shots=1000),
    )

    estimation_options = EstimationOptions(shots=100, eps=0.6)

    for method in RegionMeasurementMethod:
        print("#" * 50)
        print(f"Estimation For method: {method.value}")
        print("-" * 50)
        print()

        res = estimate_region_probability(
            problem=region_estimation_input,
            estimation_method=method,
            options=estimation_options,
        )
        print(f"Method: {method}, Result: {res}")
        print()
        print("-" * 50)
        print()
