from local_qlbm.api.grid import LBMGrid
from local_qlbm.api.measurement import measure_density
from local_qlbm.config.enums import BoundaryCondition, VelocitySet
from local_qlbm.config.roi import RegionOfInterest

"""
This tutorial demonstrates how to use the region density amplification feature of the LBM grid.

It uses QAA (Quantum Amplitude Amplification) to amplify the density of a region of interest in the grid.
"""
if __name__ == "__main__":
    # Create a grid and set initial state
    grid = LBMGrid(
        grid_size=[4, 4],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.SIMPLE_SLIP,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (0, 2),
            (0, 4),
        ]
    )
    initial_velocity = [1, 1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Define the region of interest for amplification
    roi = RegionOfInterest.from_bounds(
        bounds=[
            (1, 2),
            (1, 3),
        ]
    )

    # Define number of simulation iterations
    num_iterations = 1

    # Run simulation
    grid.run_simulation(iterations=num_iterations, init=True)

    # Measure density after simulation just for demostration purposes
    mes = grid.measure_density()
    print("Grid density:", mes)

    # Estimate region density using fast method to see the density of the region that will be amplified
    den = grid.fast_region_density_estimate(roi)
    print("Region density (fast):", den)

    # Estimate region density using precise method - IQAE (may be slower)
    pos, _, _ = grid.amplify_region(
        roi,
        amplification_iterations=1,
        simulation_iterations=num_iterations,
    )

    # Measure density after amplification
    amp_mes = measure_density(config=grid.config, position_register=pos)
    print("Amplified region density:", amp_mes)
