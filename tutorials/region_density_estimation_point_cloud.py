from local_qlbm.api.grid import LBMGrid
from local_qlbm.config.enums import BoundaryCondition, VelocitySet
from local_qlbm.config.roi import RegionOfInterest

"""
This tutorial demonstrates how to estimate the density of a region in an LBM simulation.

Uses fast region density estimation based on agregating all information on one qubit and than measure.
Uses IQAE for precise density estimation.
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

    # Define a region of interest (ROI) to estimate density
    roi = RegionOfInterest.from_points(
        set(
            [
                (1, 2),
                (1, 3),
            ]
        )
    )

    # Define simulation parameters
    simulation_iterations = 1

    # Run the simulation for purpouse of fast estimation
    grid.run_simulation(iterations=simulation_iterations, init=True)

    # Measure density after simulation just for demostration purposes
    mes = grid.measure_density()
    print("Grid density:", mes)

    # Estimate region density using fast method
    den = grid.fast_region_density_estimate(roi)
    print("Region density (fast):", den)

    # Estimate region density using precise method - IQAE (may be slower)
    den_prec = grid.estimate_region_density(
        roi,
        eps=0.1,
        alpha=0.2,
        simulation_iterations=simulation_iterations,
    )
    print("Region density (precise):", den_prec)
