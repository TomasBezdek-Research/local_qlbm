from local_qlbm.grid import LBMGrid
from local_qlbm.types import BoundaryCondition, RegionOfInterest, VelocitySet

"""
Region analysis tutorial.

This tutorial demonstrates how to use the `LBMGrid` class to analyze a region of interest in the simulation.
Use `analyze_region` to measure the density of a region of interest.

Function apply IQAE to measure the density of a region of interest.
Than use this information to amplify the region and measure the density again.
At the end, amplified densities are rescaled to the original density.
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

    # Set the region of interest to analyze
    roi = RegionOfInterest.from_bounds(
        bounds=[
            (0, 2),
            (1, 3),
        ]
    )

    # Analyze the region and print the results
    print(grid.analyze_region(roi, simulation_iterations=0, alpha=0.1))
