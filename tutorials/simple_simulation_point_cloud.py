from qlbm.api.grid import LBMGrid
from qlbm.config.enums import BoundaryCondition, VelocitySet
from qlbm.config.roi import RegionOfInterest

"""
Simple LBM simulation tutorial.

Demonstrating how to run a simple LBM simulation automatically and step-by-step.
"""


def run_simulation_automatic():
    """
    Run a simple LBM simulation automatically.
    """

    # Define the simulation grid
    grid = LBMGrid(
        grid_size=[8, 8],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.BOUNCE_BACK,
        ],
    )

    # Define the initial bubble region
    initial_bubble = RegionOfInterest.from_points(points={(2, 3), (0, 1)})

    # Define the initial velocity of the bubble
    initial_velocity = [1, -1]

    # Set the initial state of the grid
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Run the simulation with 2 iterations
    grid.run_simulation(iterations=0, init=True)
    mes = grid.measure_density()
    print("Grid density (automatic):", mes)


def run_simulation_step_by_step():
    """
    Run a simple LBM simulation step-by-step.

    Manually trigger initialization, streaming, and boundary condition application.
    """

    # Define the simulation grid
    grid = LBMGrid(
        grid_size=[8, 8],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.BOUNCE_BACK,
        ],
    )

    # Define the initial bubble region
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (2, 3),
            (0, 1),
        ]
    )

    # Define the initial velocity of the bubble
    initial_velocity = [1, -1]

    # Set the initial state of the grid
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Apply initial conditions
    grid.initialize()

    # Iterate through the simulation
    for _ in range(2):
        grid.stream()
        grid.apply_boundary_conditions()

    mes = grid.measure_density()
    print("Grid density (step-by-step):", mes)


if __name__ == "__main__":
    """
    Main entry point for the simple simulation tutorial.
    """

    # Run the simulation automatically
    run_simulation_automatic()

    # Run the simulation step-by-step
    # run_simulation_step_by_step()
