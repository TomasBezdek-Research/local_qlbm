from pathlib import Path

from local_qlbm.grid import LBMGrid
from local_qlbm.types import BoundaryCondition, RegionOfInterest, VelocitySet

"""
2D LBM Visualization Tutorial

This tutorial demonstrates how to set up and run a 2D LBM simulation with visualization.

This is just for demonstration purposes. By Measuring state after each iteration, quantum advantage is lost.
Purpouse is to visualize the simulation to confirm the simulation is running correctly.

Output is saved to the `output` directory in time series with names `2d_visualization_<iteration>.vts`.
"""

if __name__ == "__main__":
    # Define the output path for the visualization frames
    current_file = Path(__file__).resolve()
    base_dir = current_file.parent.parent
    output_path = base_dir / "output" / "2d_visualization"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the LBM grid and set initial state
    grid = LBMGrid(
        grid_size=[8, 8],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.BOUNCE_BACK,
            BoundaryCondition.PERIODIC,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (0, 1),
            (0, 1),
        ]
    )
    initial_velocity = [0, -1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Run the simulation with visualization
    grid.run_simulation_with_visualization(
        output_path, iterations=20, init=True, time_steps_per_frame=1
    )
