from pathlib import Path

from local_qlbm.grid import LBMGrid
from local_qlbm.types import BoundaryCondition, RegionOfInterest, VelocitySet

"""
3D LBM Visualization Tutorial

This tutorial demonstrates how to set up and run a 2D LBM simulation with visualization.

This is just for demonstration purposes. By Measuring state after each iteration, quantum advantage is lost.
Purpouse is to visualize the simulation to confirm the simulation is running correctly.

Output is saved to the `output` directory in time series with names `3d_visualization_<iteration>.vts`.
"""
if __name__ == "__main__":
    # Set up the output path
    current_file = Path(__file__).resolve()
    base_dir = current_file.parent.parent
    output_path = base_dir / "output" / "3d_visualization"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up the grid and initial conditions
    grid = LBMGrid(
        grid_size=[4, 4, 8],
        velocity_set=VelocitySet.D3Q27,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.SIMPLE_SLIP,
            BoundaryCondition.BOUNCE_BACK,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (1, 2),
            (2, 3),
            (1, 2),
        ]
    )
    initial_velocity = [1, -1, -1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Run the simulation with visualization
    grid.run_simulation_with_visualization(output_path, iterations=20, init=True)
