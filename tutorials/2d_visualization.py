from pathlib import Path

from local_qlbm.api.grid import LBMGrid
from local_qlbm.config.enums import BoundaryCondition, VelocitySet
from local_qlbm.config.roi import RegionOfInterest
from local_qlbm.io.vtk import clear_vtk_series

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
        grid_size=[32, 8],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.SIMPLE_SLIP,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (1, 4),
            (2, 6),
        ]
    )
    initial_velocity = [1, -1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    clear_vtk_series(output_path)

    # Run the simulation with visualization
    grid.run_simulation_with_visualization(
        output_path, iterations=40, init=True, time_steps_per_frame=1
    )
