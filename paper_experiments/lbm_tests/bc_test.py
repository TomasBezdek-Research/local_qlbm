from pathlib import Path

from qlbm.api.grid import LBMGrid
from qlbm.config.enums import BoundaryCondition, VelocitySet
from qlbm.config.roi import RegionOfInterest

"""
2D LBM Visualization of Boundary Conditions
"""


def test_bounce_back(output_path, iterations=3):
    grid = LBMGrid(
        grid_size=[4, 2],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.BOUNCE_BACK,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (0, 1),
            (0, 1),
        ]
    )
    initial_velocity = [1, 1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Run the simulation with visualization
    grid.run_simulation_with_visualization(
        f"{output_path}/bounce_back",
        iterations=iterations,
        init=True,
        time_steps_per_frame=1,
    )


def test_simple_slip(output_path, iterations=3):
    grid = LBMGrid(
        grid_size=[4, 2],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.SIMPLE_SLIP,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (0, 1),
            (0, 1),
        ]
    )
    initial_velocity = [1, 1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Run the simulation with visualization
    grid.run_simulation_with_visualization(
        f"{output_path}/simple_slip",
        iterations=iterations,
        init=True,
        time_steps_per_frame=1,
    )


def test_periodic(output_path, iterations=3):
    grid = LBMGrid(
        grid_size=[4, 2],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.PERIODIC,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (0, 1),
            (0, 1),
        ]
    )
    initial_velocity = [1, 1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Run the simulation with visualization
    grid.run_simulation_with_visualization(
        f"{output_path}/periodic",
        iterations=iterations,
        init=True,
        time_steps_per_frame=1,
    )


if __name__ == "__main__":
    # Define the output path for the visualization frames
    current_file = Path(__file__).resolve()
    base_dir = current_file.parent.parent
    output_path = base_dir / "output" / "2d_visualization"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_bc = base_dir / "output" / "bc_tests"
    output_bc.mkdir(parents=True, exist_ok=True)
    test_bounce_back(output_bc)
    test_simple_slip(output_bc)
    test_periodic(output_bc)
