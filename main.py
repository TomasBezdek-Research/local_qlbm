from local_qlbm.grid import LBMGrid
from local_qlbm.types import BoundaryCondition, RegionOfInterest, VelocitySet

if __name__ == "__main__":
    grid = LBMGrid(
        grid_size=[4, 4, 4],
        velocity_set=VelocitySet.D3Q27,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.SIMPLE_SLIP,
            BoundaryCondition.BOUNCE_BACK,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (0, 0),
            (0, 0),
            (0, 0),
        ]
    )
    initial_velocity = [1, 1, 0]
    grid.set_initial_state(initial_bubble, initial_velocity)
    grid.run_simulation(iterations=20, init=True)
    print(grid.measure_density())
