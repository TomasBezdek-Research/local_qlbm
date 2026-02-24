from qiskit.circuit.library.standard_gates.ry import numpy

from local_qlbm.grid import LBMGrid
from local_qlbm.types import BoundaryCondition, RegionOfInterest, VelocitySet

if __name__ == "__main__":
    grid = LBMGrid(
        grid_size=[4, 4],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.BOUNCE_BACK,
            BoundaryCondition.SIMPLE_SLIP,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (1, 3),
            (0, 2),
        ]
    )
    initial_points = RegionOfInterest.from_points(
        points=[
            [1, 1],
            [3, 2],
        ]
    )
    initial_velocity = [-1, -1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    # grid.apply_conditions()

    grid.run_simulation(iterations=1, init=True)
    mes = grid.measure_density()
    np_res = grid.physical_density_to_numpy(mes)

    def plot_res(numpy_results):
        import matplotlib.pyplot as plt
        import numpy as np

        data = numpy_results.T

        plt.imshow(data, cmap="hot", origin="lower")
        plt.xticks(
            ticks=np.arange(data.shape[1]),
            labels=np.arange(data.shape[1]),
            fontsize=10,
        )
        plt.yticks(
            ticks=np.arange(data.shape[0]),
            labels=np.arange(data.shape[0]),
            fontsize=10,
        )
        plt.show()

    print(mes)
    print(np_res)
    plot_res(np_res)
