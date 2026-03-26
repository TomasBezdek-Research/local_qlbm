from local_qlbm.grid import LBMGrid
from local_qlbm.types import BoundaryCondition, RegionOfInterest, VelocitySet

"""
Numpy Result Tutorial

This tutorial demonstrates how to set up and run a 2D LBM simulation and retrieve the results as a numpy array.
"""
if __name__ == "__main__":
    # Create an LBM grid and set the initial state
    grid = LBMGrid(
        grid_size=[8, 8],
        velocity_set=VelocitySet.D2Q9,
        boundary_conditions=[
            BoundaryCondition.PERIODIC,
            BoundaryCondition.BOUNCE_BACK,
        ],
    )
    initial_bubble = RegionOfInterest.from_bounds(
        bounds=[
            (2, 3),
            (0, 1),
        ]
    )
    initial_velocity = [1, -1]
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Run the simulation and measure the density
    grid.run_simulation(iterations=2, init=True)
    mes = grid.measure_density()
    print("Grid density:", mes)

    # Plot the results
    def plot_res(numpy_results):
        """
        Plot the results as a heatmap using matplotlib.

        Args:
            numpy_results (np.ndarray): The numpy array of results to plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        data = numpy_results.T

        im = plt.imshow(data, cmap="coolwarm", origin="lower")
        plt.colorbar(im)
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

    # Convert the physical density to a numpy array
    np_res = grid.physical_density_to_numpy(mes)

    # Print the numpy array
    print(np_res)

    # Plot the numpy array
    plot_res(np_res)
