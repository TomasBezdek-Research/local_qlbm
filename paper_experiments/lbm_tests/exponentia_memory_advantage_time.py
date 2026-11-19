import csv

from qrisp import QuantumModulus

from qlbm.api.grid import LBMGrid
from qlbm.config.enums import BoundaryCondition, VelocitySet
from qlbm.config.roi import RegionOfInterest

"""
Simple LBM simulation tutorial.

Demonstrating how to run a simple LBM simulation automatically and step-by-step.
"""


def run_simulation_depth(
    d: int = 1,
    m: int = 1,
    bc: BoundaryCondition = BoundaryCondition.PERIODIC,
    iterations: int = 1,
) -> QuantumModulus:
    """
    Run a simple LBM simulation automatically.
    """

    grid_size = [2**m] * d

    velocity_set = VelocitySet.D2Q9
    match d:
        case 1:
            velocity_set = VelocitySet.D1Q3
        case 2:
            velocity_set = VelocitySet.D2Q9
        case 3:
            velocity_set = VelocitySet.D3Q27
        case _:
            raise ValueError("Unsupported dimension")
    boundary_conditions = [bc] * d
    print(
        "grid_size:",
        grid_size,
        "velocity_set:",
        velocity_set,
        "boundary_conditions:",
        boundary_conditions,
    )

    # Define the simulation grid
    grid = LBMGrid(
        grid_size=grid_size,
        velocity_set=velocity_set,
        boundary_conditions=boundary_conditions,
    )

    # Define the initial bubble region
    initial_bubble = RegionOfInterest.from_bounds(bounds=[(0, 1)] * d)

    # Define the initial velocity of the bubble
    initial_velocity = [1] * d

    # Set the initial state of the grid
    grid.set_initial_state(initial_bubble, initial_velocity)

    # Run the simulation with 2 iterations
    grid.run_simulation(iterations=iterations, init=True)
    grid.measure_density()

    return grid.position_register[0]


def analyze_circuit(
    d: int = 1,
    m: int = 1,
    bc: BoundaryCondition = BoundaryCondition.PERIODIC,
    iterations: int = 1,
):
    pos = run_simulation_depth(d=d, m=m, bc=bc, iterations=iterations)
    res = pos.qs.compile().transpile()

    print(res.num_qubits())

    data = res.count_ops()
    gate_count = sum(data.values())

    result = {
        "dimension": d,
        "m": m,
        "N": 2**m,
        "grid_size": (2**m) ** d,
        "boundary_condition": bc.name,
        "iterations": iterations,
        "num_qubits": res.num_qubits(),
        "depth": res.depth(),
        "gate_count": gate_count,
        "cnot_total": res.cnot_count(),
        "x_count": data.get("x", 0),
        "h_count": data.get("h", 0),
        "u3_count": data.get("u3", 0),
        "cx_count": data.get("cx", 0),
        "t_count": data.get("t", 0),
        "t_dg_count": data.get("t_dg", 0),
        "p_count": data.get("p", 0),
    }

    # ✅ PRINT (same as before)
    print("VALUE FOR d: ", d, "\n")
    print("VALUE FOR m: ", m, ", n=2^m: ", 2**m, "\n")
    print("GRID SIZE - (2^m)^d: ", (2**m) ** d, "\n")
    print("ITERATIONS: ", iterations, "\n")
    print("BOUNDARY CONDITION:", bc.name, "\n")
    print("NUMBER OF QUBITS: ", result["num_qubits"], "\n")
    print("DEPTH OF CIRCUIT: ", result["depth"], "\n")
    print("NUMBER OF GATES: ", result["gate_count"], "\n")
    print("NUMBER OF X-GATES: ", result["x_count"], "\n")
    print("NUMBER OF HADAMARD-GATES: ", result["h_count"], "\n")
    print("NUMBER OF U3-GATES: ", result["u3_count"], "\n")
    print("NUMBER OF CX-GATES: ", result["cx_count"], "\n")
    print("NUMBER OF T-GATES: ", result["t_count"], "\n")
    print("NUMBER OF T-DG-GATES: ", result["t_dg_count"], "\n")
    print("NUMBER OF P-GATES: ", result["p_count"], "\n")
    print("COMPLETE DATA: ", data, "\n")

    return result


def save_results_to_csv(results, filename="scaling_analysis.csv"):
    if not results:
        return

    keys = results[0].keys()

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def save_results_to_txt(results, filename="scaling_analysis.txt"):
    with open(filename, "w", encoding="utf-8") as out:
        for r in results:
            out.write(f"Boundary Condition: {r['boundary_condition']}\n")
            out.write(f"Value for d: {r['dimension']}\n")
            out.write(f"Value for m: {r['m']}, n=2^m: {r['N']}\n")
            out.write(f"Grid Size - (2^m)^d: {r['grid_size']}\n")
            out.write(f"Iterations: {r['iterations']}\n\n")

            out.write(f"Number of Qubits: {r['num_qubits']}\n")
            out.write(f"Depth of circuit: {r['depth']}\n")
            out.write(f"Number of Gates: {r['gate_count']}\n")
            out.write(f"Number of x-Gates: {r['x_count']}\n")
            out.write(f"Number of Hadamard-Gates: {r['h_count']}\n")
            out.write(f"Number of u3-Gates: {r['u3_count']}\n")
            out.write(f"Number of cx-Gates: {r['cx_count']}\n")
            out.write(f"Number of t-Gates: {r['t_count']}\n")
            out.write(f"Number of t-dg-Gates: {r['t_dg_count']}\n")
            out.write(f"Number of p-Gates: {r['p_count']}\n")

            out.write("\n" + "#" * 50 + "\n\n")


if __name__ == "__main__":
    # all_results_grid_scaling = []

    # for bc in [
    #     BoundaryCondition.PERIODIC,
    #     BoundaryCondition.BOUNCE_BACK,
    #     BoundaryCondition.SIMPLE_SLIP,
    # ]:
    #     print(f"\nBoundary Condition: {bc.name}")
    #     print("#" * 50)

    #     for d in range(1, 4):
    #         for m in range(1, 10):
    #             for iter in [0, 1, 5]:
    #                 result = analyze_circuit(d=d, m=m, bc=bc, iterations=iter)
    #                 all_results_grid_scaling.append(result)

    # # ✅ Save everything
    # save_results_to_csv(all_results_grid_scaling, "grid_scaling_results.csv")
    # save_results_to_txt(all_results_grid_scaling, "grid_scaling_results.txt")

    all_results_time_scaling = []

    for bc in [
        BoundaryCondition.PERIODIC,
        BoundaryCondition.BOUNCE_BACK,
        BoundaryCondition.SIMPLE_SLIP,
    ]:
        print(f"\nBoundary Condition: {bc.name}")
        print("#" * 50)

        for d in range(1, 4):
            for i in range(0, 7):
                result = analyze_circuit(d, 5, bc, 2**i)
                all_results_time_scaling.append(result)

    # ✅ Save everything
    save_results_to_csv(all_results_time_scaling, "time_scaling_results.csv")
    save_results_to_txt(all_results_time_scaling, "time_scaling_results.txt")
