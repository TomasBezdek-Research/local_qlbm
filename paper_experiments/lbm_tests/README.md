# LBM Experiments

This folder contains experiments for analyzing the scaling behavior of the quantum LBM implementation and verifying the implemented boundary conditions.

## Grid-Size Scaling

Run:

```bash
uv run exponentia_memory_advantage.py
```

The experiment evaluates compiled quantum circuits for:

- dimensions `d = 1, 2, 3`
- grid sizes defined by `N = 2^m`
- periodic, bounce-back, and simple-slip boundary conditions
- different numbers of simulation iterations

It records circuit metrics including the number of qubits, circuit depth, total gate count, CNOT count, and individual gate counts.

Results are written to:

```text
grid_scaling_results.csv
grid_scaling_results.txt
```

## Time-Step Scaling

Run:

```bash
uv run exponentia_memory_advantage_time.py
```

This experiment keeps the grid resolution fixed and varies the number of simulation iterations to study how circuit size and depth scale with simulation time.

Results are written to:

```text
time_scaling_results.csv
time_scaling_results.txt
```

## Boundary-Condition Tests

Run:

```bash
uv run bc_test.py
```

The script runs small 2D simulations for:

- periodic boundaries
- bounce-back boundaries
- simple-slip boundaries

The simulation states are exported as VTK files to:

```text
output/bc_tests/
```

The VTK files can be inspected with a compatible visualization tool such as ParaView.

## Processing the Scaling Results

The generated scaling CSV files can be processed and visualized with:

```text
scaling_processing.ipynb
```

Run the notebook after generating the grid-size and time-step scaling results to reproduce the corresponding scaling plots and analysis.
