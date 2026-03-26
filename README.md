# Local Quantum Lattice Boltzmann (QLBM) Solver

A specialized Python implementation of a **Local Quantum Lattice Boltzmann** solver. This project provides a framework for simulating Pure Advection Equation by QLBM based on LBM.
A central feature of the solver is the capability to perform **localized density measurements**, allowing for the extraction of spatial distributions from the quantum state.

The implementation is built using [Qrisp](https://www.qrisp.eu/), a high-level programming framework for partitioning and compiling quantum algorithms.

---

## Getting Started

### Prerequisites
* **Python 3.11** or higher.
* Recommended: A modern package manager like `uv` or `conda`.

### Installation

Clone the repository to your local machine:

**Using HTTPS:**
```bash
git clone https://github.com/TomasBezdek-Research/local_qlbm.git
```

**Using SSH:**
```bash
git clone git@github.com:TomasBezdek-Research/local_qlbm.git
```

---

## Environment Setup
The project supports multiple environment managers. Choose the method that best aligns with your research workflow.

### Using **uv**
**uv** (https://docs.astral.sh/uv/) is a high-performance Python package manager. 
To set up the development environment and install all dependencies (including the local-qlbm package in editable mode), execute the following in the project root:
```bash
# Create the environment and synchronize it with pyproject.toml
uv sync
```

### Using **Conda** (Anaconda, Miniconda)
To create a dedicated environment named `qlbm_env` from the provided configuration:
```bash
# Create the environment
conda env create -f environment.yml
```
Set up the environment to your IDE or activate the conda environment in terminal by:
```bash
# Activate the environment
conda activate qlbm_env
```
In terminal with the activated environment, you can install the project in editable mode (install local package `local-qlbm` and all the dependencies):
```bash
# Install the project and dependencies in editable mode
pip install -e .
```

### Standard Virtual Environments (venv)
Create virtual environment by manager by your choise, for example.
```bash
# Create the virtual environment
python -m venv .venv
```
Activate the environment
```bash
# Activate the environment (Linux/macOS)
source .venv/bin/activate
```
```bash
# Or (Windows)
.venv\Scripts\activate
```
Than in activated environment, run 
```bash
# Install in editable mode
pip install -e .
```
to install local package `local-qlbm` in editable mode and also all the required dependencies.

---

## Project Overview
This solver implements a quantum-algebraic approach to the Lattice Boltzmann Method. 
By leveraging the Qrisp framework, the project maps the streaming step and boundary handling of the LBM onto quantum circuits to solve the Pure Advection Equation.

### Key Features
* **Quantum Advection Simulation:** Specialized for solving the Pure Advection Equation via QLBM.
* **Localized Density Measurement:** A core capability of the solver is the implementation of localized measurement operators, enabling the extraction of spatial density distributions during the simulation.
* **Qrisp Integration:** Utilizes high-level quantum functional programming for circuit synthesis and execution.
* **Editable Installation:** All setup methods install the package in editable mode (-e), ensuring source code modifications are reflected immediately.
* **Modular Research Framework:** Designed for the integration of custom quantum operators and diverse lattice geometries.
