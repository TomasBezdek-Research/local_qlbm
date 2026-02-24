from typing import Dict

import numpy as np
from qrisp import (
    IQAE,
    amplitude_amplification,
    auto_uncompute,
    control,
    mcx,
    z,
)
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_carry_path import (
    QuantumBool,
)

from ._engine import LBMEngine
from ._state import LBMState
from .types import RegionOfInterest


class LBMAnalyzer:
    """
    Class representing a quantum LBM analyzer.

    Parameters
    ----------
    lbm_state : LBMState
        The state of the LBM simulation.
    lbm_engine : LBMEngine
        The engine used to perform the LBM simulation.
    """

    def __init__(
        self,
        lbm_state: LBMState,
        lbm_engine: LBMEngine,
    ):
        self._state = lbm_state
        self._engine = lbm_engine

    ########################
    #   Complex Workflow   #
    ########################
    def analyze_region(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        eps: float = 0.01,
        shots: int = 1000,
        backend=None,
    ) -> Dict:
        """
        1. Estimates total density (f) in ROI using IQAE.
        2. Amplifies ROI to make internal distribution measurable.
        3. Measures cells and rescales so their sum equals f.

        Args:
            roi: RegionOfInterest object.
            simulation_iterations: Number of iterations to run the simulation.
            eps: The desired accuracy of the estimate.
            shots: Number of shots to measure.

        Returns:
            A dict with the following keys:
                total_density: The total density of the region.
                cells: A dict of the cells in the region.
        """
        # 1. Encode ROI for internal ghost-layer logic
        encoded_roi = self._state.encode_roi(roi)

        # 2. Get the high-precision "Mass" of the region
        print("Phase 1: Estimating total density (f)...")
        f = self.estimate_region_density(
            roi=encoded_roi,
            simulation_iterations=simulation_iterations,
            eps=eps,
            shots=shots,
            backend=backend,
        )

        if f <= 0:
            print("Density is zero; skipping amplification.")
            return {"total_density": 0, "cells": {}}

        # 3. Amplify so the shots actually land in the ROI
        print(f"Phase 2: Amplifying region (f={f:.4f})...")
        self.boost_region(
            region_density=f,
            roi=encoded_roi,
            simulation_iterations=simulation_iterations,
        )

        # 4. Measure physical density
        # Note: This dict only contains cells that were actually measured (shots > 0)
        counts = self._state.get_physical_density(shots=shots, backend=backend)

        # 5. Filter counts to only include the ROI
        # (in case some probability leaked or noise occurred)
        roi_counts = {}
        total_measured_prob = 0

        for coords, prob in counts.items():
            if roi.contains(coords):
                roi_counts[coords] = prob
                total_measured_prob += prob

        # 6. Rescale: (Measured Cell Prob / Total Measured Prob) * f
        # This makes the sum of roi_rescaled exactly equal to f.
        roi_rescaled = {}
        if total_measured_prob > 0:
            scale_factor = f / total_measured_prob
            for coords, prob in roi_counts.items():
                roi_rescaled[coords] = prob * scale_factor

        return {"total_density": f, "cells": roi_rescaled}

    #############################
    #   Measurement Functions   #
    #############################
    def estimate_region_density(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        eps: float = 0.01,
        alpha: float = 0.05,
        shots=None,
        backend=None,
    ):
        """
        Uses IQAE to estimate the probability mass (density) within
        the region defined by 'roi'.

        Args:
            roi: RegionOfInterest object.
            simulation_iterations: Number of iterations to run the simulation.
            eps: The desired accuracy of the estimate.
            alpha: The desired confidence level of the estimate.
            shots: Number of shots to use for the simulation.
            backend: Backend to use for the simulation.

        Returns:
            Estimated density of the region.
        """

        # 1. Create an init function
        def qargs():
            pos, sign, vel, anc = self._state.get_clean_registers()
            self._engine.apply_initialization_external_registers(
                position_register=pos,
                sign_register=sign,
                velocity_register=vel,
                boundary_ancillas=anc,
            )
            target_qubit = QuantumBool()
            # Use * to flatten the lists into one long tuple
            return (*pos, *sign, *vel, *anc, target_qubit)

        # 2. Define the IQAE state preparation wrapper
        def iqae_wrapper(*flat_args):
            d = self._state.dimensions
            pos = list(flat_args[0:d])
            sign = list(flat_args[d : 2 * d])
            vel = list(flat_args[2 * d : 3 * d])
            anc = list(flat_args[3 * d : -1])
            target = flat_args[-1]

            # Step A: Run your simulation logic (Initialization + Streaming + BC)
            self._engine.run_simulation_external_registers(
                position_register=pos,
                velocity_register=vel,
                sign_register=sign,
                boundary_ancillas=anc,
                iterations=simulation_iterations,
                init=False,
            )

            # Step B: Apply the oracle logic
            # We compute the condition and XOR it onto the target qubit.

            # region_oracle = self._get_region_oracle(roi=roi)
            # target ^= region_oracle
            self._get_region_oracle(roi=roi, target_qubit=target)

        # 3. Execute IQAE
        # This returns a float representing the density in the subgrid
        mes_kwargs = {"backend": backend}
        if shots is not None:
            mes_kwargs["shots"] = shots
        return IQAE(
            qargs,
            iqae_wrapper,
            eps=eps,
            alpha=alpha,
            mes_kwargs=mes_kwargs,
        )

    def boost_region(
        self, region_density, roi: RegionOfInterest, simulation_iterations: int = 1
    ) -> None:
        """
        Amplifies the density in a specific region of the grid.

        Args:
            region_density: The initial probability of the region.
            roi: RegionOfInterest object.
            simulation_iterations: Number of iterations to run the simulation.
        """
        if region_density <= 0:
            print("Density is 0, nothing to amplify.")
            return

        # 2. Calculate optimal iterations m
        m = self.get_optimal_m(region_density)

        # 3. Handle the 'No Boost Needed' case
        if m == 0:
            if region_density >= 0.5:
                print(
                    f"Density is already high ({region_density:.4f}). Skipping amplification to avoid over-rotation."
                )
            else:
                print(
                    f"Density ({region_density:.4f}) is too low to amplify effectively with integer steps."
                )
            return

        print(f"Current density: {region_density:.4f}. Amplifying with {m} iterations.")

        # 4. Perform the amplification
        # Note: ensure simulation_iterations matches the state that created region_density!
        self.amplify_region(
            roi=roi,
            amplification_iterations=m,
            simulation_iterations=simulation_iterations,
        )

    def amplify_region(
        self,
        roi: RegionOfInterest,
        amplification_iterations: int = 1,
        simulation_iterations: int = 1,
    ) -> None:
        """
        Amplifies the density in a specific region of the grid.

        Args:
            roi: RegionOfInterest object.
            amplification_iterations: Number of iterations to amplify.
            simulation_iterations: Number of iterations to run the simulation.
        """
        # args tells AA which registers are being modified
        args = self._state.positions_register

        # A operator: Prepares the state (Physics)
        def state_function(*pos_regs):
            self._engine.run_simulation(iterations=simulation_iterations)

        # O operator: Tags the state (Logic)
        def oracle_function(*pos_regs):
            # We re-evaluate the oracle based on the NEW positions
            # created by state_function
            condition = QuantumBool()
            # condition = self._get_region_oracle(roi=roi)
            self._get_region_oracle(roi=roi, target_qubit=condition)

            # Apply the phase flip required for AA
            z(condition)
            # Because of @auto_uncompute on the oracle methods,
            # 'condition' is cleaned up perfectly here.

        # Perform the amplification
        amplitude_amplification(
            args=args,
            state_function=state_function,
            oracle_function=oracle_function,
            iter=amplification_iterations,
        )

    def fast_region_density_estimate(
        self, roi: RegionOfInterest, shots=None, backend=None
    ):
        """
        Estimates the density of a region using the fast method.
        """
        state_roi = self._state.encode_roi(roi)
        in_region = QuantumBool()
        self._get_region_oracle(roi=state_roi, target_qubit=in_region)
        return in_region.get_measurement(shots=shots, backend=backend)

    def _get_region_oracle(self, roi: RegionOfInterest, target_qubit: QuantumBool):
        """
        Returns the oracle for the specified region.

        Args:
            roi: RegionOfInterest object.

        Returns:
            A QuantumBool representing the region.
        """
        if roi.is_subgrid:
            return self._get_subgrid_oracle(roi.bounds, target_qubit)
        return self._get_multi_point_oracle(roi.points_list, target_qubit)

    def _get_subgrid_oracle(self, bounds, target_qubit):
        """
        Returns the oracle for the specified subgrid.

        Args:
            bounds: List of tuples [(min_d0, max_d0), (min_d1, max_d1), ...]
            Example for 2D: [(2, 5), (1, 3)]

        Returns:
            A QuantumBool representing the subgrid.
        """
        in_subrid_ancillas = [QuantumBool() for _ in range(self._state.dimensions)]

        for d in range(self._state.dimensions):
            with self._state.positions_register[d] < bounds[d][0]:
                in_subrid_ancillas[d].flip()
            with self._state.positions_register[d] >= bounds[d][1]:
                in_subrid_ancillas[d].flip()
            in_subrid_ancillas[d].flip()

        # Use the control environment. Qrisp handles the temporary bits
        # efficiently here without creating the DAG cycle.
        mcx(in_subrid_ancillas, target_qubit, method="gray")
        # for d in range(self._state.dimensions):
        # in_subrid_ancillas[d].uncompute()

    @auto_uncompute
    def _get_multi_point_oracle(self, points_list, target_qubit):
        """
        Returns the oracle for the specified region.

        Args:
            points_list: List of coordinate lists [[x1, y1,...], [x2, y2,...], ...]
            Example for 2D: [[2, 3], [5, 7]]

        Returns:
            A QuantumBool representing the region.
        """

        def check_single_point(coords):
            res = self._state.positions_register[0] == coords[0]
            for d in range(1, self._state.dimensions):
                res &= self._state.positions_register[d] == coords[d]
            return res

        # Combine all points with OR logic
        for point in points_list:
            target_qubit |= check_single_point(point)

    ##################
    #   Utilities    #
    ##################
    @staticmethod
    def get_optimal_m(a):
        """
        Calculates the optimal number of iterations for amplification.

        Args:
            a: The initial probability of the region.

        Returns:
            The optimal number of iterations.
        """
        # If the initial probability is already low,
        if a >= 1.0:
            return 0
        if a <= 0.0:
            return 0

        # If the initial probability is already high,
        if a >= 0.5:
            return 0

        theta = np.arcsin(np.sqrt(a))
        # Using floor ensures we don't rotate PAST the 100% mark
        m = int(np.floor(np.pi / (4 * theta)))

        return m

    @staticmethod
    def get_optimal_m_rounded(a):
        """
        Calculates the optimal number of iterations for amplification.

        Args:
            a: The initial probability of the region.

        Returns:
            The optimal number of iterations.
        """
        if a >= 0.5 or a <= 0.0:
            return 0
        theta = np.arcsin(np.sqrt(a))
        # This centers the rotation around the 90-degree (pi/2) target
        return max(0, int(np.round(np.pi / (4 * theta) - 0.5)))
