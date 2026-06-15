from __future__ import annotations

import numpy as np
from qrisp import (
    IQAE,
    QuantumBool,
    QuantumModulus,
    amplitude_amplification,
    auto_uncompute,
    mcx,
    z,
)

from local_qlbm.api.measurement import measure_density
from local_qlbm.config.model import LBMConfig
from local_qlbm.config.roi import RegionOfInterest
from local_qlbm.kernels.engine_jasp import (
    initialize_and_run,
    run_simulation,
)
from local_qlbm.kernels.initialization import initialize_state
from local_qlbm.kernels.registers import allocate_registers


class LBMAnalyzer:
    def __init__(self, config: LBMConfig) -> None:
        self._config = config

    # ======================================================
    # FULL WORKFLOW (same logic as original)
    # ======================================================

    def analyze_region(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        eps: float = 0.01,
        alpha: float = 0.05,
        shots: int | None = None,
        backend: object | None = None,
    ) -> dict:
        """
        Analyzes the density of a region of interest in the LBM grid.

        Args:
            roi (RegionOfInterest): The region of interest to analyze.
            simulation_iterations (int, optional): Number of simulation iterations. Defaults to 1.
            eps (float, optional): Tolerance for IQAE estimate. Defaults to 0.01.
            alpha (float, optional): Significance level for IQAE estimate. Defaults to 0.05.
            shots (int | None, optional): Number of shots for quantum measurement. Defaults to None.
            backend (object | None, optional): Backend for quantum simulation. Defaults to None.

        Returns:
            dict: A dictionary containing the total density and cell measurements.
        """

        # 1. IQAE estimate
        print("Phase 1: Estimating total density (f)...")
        f = self.estimate_region_density(
            roi=roi,
            simulation_iterations=simulation_iterations,
            eps=eps,
            alpha=alpha,
            shots=shots,
            backend=backend,
        )

        if f <= 0:
            print("Density is zero; skipping amplification.")
            return {"total_density": 0, "cells": {}}

        # 2. Amplification
        print(f"Phase 2: Amplifying region (f={f:.4f})...")
        res = self.boost_region(
            region_density=f,
            roi=roi,
            simulation_iterations=simulation_iterations,
        )

        if res is None:
            return {"total_density": 0.0, "cells": {}}
        pos, sign, vel = res

        # 3. Measurement AFTER amplification
        counts = measure_density(
            config=self._config, position_register=pos, shots=shots, backend=backend
        )

        roi_counts: dict[tuple[int, ...], float] = {}
        total_measured_prob = 0.0

        for coords, prob in counts.items():
            if roi.contains(coords):
                roi_counts[coords] = roi_counts.get(coords, 0.0) + prob
                total_measured_prob += prob

        roi_rescaled: dict[tuple[int, ...], float] = {}

        if total_measured_prob > 0:
            scale = f / total_measured_prob
            for coords, prob in roi_counts.items():
                roi_rescaled[coords] = prob * scale

        return {
            "total_density": f,
            "cells": roi_rescaled,
            "pure_amplified_density": counts,
        }

    # ======================================================
    # IQAE
    # ======================================================

    def estimate_region_density(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        eps: float = 0.01,
        alpha: float = 0.05,
        shots: int | None = None,
        backend: object | None = None,
    ) -> float:
        """
        Estimates the density of a region of interest in the LBM grid using IQAE.

        Args:
            roi (RegionOfInterest): The region of interest to estimate.
            simulation_iterations (int, optional): Number of simulation iterations. Defaults to 1.
            eps (float, optional): Tolerance for IQAE estimate. Defaults to 0.01.
            alpha (float, optional): Significance level for IQAE estimate. Defaults to 0.05.
            shots (int | None, optional): Number of shots for quantum measurement. Defaults to None.
            backend (object | None, optional): Backend for quantum simulation. Defaults to None.

        Returns:
            float: The estimated density of the region of interest.
        """

        encoded_roi = self._config.encode_roi(roi)

        def qargs():
            pos, sign, vel, anc, oracle = allocate_registers(
                self._config,
                create_oracle_ancillas=True,
            )
            target = QuantumBool()
            return (*pos, *sign, *vel, *anc, *oracle, target)

        def wrapper(*flat_args):
            d = self._config.dimensions

            pos = list(flat_args[0:d])
            sign = list(flat_args[d : 2 * d])
            vel = list(flat_args[2 * d : 3 * d])
            anc = list(flat_args[3 * d : 4 * d])
            oracle_ancillas = list(flat_args[-1 - d : -1])
            target = flat_args[-1]

            initialize_state(self._config, pos, vel, sign)

            run_simulation(
                self._config,
                pos,
                vel,
                sign,
                anc,
                iterations=simulation_iterations,
            )

            self._get_region_oracle(
                roi=encoded_roi,
                position_register=pos,
                oracle_ancillas=oracle_ancillas,
                target_qubit=target,
            )

        mes_kwargs: dict = {"backend": backend}
        if shots is not None:
            mes_kwargs["shots"] = shots

        return float(
            IQAE(
                qargs,
                wrapper,
                eps=eps,
                alpha=alpha,
                mes_kwargs=mes_kwargs,
            )
        )

    # ======================================================
    # AMPLIFICATION
    # ======================================================

    def boost_region(
        self,
        region_density: float,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
    ) -> (
        tuple[
            list,
            list,
            list,
        ]
        | None
    ):
        """
        Boosts the density of a region of interest in the LBM grid.

        Args:
            region_density (float): The density of the region to boost.
            roi (RegionOfInterest): The region of interest to boost.
            simulation_iterations (int, optional): Number of simulation iterations. Defaults to 1.

        Returns:
            tuple[list, list, list]: The amplified region as a tuple of position, sign, and velocity registers.
        """
        if region_density <= 0:
            return

        m = self.get_optimal_m(region_density)

        if m == 0:
            return

        return self.amplify_region(
            roi=roi,
            amplification_iterations=m,
            simulation_iterations=simulation_iterations,
        )

    def amplify_region(
        self,
        roi: RegionOfInterest,
        amplification_iterations: int = 1,
        simulation_iterations: int = 1,
    ) -> tuple[
        list,
        list,
        list,
    ]:
        """
        Amplifies the density of a region of interest in the LBM grid.

        Args:
            roi (RegionOfInterest): The region of interest to amplify.
            amplification_iterations (int, optional): Number of amplification iterations. Defaults to 1.
            simulation_iterations (int, optional): Number of simulation iterations. Defaults to 1.

        Returns:
            tuple[
                list[QuantumModulus]: The position quantum modulus.
                list[QuantumBool]: The sign quantum bool.
                list[QuantumModulus]: The velocity quantum modulus.
                list[QuantumBool]: The ancilla quantum bool.
                QuantumBool: The target quantum bool.
            ]: The amplified state and oracle ancillas.
        """
        encoded_roi = self._config.encode_roi(roi)

        def qargs():
            pos, sign, vel, anc, oracle = allocate_registers(
                self._config,
                create_oracle_ancillas=True,
            )
            target = QuantumBool()
            return (*pos, *sign, *vel, *anc, *oracle, target)

        def state_fn(*flat_args):
            d = self._config.dimensions

            pos = list(flat_args[0:d])
            sign = list(flat_args[d : 2 * d])
            vel = list(flat_args[2 * d : 3 * d])
            anc = list(flat_args[3 * d : 4 * d])

            initialize_state(self._config, pos, vel, sign)

            run_simulation(
                self._config,
                pos,
                vel,
                sign,
                anc,
                iterations=simulation_iterations,
            )

        def oracle_fn(*flat_args):
            d = self._config.dimensions
            pos = list(flat_args[0:d])
            oracle_ancillas = list(flat_args[-1 - d : -1])
            target = flat_args[-1]

            self._get_region_oracle(encoded_roi, pos, oracle_ancillas, target)
            z(target)
            self._get_region_oracle(encoded_roi, pos, oracle_ancillas, target)

        args = qargs()

        state_fn(*args)

        amplitude_amplification(
            args=args,
            state_function=state_fn,
            oracle_function=oracle_fn,
            iter=amplification_iterations,
        )

        pos = list(args[0 : self._config.dimensions])
        sign = list(args[self._config.dimensions : 2 * self._config.dimensions])
        vel = list(args[2 * self._config.dimensions : 3 * self._config.dimensions])
        return pos, sign, vel

    # ======================================================
    # FAST ESTIMATE
    # ======================================================

    def fast_region_density_estimate(
        self,
        roi: RegionOfInterest,
        simulation_iterations: int = 1,
        shots: int | None = None,
        backend: object | None = None,
    ) -> dict:
        """
        Estimates the density of a region of interest in the LBM grid using a fast method.

        Args:
            roi (RegionOfInterest): The region of interest to estimate.
            simulation_iterations (int, optional): Number of simulation iterations. Defaults to 1.
            shots (int | None, optional): Number of shots for quantum measurement. Defaults to None.
            backend (object | None, optional): Backend for quantum simulation. Defaults to None.

        Returns:
            dict: A dictionary containing the measurement results.
        """

        pos, sign, vel, anc, oracle = initialize_and_run(
            self._config,
            iterations=simulation_iterations,
            create_oracle_ancillas=True,
        )

        in_region = QuantumBool()

        self._get_region_oracle(
            roi=self._config.encode_roi(roi),
            position_register=pos,
            oracle_ancillas=oracle,
            target_qubit=in_region,
        )

        return in_region.get_measurement(shots=shots, backend=backend)

    # ======================================================
    # ORACLE
    # ======================================================

    def _get_region_oracle(
        self,
        roi: RegionOfInterest,
        position_register: list[QuantumModulus],
        oracle_ancillas: list[QuantumBool],
        target_qubit: QuantumBool,
    ) -> None:
        """
        Constructs the oracle for the region of interest.

        Args:
            roi (RegionOfInterest): The region of interest to construct the oracle for.
            position_register: The position register to use for the oracle.
            target_qubit (QuantumBool): The target qubit to use for the oracle.
        """
        if roi.is_subgrid:
            self._get_subgrid_oracle(
                roi.bounds, position_register, oracle_ancillas, target_qubit
            )
        elif roi.is_point_cloud:
            self._get_point_cloud_oracle(
                roi.points_set, position_register, oracle_ancillas, target_qubit
            )
        return

    def _get_subgrid_oracle(
        self,
        bounds: list[tuple[int, int]],
        position_register: list[QuantumModulus],
        oracle_ancillas: list[QuantumBool],
        target_qubit: QuantumBool,
    ) -> None:
        """
        Constructs the oracle for the subgrid region of interest.

        Args:
            bounds (list[tuple[int, int]]): The bounds of the subgrid.
            position_register: The position register to use for the oracle.
            target_qubit (QuantumBool): The target qubit to use for the oracle.
        """

        for d in range(self._config.dimensions):
            with position_register[d] < bounds[d][0]:
                oracle_ancillas[d].flip()
            with position_register[d] >= bounds[d][1]:
                oracle_ancillas[d].flip()
            oracle_ancillas[d].flip()

        mcx(oracle_ancillas, target_qubit, method="gray")

        for d in range(self._config.dimensions):
            oracle_ancillas[d].flip()
            with position_register[d] < bounds[d][0]:
                oracle_ancillas[d].flip()
            with position_register[d] >= bounds[d][1]:
                oracle_ancillas[d].flip()

    @auto_uncompute
    def _get_point_cloud_oracle(
        self,
        points_set: set[tuple[int, ...]],
        position_register: list[QuantumModulus],
        oracle_ancillas: list[QuantumBool],
        target_qubit: QuantumBool,
    ) -> None:
        """
        Returns the oracle for the specified region.

        Args:
            points_list: List of coordinate lists [[x1, y1,...], [x2, y2,...], ...]
            Example for 2D: [[2, 3], [5, 7]]

        Returns:
            A QuantumBool representing the region.
        """

        def check_single_point(coords: tuple[int, ...]):
            for d in range(self._config.dimensions):
                with position_register[d] == coords[d]:
                    oracle_ancillas[d].flip()

            mcx(oracle_ancillas, target_qubit, method="gray")

            for d in range(self._config.dimensions):
                with position_register[d] == coords[d]:
                    oracle_ancillas[d].flip()

        for point in points_set:
            check_single_point(point)

    # ======================================================
    # UTILITIES (UNCHANGED)
    # ======================================================

    @staticmethod
    def get_optimal_m(a: float) -> int:
        """
        Computes the optimal number of ancillas for a given probability `a`.

        Args:
            a (float): The probability value.

        Returns:
            int: The optimal number of ancillas.
        """

        if a <= 0 or a >= 1.0:
            return 0
        if a >= 0.5:
            return 0

        theta = np.arcsin(np.sqrt(a))
        return int(np.floor(np.pi / (4 * theta)))
