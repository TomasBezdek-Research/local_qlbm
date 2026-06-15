import numpy as np
from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumModulus,
    QuantumVariable,
    auto_uncompute,
    mcx,
    prepare,
)

from ._state import BCCombination, LBMState
from .types import BoundaryCondition


class LBMEngine:
    """
    Class representing a quantum LBM engine.

    Parameters
    ----------
    lbm_state : LBMState
        The state of the quantum LBM engine.
    """

    def __init__(
        self,
        lbm_state: LBMState,
    ):
        self._state = lbm_state

    ################################
    #   Initialization Functions   #
    ################################
    def apply_initialization_external_registers(
        self,
        position_register: list,
        velocity_register: list,
        sign_register: list,
        boundary_ancillas: list,
        oracle_ancillas: list = [],
    ) -> None:
        """
        Applies the gates to set the registers to t=0.
        This is called internally by IQAE/AA.
        """
        self._state.set_registers(
            position_register,
            sign_register,
            velocity_register,
            boundary_ancillas,
            oracle_ancillas,
        )
        self.apply_initialization()

    def apply_initialization(self) -> None:
        """
        Applies the gates to set the registers to t=0.
        This is called internally by IQAE/AA.
        """
        if self._state.initial_bubble.is_subgrid:
            self._init_subgrid()
        elif self._state.initial_bubble.is_point_cloud:
            self._init_point_cloud()
        else:
            raise ValueError(
                "Initial bubble must be a subgrid or point cloud. Use set_initial_state() to set the initial state."
            )

        # Prepare Velocity
        self._set_global_velocity(self._state.initial_velocity)

    def _init_subgrid(self) -> None:
        # Prepare Positions
        encoded_bounds = self._state.get_encoded_bounds(
            self._state.initial_bubble.bounds
        )
        for d in range(self._state.dimensions):
            num_states = 2 ** self._state.positions_register[d].size
            statevector = np.zeros(num_states)
            start, end = encoded_bounds[d]
            statevector[start:end] = 1.0
            statevector /= np.linalg.norm(statevector)
            prepare(self._state.positions_register[d], statevector)

    def _init_point_cloud(self) -> None:
        raise NotImplementedError(
            "Initial bubble as point cloud is not implemented yet."
        )
        points = self._state.get_encoded_points(self._state.initial_bubble.points_set)

        dims_sizes = [reg.size for reg in self._state.positions_register]
        total_states = 2 ** sum(dims_sizes)

        full_statevector = np.zeros(total_states)

        for pt in points:
            flat_index = 0
            current_bit_shift = 0

            for coord, size in zip(pt, dims_sizes):
                flat_index += int(coord) << current_bit_shift
                current_bit_shift += size
            full_statevector[flat_index] = 1.0

        full_statevector /= np.linalg.norm(full_statevector)

        master_session = self._state.positions_register[0].qs
        r = []
        for reg in self._state.positions_register:
            r += reg[:]
        qv = QuantumVariable(len(r), qs=master_session)
        qv.reg = r

        prepare(
            qv,
            full_statevector,
        )

    def _set_global_velocity(
        self, velocity_vector: np.typing.NDArray[np.int_] | list[int]
    ) -> None:
        """
        Sets the velocity u for the transport equation.
        velocity_vector: list of floats/ints like [1, -1]

        Args:
            velocity_vector: The velocity vector.
        """
        for d in range(self._state.dimensions):
            u_val = velocity_vector[d]

            num_states = 2 ** self._state.velocity_register[d].size
            statevector = np.zeros(num_states)
            statevector[abs(u_val) : abs(u_val) + 1] = 1.0
            prepare(self._state.velocity_register[d], statevector)
            # Encode magnitude
            # self._state.velocity_register[d].encode(abs(u_val))

            # Encode sign: 1 for positive, 0 for negative
            # (Note: In your logic, True/1 is positive, False/0 is negative)
            if u_val >= 0:
                prepare(self._state.sign_register[d], np.array([0, 1]))
                # self._state.sign_register[d].encode(True)
            else:
                prepare(self._state.sign_register[d], np.array([1, 0]))
                # self._state.sign_register[d].encode(False)

    ###############################
    #   Simulation Functions    #
    ###############################
    def run_simulation(self, iterations: int = 1, init: bool = True) -> None:
        """
        Run the simulation for a given number of iterations.

        Args:
            iterations: The number of iterations to run.
            init: Whether to apply the initial conditions before running the simulation.
        """
        if init:
            self.apply_initialization()
        for _ in range(iterations):
            self.stream()
            self.apply_boundary_conditions()

    def run_simulation_external_registers(
        self,
        position_register: list[QuantumModulus],
        velocity_register: list[QuantumFloat],
        sign_register: list[QuantumBool],
        boundary_ancillas: list[QuantumBool],
        oracle_ancillas: list[QuantumBool] = [],
        iterations: int = 1,
        init: bool = False,
    ) -> None:
        """
        Run the simulation for a given number of iterations.

        Args:
            iterations: The number of iterations to run.
            init: Whether to apply the initial conditions before running the simulation.
        """
        self._state.set_registers(
            position_register,
            velocity_register,
            sign_register,
            boundary_ancillas,
            oracle_ancillas,
        )
        if init:
            self.apply_initialization()

        for _ in range(iterations):
            self.stream()
            self.apply_boundary_conditions()

    def stream(self) -> None:
        """
        Do one streaming step.
        """
        for d in range(self._state.dimensions):
            with self._state.sign_register[d]:
                self._state.positions_register[d] += self._state.velocity_register[d]

            with ~self._state.sign_register[d]:
                self._state.positions_register[d] -= self._state.velocity_register[d]

    @auto_uncompute
    def apply_boundary_conditions(self) -> None:
        """
        Aplly the boundary conditions to the positions.
        """
        if self._state.bc_combination == BCCombination.PERIODIC:
            return

        if self._state.bc_combination == BCCombination.SLIP:
            self._apply_bc_slip()
            return

        if self._state.bc_combination == BCCombination.BB:
            self._apply_bc_bb()
            return

        if self._state.bc_combination == BCCombination.BB_AND_SLIP:
            self._apply_bc_bb_slip()
            return

    @auto_uncompute
    def _apply_bc_slip(self) -> None:
        """
        Apply the boundary conditions to the positions.
        Used when if some BC is Slip and rest Slip or Periodic.
        """
        for d in range(self._state.dimensions):
            if self._state.boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue

            self._is_out_boundary(d)
            with self._state.boundary_ancillas[self._state.boundary_ancilla_indexes[d]]:
                # Flip the sign register if the position is out of bounds
                self._state.sign_register[d].flip()
                # Reflect the position (streaming correction step)
                with self._state.sign_register[d]:
                    self._state.positions_register[d] += self._state.velocity_register[
                        d
                    ]
                with ~self._state.sign_register[d]:
                    self._state.positions_register[d] -= self._state.velocity_register[
                        d
                    ]
            self._comes_from_boundary(d)

    @auto_uncompute
    def _apply_bc_bb(self) -> None:
        """
        Apply the boundary conditions to the positions.
        Used when if some BC is Bounce Back and rest Bounce Back or Periodic.
        """
        for d in range(self._state.dimensions):
            if self._state.boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue
            self._is_out_boundary(d)

        apply_bb = QuantumBool()
        apply_bb.flip()
        mcx(
            [anc for anc in self._state.boundary_ancillas],
            apply_bb,
            method="gray",
            ctrl_state=0,
        )
        with apply_bb:
            # if False:
            for d in range(self._state.dimensions):
                # Flip the sign register if the position is out of bounds
                self._state.sign_register[d].flip()
                # Reflect the position (streaming correction step)
                with self._state.sign_register[d]:
                    self._state.positions_register[d] += self._state.velocity_register[
                        d
                    ]
                with ~self._state.sign_register[d]:
                    self._state.positions_register[d] -= self._state.velocity_register[
                        d
                    ]
        apply_bb.uncompute()

        for d in range(self._state.dimensions):
            if self._state.boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue
            self._comes_from_boundary(d)

    @auto_uncompute
    def _apply_bc_bb_slip(self) -> None:
        """
        Apply the boundary conditions to the positions.
        Used when if some BC is Bounce Back, some Slip and rest Bounce Back, Slip or Periodic.
        """
        for d in range(self._state.dimensions):
            if self._state.boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue
            self._is_out_boundary(d)

        apply_bb = QuantumBool()
        apply_bb.flip()
        mcx(
            [
                self._state.boundary_ancillas[idx]
                for idx in self._state.bb_boundary_ancilla_indexes.values()
            ],
            apply_bb,
            method="gray",
            ctrl_state=0,
        )
        for d in range(self._state.dimensions):
            if (
                self._state.boundary_conditions[d] == BoundaryCondition.PERIODIC
                or self._state.boundary_conditions[d] == BoundaryCondition.BOUNCE_BACK
            ):
                with apply_bb:
                    # Flip the sign register if the position is out of bounds
                    self._state.sign_register[d].flip()
                    # Reflect the position (streaming correction step)
                    with self._state.sign_register[d]:
                        self._state.positions_register[d] += (
                            self._state.velocity_register[d]
                        )
                    with ~self._state.sign_register[d]:
                        self._state.positions_register[d] -= (
                            self._state.velocity_register[d]
                        )

            if self._state.boundary_conditions[d] == BoundaryCondition.SIMPLE_SLIP:
                apply_slip = QuantumBool()
                apply_slip.flip()
                mcx(
                    [
                        apply_bb,
                        self._state.boundary_ancillas[
                            self._state.boundary_ancilla_indexes[d]
                        ],
                    ],
                    apply_slip,
                    method="gray",
                    ctrl_state=0,
                )
                with apply_slip:
                    # Flip the sign register if the position is out of bounds
                    self._state.sign_register[d].flip()
                    # Reflect the position (streaming correction step)
                    with self._state.sign_register[d]:
                        self._state.positions_register[d] += (
                            self._state.velocity_register[d]
                        )
                    with ~self._state.sign_register[d]:
                        self._state.positions_register[d] -= (
                            self._state.velocity_register[d]
                        )
                apply_slip.uncompute()
        apply_bb.uncompute()

        for d in range(self._state.dimensions):
            if self._state.boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue
            self._comes_from_boundary(d)

    #########################
    #   Boundary Ancillas   #
    #########################

    @auto_uncompute
    def _is_out_boundary(self, dim: int) -> None:
        """
        Flips the boundary ancilla for the given dimension if the position is out of bounds.

        Args:
            dim: The dimension to check.
        """
        if dim not in self._state.boundary_ancilla_indexes:
            return

        if self._state.boundary_offsets[dim] > 1:
            self._is_out_boundary_multi_layer(dim)
        else:
            self._is_out_boundary_single_layer(dim)

    @auto_uncompute
    def _is_out_boundary_single_layer(self, dim: int) -> None:
        """
        Flips the boundary ancilla for the given dimension if the position is out of bounds.

        Args:
            dim: The dimension to check.
        """
        with self._state.positions_register[dim] == (
            self._state.boundary_offsets[dim] - 1
        ):
            self._state.boundary_ancillas[
                self._state.boundary_ancilla_indexes[dim]
            ].flip()
        with self._state.positions_register[dim] == int(
            self._state.grid_size[dim] + self._state.boundary_offsets[dim]
        ):
            self._state.boundary_ancillas[
                self._state.boundary_ancilla_indexes[dim]
            ].flip()

    @auto_uncompute
    def _is_out_boundary_multi_layer(self, dim: int) -> None:
        """
        Flips the boundary ancilla for the given dimension if the position is out of bounds.

        Args:
            dim: The dimension to check.
        """
        with self._state.positions_register[dim] < self._state.boundary_offsets[dim]:
            self._state.boundary_ancillas[
                self._state.boundary_ancilla_indexes[dim]
            ].flip()
        with self._state.positions_register[dim] >= int(
            self._state.grid_size[dim] + self._state.boundary_offsets[dim]
        ):
            self._state.boundary_ancillas[
                self._state.boundary_ancilla_indexes[dim]
            ].flip()

    @auto_uncompute
    def _comes_from_boundary(self, dim: int) -> None:
        """
        Flips the boundary ancilla for the given dimension if the position is coming out of bounds.

        Args:
            dim: The dimension to check.
        """
        # 1. Condition: At lower boundary AND moving from it (sign is True/1)
        with self._state.positions_register[dim] <= self._state.velocity_register[dim]:
            with self._state.sign_register[dim]:
                self._state.boundary_ancillas[
                    self._state.boundary_ancilla_indexes[dim]
                ].flip()

        # 2. Condition: At upper boundary AND moving from it (sign is False/0)
        upper_bound = int(
            self._state.grid_size[dim] + self._state.boundary_offsets[dim]
        )
        for i in range(1, self._state.boundary_offsets[dim] + 1):
            with self._state.positions_register[dim] == upper_bound - i:
                with self._state.velocity_register[dim] >= i:
                    with ~self._state.sign_register[dim]:
                        self._state.boundary_ancillas[
                            self._state.boundary_ancilla_indexes[dim]
                        ].flip()
