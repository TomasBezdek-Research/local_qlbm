import numpy as np
from qrisp import auto_uncompute, prepare

from ._state import LBMState
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
    def apply_initialization(self):
        """
        Applies the gates to set the registers to t=0.
        This is called internally by IQAE/AA.
        """
        # 1. Prepare Positions
        encoded_bounds = self._state.get_encoded_bounds(
            self._state.initial_bubble.bounds
        )
        # _initial_bubble.bounds)
        for d in range(self._state.dimensions):
            num_states = 2 ** self._state.positions_register[d].size
            statevector = np.zeros(num_states)
            start, end = encoded_bounds[d]
            statevector[start : end + 1] = 1.0
            statevector /= np.linalg.norm(statevector)
            prepare(self._state.positions_register[d], statevector)

        # 2. Prepare Velocity
        self._set_global_velocity(self._state.initial_velocity)

    def _set_global_velocity(self, velocity_vector):
        """
        Sets the velocity u for the transport equation.
        velocity_vector: list of floats/ints like [1, -1]

        Args:
            velocity_vector: The velocity vector.
        """
        for d in range(self._state.dimensions):
            u_val = velocity_vector[d]

            # Encode magnitude
            self._state.velocity_register[d].encode(abs(u_val))

            # Encode sign: 1 for positive, 0 for negative
            # (Note: In your logic, True/1 is positive, False/0 is negative)
            if u_val >= 0:
                self._state.sign_register[d].encode(1)
            else:
                self._state.sign_register[d].encode(0)

    ###############################
    #   Simulation Functions    #
    ###############################
    def run_simulation(self, iterations=1, init=True):
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

    def stream(self):
        """
        Do one streaming step.
        """
        for d in range(self._state.dimensions):
            with self._state.sign_register[d]:
                self._state.positions_register[d] += self._state.velocity_register[d]

            with ~self._state.sign_register[d]:
                self._state.positions_register[d] -= self._state.velocity_register[d]

    @auto_uncompute
    def apply_boundary_conditions(self):
        """
        Check if the positions are within the grid boundaries.
        Flip the sign register if the position is out of bounds.
        """
        for d in range(self._state.dimensions):
            if self._state.boundary_conditions[d] == BoundaryCondition.PERIODIC:
                continue
            # Create a clear condition for the boundary
            at_lower = self._state.positions_register[d] == 0
            at_upper = self._state.positions_register[d] == self._state.grid_size[d] + 1
            is_at_boundary = at_lower | at_upper
            with is_at_boundary:
                # Flip the sign register if the position is out of bounds
                self._state.sign_register[d] = ~self._state.sign_register[d]
                # Reflect the position (streaming correction step)
                with self._state.sign_register[d]:
                    self._state.positions_register[d] += self._state.velocity_register[
                        d
                    ]
                with ~self._state.sign_register[d]:
                    self._state.positions_register[d] -= self._state.velocity_register[
                        d
                    ]
