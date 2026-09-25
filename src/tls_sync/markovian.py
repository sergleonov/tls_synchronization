"""Markovian Lindblad solver (qutip.mesolve backend).

Implements only the per-frequency primitive `_run_one` and the bath-rendering
`_prepare`; the base Solver provides the parallel `sweep` and the single-frequency
`single_run` on top of them. `_prepare` turns the model's structured bath into
thermal collapse operators -- the Markovian counterpart to HEOM's coefficient
expansion and TEMPO's correlation function. Requires a QuTiP backend.
"""
import qutip as qt
import numpy as np
from typing import Any
from tls_sync.backend import QutipBackend
from tls_sync.solver import Solver
from tls_sync.model import SD_TYPES

class MarkovianSolver(Solver):
    """Markovian Lindblad integration via qutip.mesolve.

    The QuTiP backend is fixed via the ``BACKEND`` class attribute, so callers
    construct the solver with just the model (no backend argument).
    """

    # Mesolve reads only the bath's coupling and temperature (weak-coupling
    # limit) and ignores the spectral-density shape, so any family is acceptable.
    SUPPORTED_SD = SD_TYPES
    BACKEND = QutipBackend()

    def __init__(self, model: Any, *,
                 T_total: float, dt: float,
                 nsteps: int = 5000) -> None:
        # base builds self.ops, self.H, self.rho0, self.times
        super().__init__(model, self.BACKEND, T_total=T_total, dt=dt)
        self.nsteps = nsteps

    def _prepare(self) -> list[Any]:
        """Lindblad collapse operators for the run (frequency-independent).

        The model renders its own dissipation via ``build_dissipators`` -- the
        thermal bath collapse operators plus any phenomenological/cavity channels
        -- so the solver just collects them. A bath-less model yields ``[]``
        (a closed, unitary run).
        """
        return self.model.build_dissipators(self.backend, self.ops)

    def _run_one(self, omega_d, prepared, e_ops, store_states) -> tuple[list[np.ndarray], list[Any] | None]:
        """Integrate one drive frequency with mesolve (runs in a worker process).

        `prepared` is the collapse-operator list from `_prepare`. The drive
        coefficient closure is rebuilt here so nothing unpicklable crosses the
        process boundary.
        """

        drive = self.model.drive(self.ops)

        def coeff(t, args):
            return drive.coefficient(t, args["omega"])

        H = qt.QobjEvo([self.H, [drive.operator, coeff]], args={"omega": omega_d})
        result = qt.mesolve(
            H, self.rho0, self.times, c_ops=prepared, e_ops=list(e_ops),
            options={"nsteps": self.nsteps, "progress_bar": "",
                     "store_states": store_states},
        )
        if store_states:
            return result.expect, result.states
        return result.expect
