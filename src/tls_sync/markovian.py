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
from tls_sync.solver import Solver
from tls_sync.model import SD_TYPES

class MarkovianSolver(Solver):
    """Markovian Lindblad integration via qutip.mesolve."""

    # Mesolve reads only the bath's coupling and temperature (weak-coupling
    # limit) and ignores the spectral-density shape, so any family is acceptable.
    SUPPORTED_SD = SD_TYPES

    def __init__(self, model: Any, backend: Any, *,
                 T_total: float, dt: float, dt_output: float | None = None,
                 nsteps: int = 5000) -> None:
        # base builds self.ops, self.H, self.rho0, self.times
        super().__init__(model, backend, T_total=T_total, dt=dt, dt_output=dt_output)
        self.nsteps = nsteps

    def _prepare(self) -> list[Any]:
        """Collapse operators for the run: model phenomenological + bath-Markovian.

        Built once per run (frequency-independent) and reused for every frequency.
        """
        return (self.model.build_dissipators(self.backend, self.ops)
                + self._bath_to_collapse_ops())

    def _bath_to_collapse_ops(self) -> list[Any]:
        """Render the model's bath into individual thermal collapse operators.

            sqrt(lam * (n_i + 1)) * sm_i    (emission)
            sqrt(lam *  n_i     ) * sp_i    (absorption)

        with ``lam = bath.coupling`` and ``n_i = 1/(exp(omega_i/T) - 1)`` at
        ``T = bath.temperature``. Empty if the model carries no bath.
        """
        bath = self._require_bath()

        lam, T = bath.coupling, bath.temperature
        c_ops: list[Any] = []
        for i in range(self.model.n_tls):
            n_i = 1.0 / (np.exp(self.model.omega_tls[i] / T) - 1.0)
            c_ops.append(float(np.sqrt(lam * (n_i + 1.0))) * self.ops.sm[i])
            c_ops.append(float(np.sqrt(lam * n_i)) * self.ops.sp[i])
        return c_ops

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