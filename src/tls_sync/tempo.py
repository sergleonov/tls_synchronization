"""TEMPO solver (oqupy process-tensor backend).

Implements only the per-frequency primitive ``_run_one`` and the bath-rendering
``_prepare``; the base :class:`~tls_sync.solver.Solver` supplies the parallel
``sweep`` and single-frequency ``single_run`` on top of them. ``_prepare`` renders
the model's structured :class:`~tls_sync.model.Bath` into an oqupy correlation
function (``PowerLawSD``) and builds a TEMPO *process tensor* -- the non-Markovian
counterpart to the Markovian solver's thermal collapse operators and HEOM's
exponential bath expansion. Use with an :class:`~tls_sync.backend.OqupyBackend`
(operators must be dense, oqupy-compatible numpy arrays).

Only the 'power' (power-law) spectral-density family is supported (oqupy's
``PowerLawSD``); ``SUPPORTED_SD`` plus the inherited ``_require_bath`` reject any
other family. 'power' is the common subset used for HEOM-vs-TEMPO comparison.

Picklability (parallel ``sweep``): the frequency-independent process tensor is
built once in ``_prepare`` and pickled to each worker along with the solver. No
unpicklable oqupy objects are stored on the solver -- bath, parameters, and tensor
are all built inside ``_prepare`` -- so both cross the process boundary cleanly.

The oqupy dynamics are computed on the process tensor's ``dt`` grid spanning
``0 .. T_total`` inclusive, i.e. the same grid the base builds as ``self.times``,
so expectations and states map straight onto the solver's time axis.
"""
import numpy as np
import oqupy
from typing import Any

from tls_sync.solver import Solver
from tls_sync.backend import OqupyBackend


class TempoSolver(Solver):
    """Non-Markovian integration via oqupy's process-tensor TEMPO.

    Truncation / accuracy knobs live here (solver numerics), never on the Bath:

    tcut : float
        Process-tensor memory cutoff time (the bath memory length).
    epsrel : float
        Relative singular-value tolerance for the TEMPO MPO compression.

    The spectral density itself -- power-law exponent (``ohmicity``), cutoff, its
    ``cutoff_type``, coupling, and temperature -- is read from ``model.bath``.
    """

    # oqupy's PowerLawSD renders the 'power' (power-law) family; the drude family
    # is not rendered here. ('power' is the common subset of model.SD_TYPES used
    # for HEOM-vs-TEMPO comparison.)
    SUPPORTED_SD = ("power",)
    BACKEND =  OqupyBackend()

    def __init__(self, model: Any, *,
                 T_total: float, dt: float,
                 tcut: float = 2.5, epsrel: float = 1e-5) -> None:
        # base builds self.ops, self.H, self.rho0, self.times
        super().__init__(model, self.BACKEND, T_total=T_total, dt=dt)
        self.tcut = tcut
        self.epsrel = epsrel

    def _prepare(self):
        """Render the model's bath and build the process tensor (once per run).

        The oqupy bath, TEMPO parameters, and process tensor are all constructed
        here rather than stored on the solver, so nothing unpicklable lives on
        ``self``. Enforces the capability contract via ``_require_bath``.
        """
        bath = self._require_bath()
        correlations = oqupy.PowerLawSD(
            alpha=bath.coupling, zeta=bath.ohmicity, cutoff=bath.cutoff,
            cutoff_type=bath.cutoff_type, temperature=bath.temperature)
        oq_bath = oqupy.Bath(self.model.bath_coupling_op(self.ops), correlations)
        params = oqupy.TempoParameters(dt=self.dt, tcut=self.tcut, epsrel=self.epsrel)
        return oqupy.pt_tempo_compute(
            bath=oq_bath, start_time=0.0, end_time=self.T_total, parameters=params)

    def _run_one(self, omega_d, prepared, e_ops, store_states):
        """Integrate one drive frequency against the prebuilt process tensor.

        ``prepared`` is the process tensor from ``_prepare``. The drive-coefficient
        closure and the time-dependent system are rebuilt here; expectations (and,
        if requested, states) are read off the oqupy ``Dynamics`` object, whose
        time grid already matches ``self.times``.
        """
        drive = self.model.drive(self.ops)

        def ham(t):
            return self.H + drive.coefficient(t, omega_d) * drive.operator

        system = oqupy.TimeDependentSystem(ham)
        dynamics = oqupy.compute_dynamics(
            process_tensor=prepared, system=system,
            initial_state=self.rho0, start_time=0.0, progress_type="silent")

        expect = [np.asarray(dynamics.expectations(np.asarray(op), real=False)[1])
                  for op in e_ops]
        if store_states:
            return expect, list(dynamics.states)
        return expect