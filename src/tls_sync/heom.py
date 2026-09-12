"""HEOM solver (qutip Hierarchical Equations of Motion backend).

Implements only the per-frequency primitive ``_run_one`` and the bath-rendering
``_prepare``; the base :class:`~tls_sync.solver.Solver` supplies the parallel
``sweep`` and single-frequency ``single_run`` on top of them. ``_prepare`` turns
the model's structured :class:`~tls_sync.model.Bath` into a multi-exponential
expansion of the bath correlation function -- the HEOM counterpart to the
Markovian solver's thermal collapse operators and TEMPO's correlation function.
Requires a QuTiP backend.

Unlike the Markovian solver, HEOM captures the bath non-perturbatively, so it
uses no Lindblad collapse operators; the model's phenomenological dissipators
(``model.build_dissipators``) are not consumed here.

Picklability (parallel ``sweep``): a QuTiP environment / HEOM bath object is not
picklable, so ``_prepare`` returns only the bare exponential coefficients (numpy
arrays + temperature) and ``_run_one`` rebuilds the bath inside the worker
process -- the same trick used for the drive-coefficient closure. Nothing
unpicklable crosses the process boundary.
"""
import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver as QutipHeom
from qutip.core.environment import (
    DrudeLorentzEnvironment,
    OhmicEnvironment,
    ExponentialBosonicEnvironment,
)
from typing import Any

from tls_sync.solver import Solver
from tls_sync.model import SD_TYPES


class HeomSolver(Solver):
    """Numerically-exact open-system integration via qutip's ``HEOMSolver``.

    HEOM reconstructs the full bath correlation function, so both spectral
    density families are supported: 'drude' via a Matsubara expansion and
    'power' (power-law) via a correlation-function fit. The unsupported-family / missing-bath
    guard is the inherited ``_require_bath`` (HEOM requires a bath).

    Truncation knobs live here (solver numerics), never on the Bath (physics):

    Nk : int
        Number of exponential expansion terms -- Matsubara terms for 'drude',
        and the real/imaginary fit counts (``Nr_max = Ni_max = Nk``) for 'power'.
    max_depth : int
        Hierarchy truncation depth.
    nsteps : int
        Cap on internal integrator steps per output step.
    """

    # HEOM represents the bath by its exponential expansion, so any supported
    # spectral-density family is fine (Markovian-style capability check only).
    SUPPORTED_SD = SD_TYPES

    def __init__(self, model: Any, backend: Any, *,
                 T_total: float, dt: float,
                 Nk: int = 3, max_depth: int = 5, nsteps: int = 5000) -> None:
        # base builds self.ops, self.H, self.rho0, self.times
        super().__init__(model, backend, T_total=T_total, dt=dt)
        self.Nk = Nk
        self.max_depth = max_depth
        self.nsteps = nsteps

    # --- bath rendering ---------------------------------------------------- #
    def _prepare(self) -> tuple:
        """Render the model's bath into picklable exponential coefficients.

        Built once per run (frequency-independent). The coefficients -- not the
        QuTiP environment object, which is unpicklable -- are what gets shipped
        to each worker, where ``_run_one`` rebuilds the bath from them.
        """
        return self._bath_to_coeffs(self._build_bath())

    def _build_bath(self):
        """Approximate the model's bath as a multi-exponential environment.

        Consumes the physical :class:`~tls_sync.model.Bath` on the model and
        enforces the capability contract via ``_require_bath``. Drude uses a
        Matsubara expansion; power-law uses a correlation-function fit over the
        solver's output grid (``self.times``).
        """
        bath = self._require_bath()
        match bath.sd_type:
            case "drude":  # Drude-Lorentz: lam = coupling, gamma = cutoff
                env = DrudeLorentzEnvironment(
                    T=bath.temperature, lam=bath.coupling, gamma=bath.cutoff)
                return env.approximate("matsubara", Nk=self.Nk)
            case "power":  # power-law: alpha = coupling, wc = cutoff, s = ohmicity
                env = OhmicEnvironment(
                    T=bath.temperature, alpha=bath.coupling, wc=bath.cutoff,
                    s=bath.ohmicity)
                approx, _info = env.approximate(
                    method="cf", tlist=self.times, target_rmse=None,
                    Nr_max=self.Nk, Ni_max=self.Nk, maxfev=int(1e8))
                return approx
            case _:  # unreachable: _require_bath already rejected other families
                raise ValueError(
                    f"HeomSolver cannot represent sd_type {bath.sd_type!r}.")

    @staticmethod
    def _bath_to_coeffs(bath) -> tuple:
        """Extract picklable exponential coefficients from an approximated bath.

        Returns ``(ck_real, vk_real, ck_imag, vk_imag, T)``: the real and
        imaginary amplitudes, their decay rates, and the bath temperature. This
        is the picklable payload that crosses the process boundary in a sweep.
        """
        ck_real, vk_real, ck_imag, vk_imag = [], [], [], []
        for exp in bath.exponents:
            etype = getattr(exp.type, "name", str(exp.type))
            if etype == "R":
                ck_real.append(complex(exp.ck)); vk_real.append(complex(exp.vk))
            elif etype == "I":
                ck_imag.append(complex(exp.ck)); vk_imag.append(complex(exp.vk))
            elif etype == "RI":
                # combined term: real part uses ck, imaginary part uses ck2,
                # both sharing the same decay rate vk
                ck_real.append(complex(exp.ck)); vk_real.append(complex(exp.vk))
                ck_imag.append(complex(exp.ck2)); vk_imag.append(complex(exp.vk))
            else:
                raise ValueError(f"Unexpected bath exponent type {etype!r}.")
        return (
            np.array(ck_real, dtype=complex),
            np.array(vk_real, dtype=complex),
            np.array(ck_imag, dtype=complex),
            np.array(vk_imag, dtype=complex),
            getattr(bath, "T", None),
        )

    @staticmethod
    def _coeffs_to_bath(coeffs) -> ExponentialBosonicEnvironment:
        """Rebuild an exponential bosonic environment from ``_bath_to_coeffs``."""
        ck_real, vk_real, ck_imag, vk_imag, T = coeffs
        return ExponentialBosonicEnvironment(
            ck_real=list(ck_real), vk_real=list(vk_real),
            ck_imag=list(ck_imag), vk_imag=list(vk_imag), T=T)

    # --- per-frequency integration ----------------------------------------- #
    def _run_one(self, omega_d, prepared, e_ops, store_states):
        """Integrate one drive frequency with HEOM (runs in a worker process).

        ``prepared`` is the coefficient tuple from ``_prepare``; both the bath
        object and the drive-coefficient closure are rebuilt here so nothing
        unpicklable crosses the process boundary.
        """
        drive = self.model.drive(self.ops)

        def coeff(t, args):
            return drive.coefficient(t, args["omega"])

        H = qt.QobjEvo([self.H, [drive.operator, coeff]], args={"omega": omega_d})

        bath = self._coeffs_to_bath(prepared)
        coupling_op = self.model.bath_coupling_op(self.ops)
        solver = QutipHeom(
            H, (bath, coupling_op), max_depth=self.max_depth,
            options={"nsteps": self.nsteps, "progress_bar": "",
                     "store_states": store_states},
        )
        result = solver.run(self.rho0, self.times, e_ops=list(e_ops))
        if store_states:
            return result.expect, result.states
        return result.expect