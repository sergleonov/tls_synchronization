"""Semiclassical (mean-field) Maxwell-Bloch solver.

The cavity is a *classical* complex amplitude ``alpha`` evolved alongside the TLS
density matrix ``rho`` -- a mean-field Maxwell-Bloch system -- integrated with a
fixed-step, batched RK4 over the whole drive-frequency sweep at once 
(original physics from Salil Bedkihal); see :class:`~tls_sync.model.SemiclassicalCavityModel` 
for the parameters.

Unlike the Lindblad solvers, a single frequency is not the natural unit here --
the RK4 is vectorized over frequencies -- so this solver **overrides ``sweep``**
(and ``single_run``) rather than dispatching per-frequency ``_run_one`` calls to
worker processes. The classical cavity amplitude is returned in
``Dynamics.extra["alpha"]`` with shape ``(n_omega, n_time)``; the TLS collective
observables (``e_ops``) are read off ``rho`` exactly as for the other solvers, so
a semiclassical :class:`~tls_sync.helpers.Dynamics` lines up with theirs.

Uses a :class:`~tls_sync.backend.NumpyBackend` (dense numpy operators; the RK4 is
plain numpy). Post-processing -- demodulation, FFTs, plots -- lives outside the
solver (``plotting.py`` / ``utils.py``), operating on the returned ``Dynamics``.

Equations of motion (single, uniform couplings ``g``; sz-sz ``J``; per-TLS
fixed rates ``gamma``/``gamma_phi``; classical cavity ``omega_c``/``kappa``/``eta``)::

    d/dt alpha = -i omega_c alpha - (kappa/2) alpha - i eta f(t)
                 - i g <Sx>                                     (Sx = sum_i sx_i)

    d/dt rho   = -i [H_tls + Omega(t) Sx + g (alpha + alpha*) Sx, rho]  +  Lindblad(rho)

with the collective drive carrier ``f(t) = cos(omega_d t)`` gated to
``0 <= t <= T_drive`` and ``Omega(t) = 0.5 Omega_amp f(t)`` (matching the model's
``drive()`` convention). ``H_tls`` and the Lindblad collapse operators come from
the model.
"""
import numpy as np
from tqdm import tqdm
from typing import Any

from tls_sync.solver import Solver
from tls_sync.backend import NumpyBackend
from tls_sync.helpers import Dynamics


class SemiclassicalSolver(Solver):
    """Batched mean-field Maxwell-Bloch integrator for a TLS chain + classical cavity.

    Consumes a :class:`~tls_sync.model.SemiclassicalCavityModel`: the TLS live in
    the (dense, numpy) Hilbert space, and the cavity's scalar parameters
    (``omega_c``, ``g``, ``kappa``, ``eta``) drive a classical amplitude ``alpha``
    coupled to the collective polarization ``<Sx>``. The frequency sweep is
    integrated as one vectorized RK4, so ``sweep`` runs in-process (no worker
    pool) and ``max_workers`` is ignored.

    Parameters
    ----------
    model : SemiclassicalCavityModel
        Supplies the TLS operators / bare Hamiltonian / collapse operators and
        the classical cavity scalars.
    T_total : float
        Total integration time.
    dt : float
        Internal (fixed) RK4 step.
    dt_output : float, optional
        Output sampling step; must be an integer multiple of ``dt`` (defaults to
        ``dt``). ``self.times`` is built on this coarser grid.
    """

    BACKEND = NumpyBackend()

    def __init__(self, model: Any, *,
                 T_total: float, dt: float, dt_output: float | None = None) -> None:
        # base builds self.ops, self.H, self.rho0, self.times (on the dt grid);
        # self.dt stays the RK4 step, and we override self.times onto dt_output.
        super().__init__(model, self.BACKEND, T_total=T_total, dt=dt)
        self.dt_output = dt if dt_output is None else dt_output
        ratio = self.dt_output / dt
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("dt_output must be an integer multiple of dt.")
        self.times = np.arange(0.0, T_total + 0.5 * self.dt_output, self.dt_output)

    def _prepare(self):
        """Collapse-operator triples ``(C, C^dag, C^dag C)`` for the Lindblad term.

        Built once per run from the model's Lindblad channels
        (``build_dissipators``: ``gamma``/``gamma_phi`` and any thermal bath). The
        classical cavity damping ``kappa`` is *not* here -- it enters the
        ``alpha`` equation of motion directly.
        """
        triples = []
        for c in self.model.build_dissipators(self.backend, self.ops):
            C = np.asarray(c)
            Cd = C.conj().T
            triples.append((C, Cd, Cd @ C))
        return triples

    def _rhs(self, t, alpha, rho, omega_d, Sx, triples):
        """Maxwell-Bloch right-hand side, vectorized over the frequency batch.

        ``alpha`` is ``(nf,)``, ``rho`` is ``(nf, d, d)``, ``omega_d`` is ``(nf,)``.
        Returns ``(dalpha, drho)``.
        """
        m = self.model
        on = 1.0 if 0.0 <= t <= m.T_drive else 0.0
        carrier = on * np.cos(omega_d * t)                  # (nf,) shared drive carrier
        tls_coeff = 0.5 * m.Omega_amp * carrier             # mirrors model.drive: 0.5*amp*cos
        cav_drive = m.eta * carrier                         # direct cavity drive

        # cavity amplitude equation (classical mean field)
        sx_exp = np.einsum("ij,nji->n", Sx, rho)            # <Sx> per frequency
        dalpha = (-1j * m.omega_c * alpha
                  - 0.5 * m.kappa * alpha
                  - 1j * cav_drive
                  - 1j * m.g * sx_exp)

        # TLS Hamiltonian: bare + collective drive + cavity back-action g(alpha+alpha*)Sx
        E_cav = alpha + alpha.conjugate()
        H = (self.H[None]
             + tls_coeff[:, None, None] * Sx[None]
             + m.g * E_cav[:, None, None] * Sx[None])

        drho = -1j * (np.matmul(H, rho) - np.matmul(rho, H))
        for C, Cd, CdC in triples:
            drho = drho + (np.matmul(np.matmul(C[None], rho), Cd[None])
                           - 0.5 * (np.matmul(CdC[None], rho) + np.matmul(rho, CdC[None])))
        return dalpha, drho

    def _run_batch(self, omega_d_vals, e_ops, store_states) -> Dynamics:
        """Vectorized fixed-step RK4 over the frequency batch -> a Dynamics.

        Shared by ``single_run`` and ``sweep``. TLS observables (``e_ops``) are
        read off ``rho`` at each output step; the classical cavity amplitude is
        returned in ``extra['alpha']`` with shape ``(n_omega, n_time)``.
        """
        omega_d_vals = np.asarray(omega_d_vals, dtype=float)
        e_ops = list(e_ops) if e_ops is not None else self._default_e_ops()
        triples = self._prepare()

        nf = len(omega_d_vals)
        d = self.rho0.shape[0]
        Sx = sum(self.ops.sx)
        dt = self.dt
        n_steps = int(round(self.T_total / dt))
        steps_per_output = int(round(self.dt_output / dt))
        n_out = len(self.times)

        alpha = np.zeros(nf, dtype=complex)
        rho = np.broadcast_to(np.asarray(self.rho0, dtype=complex), (nf, d, d)).copy()

        alpha_out = np.zeros((nf, n_out), dtype=complex)
        expect_out = np.zeros((len(e_ops), nf, n_out), dtype=complex)
        states_out = [[None] * n_out for _ in range(nf)] if store_states else None

        def record(idx):
            alpha_out[:, idx] = alpha
            for j, op in enumerate(e_ops):
                expect_out[j, :, idx] = np.einsum("ij,nji->n", np.asarray(op), rho)
            if store_states:
                for i in range(nf):
                    states_out[i][idx] = rho[i].copy()

        record(0)
        out_idx = 1
        for step in tqdm(range(n_steps), desc=f"{type(self).__name__} sweep"):
            t = step * dt
            k1a, k1r = self._rhs(t, alpha, rho, omega_d_vals, Sx, triples)
            k2a, k2r = self._rhs(t + 0.5 * dt, alpha + 0.5 * dt * k1a,
                                 rho + 0.5 * dt * k1r, omega_d_vals, Sx, triples)
            k3a, k3r = self._rhs(t + 0.5 * dt, alpha + 0.5 * dt * k2a,
                                 rho + 0.5 * dt * k2r, omega_d_vals, Sx, triples)
            k4a, k4r = self._rhs(t + dt, alpha + dt * k3a,
                                 rho + dt * k3r, omega_d_vals, Sx, triples)
            alpha = alpha + dt / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
            rho = rho + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
            if (step + 1) % steps_per_output == 0 and out_idx < n_out:
                record(out_idx)
                out_idx += 1

        return Dynamics(backend=self.backend, times=self.times, omegas=omega_d_vals,
                        expectations=expect_out if e_ops else None,
                        states=states_out, extra={"alpha": alpha_out})

    def _run_one(self, omega_d, prepared, e_ops, store_states):
        """Not applicable: this solver integrates the whole frequency sweep in one
        batched RK4 (see ``sweep`` / ``single_run``), so the per-frequency
        primitive is unused. Raising keeps a stray call from silently returning
        ``None`` and violating the ``(expect, states)`` contract.
        """
        raise NotImplementedError(
            "SemiclassicalSolver integrates all drive frequencies in one batched "
            "RK4; use single_run() or sweep(), not _run_one().")

    def single_run(self, omega_d, *, e_ops=None, store_states=True) -> Dynamics:
        """Integrate one drive frequency; the cavity field is in ``extra['alpha']``."""
        return self._run_batch(np.array([omega_d], dtype=float), e_ops, store_states)

    def sweep(self, omega_d_vals, *, e_ops=None, max_workers=None) -> Dynamics:
        """Integrate the whole drive-frequency batch at once (vectorized RK4).

        Overrides the base parallel sweep: the RK4 is already vectorized over
        frequencies, so it runs in-process and ``max_workers`` is ignored. The
        cavity amplitude is returned in ``Dynamics.extra['alpha']``
        (``(n_omega, n_time)``); states are not stored (use ``single_run``).
        """
        return self._run_batch(omega_d_vals, e_ops, store_states=False)