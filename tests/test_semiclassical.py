"""Tests for SemiclassicalSolver (mean-field Maxwell-Bloch RK4).

Everything drives the *real* solver -- no stubs, mocks, or fakes. The problems
are deliberately tiny (2 TLS, 3 output points, a couple of drive frequencies, a
handful of RK4 steps). Coverage:

* initialization      -- grid, stored knobs, the dt/dt_output split + validation
* dissipators         -- gamma / gamma_phi rendered into Lindblad triples; the
                         bath-less "no warning" contract; _run_one refuses (batched)
* results             -- real batched RK4 via single_run and sweep, with the
                         classical cavity field in ``extra["alpha"]``
* vectorization       -- a batched sweep reproduces each single-frequency run
                         (the frequencies are independent in the batch)

Run with `pytest test_semiclassical.py`.
"""
import warnings

import numpy as np
import pytest

from tls_sync.model import SemiclassicalCavityModel
from tls_sync.semiclassical import SemiclassicalSolver
from tls_sync.helpers import Dynamics


def make_model(*, gamma=5e-4, gamma_phi=5e-5, bath=None):
    return SemiclassicalCavityModel(
        omega_tls=[3.95, 4.05], J=0.001, Omega_amp=0.1, T_drive=0.5,
        omega_c=4.0, g=0.02, kappa=0.001, eta=1e-4,
        gamma=gamma, gamma_phi=gamma_phi, n_tls=2, bath=bath)


def make_solver(*, T_total=1.0, dt=0.1, dt_output=0.5, **model_kw):
    # T_total=1.0, dt=0.1, dt_output=0.5 -> times = [0.0, 0.5, 1.0] (10 RK4 steps)
    return SemiclassicalSolver(make_model(**model_kw),
                               T_total=T_total, dt=dt, dt_output=dt_output)


# --- initialization -------------------------------------------------------- #

def test_init_builds_state_grid_and_knobs():
    s = make_solver()
    assert s.backend.name == "numpy"
    assert s.H.shape == (4, 4)
    assert np.allclose(s.H, s.H.conj().T)                    # Hermitian (dense array)
    assert np.isclose(np.trace(s.rho0), 1.0)                 # normalized state
    assert len(s.times) == 3 and s.times[0] == 0.0 and np.isclose(s.times[-1], 1.0)
    assert s.dt == 0.1 and s.dt_output == 0.5                # integration vs output step


def test_dt_output_must_be_multiple_of_dt():
    with pytest.raises(ValueError):
        make_solver(dt=0.03, dt_output=0.1)                  # 0.1 / 0.03 is not integer


def test_default_dt_output_equals_dt():
    s = SemiclassicalSolver(make_model(), T_total=1.0, dt=0.25)   # no dt_output
    assert s.dt_output == 0.25
    assert len(s.times) == 5                                 # [0, .25, .5, .75, 1.0]


# --- dissipators ----------------------------------------------------------- #

def test_prepare_dissipator_triples():
    s = make_solver(gamma=5e-4, gamma_phi=5e-5)
    triples = s._prepare()
    assert len(triples) == 2 * s.model.n_tls                 # relaxation + dephasing per TLS
    C, Cd, CdC = triples[0]                                  # first channel: sqrt(gamma) sm_0
    assert np.allclose(C, np.sqrt(s.model.gamma) * s.ops.sm[0])
    assert np.allclose(Cd, C.conj().T)                       # C^dag
    assert np.allclose(CdC, C.conj().T @ C)                  # C^dag C


def test_prepare_empty_without_dissipation():
    # no rates and no bath -> no collapse operators (closed, unitary TLS)
    s = make_solver(gamma=0.0, gamma_phi=0.0)
    assert s._prepare() == []


def test_no_bath_does_not_warn():
    # SemiclassicalCavityModel is bath-less by design (dissipation is
    # gamma/gamma_phi), so rendering must NOT warn -- unlike the quantum models.
    s = make_solver()                                        # bath=None
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)          # any UserWarning -> error
        s._prepare()


def test_run_one_is_not_implemented():
    # batched solver: the per-frequency primitive is unused and must fail loudly.
    s = make_solver()
    with pytest.raises(NotImplementedError):
        s._run_one(4.0, s._prepare(), s._default_e_ops(), False)


# --- real batched RK4: single_run ------------------------------------------ #

def test_single_run_from_ground_state():
    s = make_solver()
    dyn = s.single_run(4.0, store_states=True)               # default e_ops = [exc, S+]
    n_t = len(s.times)
    assert isinstance(dyn, Dynamics)
    assert dyn.expectations.shape == (2, 1, n_t)             # (n_ops, n_omega=1, n_time)
    # states follow the package [i_omega][i_time] convention (one omega here)
    assert dyn.states is not None and len(dyn.states) == 1 and len(dyn.states[0]) == n_t
    # collective excitation S+S- vanishes in the product ground state at t=0
    assert np.isclose(dyn.expectations[0][0, 0].real, 0.0, atol=1e-12)
    assert np.all(np.isfinite(dyn.expectations[0].real))
    # the classical cavity field lives in extra, indexed [i_omega][i_time]
    alpha = dyn.extra["alpha"]
    assert alpha.shape == (1, n_t)
    assert alpha[0, 0] == 0.0                                # cavity starts empty
    assert np.any(np.abs(alpha[0, 1:]) > 0.0)               # ... and gets driven
    # the RK4 generator is traceless, so Tr(rho) is preserved to ~machine precision
    for rho in dyn.states[0]:
        assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)


def test_single_run_without_states_custom_eops():
    s = make_solver()
    dyn = s.single_run(4.0, e_ops=[s.ops.collective_exc], store_states=False)
    assert dyn.states is None
    assert dyn.expectations.shape == (1, 1, len(s.times))
    assert dyn.extra["alpha"].shape == (1, len(s.times))


# --- real batched RK4: sweep ----------------------------------------------- #

def test_sweep_batched_shapes_and_ground():
    s = make_solver()
    freqs = [3.95, 4.0, 4.05]
    dyn = s.sweep(freqs, max_workers=2)                      # max_workers accepted but ignored
    n_t = len(s.times)
    assert dyn.expectations.shape == (2, len(freqs), n_t)    # (n_ops, n_omega, n_time)
    assert dyn.states is None
    assert np.allclose(dyn.omegas, freqs)
    alpha = dyn.extra["alpha"]
    assert alpha.shape == (len(freqs), n_t)
    assert np.allclose(alpha[:, 0], 0.0)                     # every frequency starts empty
    assert np.allclose(dyn.expectations[0][:, 0].real, 0.0, atol=1e-12)  # ground at t=0
    assert np.all(np.isfinite(dyn.expectations[0].real)) and np.all(np.isfinite(alpha))


def test_batch_matches_single_frequency():
    # The sweep is vectorized over *independent* frequencies -- no cross-frequency
    # coupling -- so each frequency's batched result must equal running it alone.
    s = make_solver()
    freqs = [3.95, 4.05]
    swept = s.sweep(freqs)
    for k, w in enumerate(freqs):
        one = s.single_run(w, store_states=False)
        assert np.allclose(swept.expectations[:, k, :], one.expectations[:, 0, :])
        assert np.allclose(swept.extra["alpha"][k], one.extra["alpha"][0])