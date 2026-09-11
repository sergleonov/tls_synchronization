"""Tests for MarkovianSolver against a real QuTiP backend.

Covers construction, bath rendering, the bath-capability contract, result
assembly, and the real `mesolve` paths (`single_run` and the parallel `sweep`).
Simulations are deliberately tiny (2 TLS, 3 time points, a few drive
frequencies) so the suite runs quickly.

Flat imports match the current working layout; inside the package these become
relative. Run with `pytest test_markovian.py`.
"""

import numpy as np
import pytest

from tls_sync.backend import QutipBackend
from tls_sync.model import Bath, TLSChainModel, SD_TYPES
from tls_sync.markovian import MarkovianSolver
from tls_sync.helpers import Dynamics


def make_model(bath=True):
    b = Bath("ohmic", coupling=0.02, cutoff=0.05, temperature=0.5, ohmicity=1.0) if bath else None
    return TLSChainModel(omega_tls=[3.95, 4.05], J=0.001, Omega_amp=0.1,
                         T_drive=0.5, n_tls=2, bath=b)


def make_solver(bath=True, **kw):
    # T_total=1.0, dt_output=0.5 -> times = [0.0, 0.5, 1.0]
    return MarkovianSolver(make_model(bath=bath), QutipBackend(),
                           T_total=1.0, dt=0.25, dt_output=0.5, **kw)


# --- construction ---------------------------------------------------------- #

def test_init_builds_state_and_grid():
    s = make_solver()
    assert s.ops is not None
    assert s.H.shape == (4, 4)
    assert (s.H - s.H.dag()).norm() < 1e-12                 # Hermitian
    assert abs(s.rho0.tr() - 1.0) < 1e-12                   # normalized state
    assert len(s.times) == 3 and s.times[0] == 0.0 and np.isclose(s.times[-1], 1.0)
    assert s.nsteps == 5000


def test_supported_sd():
    assert set(MarkovianSolver.SUPPORTED_SD) == set(SD_TYPES)


# --- bath rendering -------------------------------------------------------- #

def test_collapse_ops_rates():
    s = make_solver(bath=True)
    c = s._bath_to_collapse_ops()
    assert len(c) == 2 * s.model.n_tls                      # emission + absorption per TLS
    lam, T = s.model.bath.coupling, s.model.bath.temperature
    n0 = 1.0 / (np.exp(s.model.omega_tls[0] / T) - 1.0)
    assert (c[0] - np.sqrt(lam * (n0 + 1.0)) * s.ops.sm[0]).norm() < 1e-12  # emission, TLS 0
    assert (c[1] - np.sqrt(lam * n0) * s.ops.sp[0]).norm() < 1e-12          # absorption, TLS 0


def test_collapse_ops_empty_without_bath():
    with pytest.raises(ValueError):
        make_solver(bath=False)._bath_to_collapse_ops()


def test_prepare_combines_model_and_bath():
    s = make_solver(bath=True)
    assert s.model.build_dissipators(s.backend, s.ops) == []   # chain has no ad-hoc terms
    assert len(s._prepare()) == 2 * s.model.n_tls              # so _prepare == bath ops


def test_default_e_ops_are_collective():
    s = make_solver()
    e = s._default_e_ops()
    assert len(e) == 2 and e[0] is s.ops.collective_exc and e[1] is s.ops.collective_sp


# --- bath-capability contract (_require_bath, inherited) ------------------- #

def test_require_bath_ok():
    assert make_solver(bath=True)._require_bath().sd_type == "ohmic"


def test_require_bath_missing_raises():
    with pytest.raises(ValueError):
        make_solver(bath=False)._require_bath()


def test_require_bath_unsupported_sd_raises():
    class DrudeOnly(MarkovianSolver):
        SUPPORTED_SD = ("drude",)
    s = DrudeOnly(make_model(bath=True), QutipBackend(), T_total=1.0, dt=0.25, dt_output=0.5)
    with pytest.raises(ValueError):
        s._require_bath()                                      # model bath is 'ohmic'


# --- _collect (operator-first assembly) ------------------------------------ #

def test_collect_operator_first_layout():
    s = make_solver()
    n_t = len(s.times)
    results = [[np.full(n_t, 10 * k + j) for j in range(2)] for k in range(3)]
    dyn = s._collect([3.9, 4.0, 4.1], results, store_states=False)
    assert isinstance(dyn, Dynamics) and isinstance(dyn.expectations, np.ndarray)
    assert dyn.expectations.shape == (2, 3, n_t)               # (n_ops, n_omega, n_time)
    assert dyn.states is None
    assert np.allclose(dyn.expectations[0][:, 0], [0, 10, 20])
    assert np.allclose(dyn.expectations[1][:, 0], [1, 11, 21])
    assert np.allclose(dyn.omegas, [3.9, 4.0, 4.1])


def test_collect_with_states():
    s = make_solver()
    n_t = len(s.times)
    results = [([np.full(n_t, k) for _ in range(2)], [None] * n_t) for k in range(2)]
    dyn = s._collect([3.9, 4.0], results, store_states=True)
    assert dyn.expectations.shape == (2, 2, n_t)
    assert dyn.states is not None and len(dyn.states) == 2 and len(dyn.states[0]) == n_t


def test_collect_empty_eops_is_none():
    s = make_solver()
    assert s._collect([3.9, 4.0], [[], []], store_states=False).expectations is None


# --- real mesolve: single_run --------------------------------------------- #

def test_single_run_from_ground_state():
    s = make_solver()
    dyn = s.single_run(4.0, store_states=True)                 # default e_ops = [exc, S+]
    n_t = len(s.times)
    assert dyn.expectations.shape == (2, 1, n_t)
    assert dyn.states is not None and len(dyn.states[0]) == n_t
    # collective excitation S+ S- vanishes in the product ground state at t=0
    assert np.isclose(dyn.expectations[0][0, 0].real, 0.0, atol=1e-9)
    assert np.all(np.isfinite(dyn.expectations[0].real))


def test_single_run_without_states():
    s = make_solver()
    dyn = s.single_run(4.0, e_ops=[s.ops.collective_exc], store_states=False)
    assert dyn.states is None
    assert dyn.expectations.shape == (1, 1, len(s.times))


# --- real mesolve: parallel sweep ----------------------------------------- #

def test_sweep_shapes_order_and_no_states():
    s = make_solver()
    freqs = [3.95, 4.0, 4.05]
    dyn = s.sweep(freqs, max_workers=2)
    assert dyn.expectations.shape == (2, len(freqs), len(s.times))
    assert dyn.states is None
    assert np.allclose(dyn.omegas, freqs)
    assert np.all(np.isfinite(dyn.expectations[0].real))
    # every frequency starts in the ground state -> zero excitation at t=0
    assert np.allclose(dyn.expectations[0][:, 0].real, 0.0, atol=1e-9)


def test_sweep_custom_e_ops():
    s = make_solver()
    dyn = s.sweep([3.95, 4.05], e_ops=[s.ops.collective_sp], max_workers=1)
    assert dyn.expectations.shape == (1, 2, len(s.times))