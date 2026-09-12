"""Tests for HeomSolver against a real QuTiP HEOM backend.

Everything here drives the *real* solver -- no stubs, mocks, or fakes. The full
pipeline is exercised on deliberately tiny problems (2 TLS, 5 time points, a
couple of drive frequencies, shallow hierarchy):

* initialization      -- construction, grid, stored truncation knobs
* bath creation       -- both spectral-density families (drude Matsubara,
                         power-law correlation-function fit) and the coefficient
                         extraction
* pickling            -- the prepared coefficients *and* the whole solver
                         survive a pickle round-trip (this is what lets
                         ``sweep`` ship work to worker processes)
* results             -- real ``HEOMSolver.run`` via ``single_run`` and the
                         parallel ``sweep``

The heavy HEOM integration (``single_run``/``sweep``) runs on the Drude bath
only; the integration path (``_run_one`` -> ``_coeffs_to_bath`` ->
``HEOMSolver.run``) is identical for both families, and only ``_build_bath``
branches on ``sd_type`` -- which the parametrized bath-creation test covers for
both -- so a second, expensive power-law run would add cost without new coverage.

Run with `pytest test_heom.py` in an environment with qutip installed.
"""
import pickle

import numpy as np
import pytest
from qutip.core.environment import ExponentialBosonicEnvironment

from tls_sync.backend import QutipBackend
from tls_sync.model import Bath, TLSChainModel, SD_TYPES
from tls_sync.heom import HeomSolver
from tls_sync.helpers import Dynamics


def make_model(sd="drude"):
    baths = {
        "drude": Bath("drude", coupling=0.05, cutoff=0.5, temperature=0.5),
        "power": Bath("power", coupling=0.05, cutoff=1.0, temperature=0.5, ohmicity=1.0),
        None: None,
    }
    return TLSChainModel(omega_tls=[3.95, 4.05], J=0.001, Omega_amp=0.1,
                         T_drive=0.5, n_tls=2, bath=baths[sd])


def make_solver(sd="drude", *, T_total=1.0, dt=0.25,
                Nk=1, max_depth=2, **kw):
    # T_total=1.0, dt=0.25; times = [0.0, 0.25, 0.5, 0.75, 1.0], small Nk/max_depth
    # keep the hierarchy tiny so real runs finish quickly.
    return HeomSolver(make_model(sd), QutipBackend(),
                      T_total=T_total, dt=dt,
                      Nk=Nk, max_depth=max_depth, **kw)


# --- initialization -------------------------------------------------------- #

def test_init_builds_state_grid_and_knobs():
    s = make_solver(Nk=1, max_depth=2, nsteps=1234)
    assert s.H.shape == (4, 4)
    assert (s.H - s.H.dag()).norm() < 1e-12                 # Hermitian
    assert abs(s.rho0.tr() - 1.0) < 1e-12                   # normalized state
    assert len(s.times) == 5 and s.times[0] == 0.0 and np.isclose(s.times[-1], 1.0)
    assert s.Nk == 1 and s.max_depth == 2 and s.nsteps == 1234


def test_supported_sd():
    assert set(HeomSolver.SUPPORTED_SD) == set(SD_TYPES)


# --- bath creation --------------------------------------------------------- #

def test_require_bath_missing_raises():
    # HEOM is meaningless without a bath; rendering must refuse one that's absent.
    with pytest.raises(ValueError):
        make_solver(sd=None)._prepare()


@pytest.mark.parametrize("sd", ["drude", "power"])
def test_build_bath_and_coeffs(sd):
    # The power-law correlation-function fit wants a few sample points, so give it a
    # slightly longer grid. No HEOM run happens here, so it stays cheap.
    s = make_solver(sd, T_total=2.0, dt=0.25) if sd == "power" \
        else make_solver(sd)

    bath = s._build_bath()
    assert len(bath.exponents) >= 1                         # a real multi-exp env

    coeffs = s._bath_to_coeffs(bath)
    assert len(coeffs) == 5
    ck_r, vk_r, ck_i, vk_i, _T = coeffs
    for arr in (ck_r, vk_r, ck_i, vk_i):
        assert isinstance(arr, np.ndarray) and arr.dtype == complex
        assert np.isfinite(arr).all()                       # finite real & imag parts
    assert len(ck_r) == len(vk_r) and len(ck_i) == len(vk_i)
    # each exponent contributes >=1 (real or imag) coefficient; RI contributes 2
    assert len(ck_r) + len(ck_i) >= len(bath.exponents)


# --- pickling (what makes the parallel sweep possible) --------------------- #

def test_prepared_coeffs_are_picklable_and_rebuild_a_bath():
    s = make_solver("drude")
    prepared = s._prepare()

    restored = pickle.loads(pickle.dumps(prepared))         # crosses the boundary
    for a, b in zip(prepared[:4], restored[:4]):
        assert np.array_equal(a, b)
    assert prepared[4] == restored[4]

    bath = HeomSolver._coeffs_to_bath(restored)
    assert isinstance(bath, ExponentialBosonicEnvironment)


def test_solver_is_picklable_for_workers():
    # sweep pickles the whole solver (backend + Qobj operators + H + rho0) to
    # each worker; verify that survives without the legacy __getstate__ hooks.
    s = make_solver("drude")
    s2 = pickle.loads(pickle.dumps(s))
    assert s2.H.shape == s.H.shape
    assert abs(s2.rho0.tr() - 1.0) < 1e-12
    assert (s2.Nk, s2.max_depth, s2.nsteps) == (s.Nk, s.max_depth, s.nsteps)
    assert s2.ops.collective_exc is not None


# --- real HEOM run: single_run --------------------------------------------- #

def test_single_run_from_ground_state():
    s = make_solver("drude")
    dyn = s.single_run(4.0, store_states=True)              # default e_ops = [exc, S+]
    n_t = len(s.times)
    assert isinstance(dyn, Dynamics) and isinstance(dyn.expectations, np.ndarray)
    assert dyn.expectations.shape == (2, 1, n_t)
    assert dyn.states is not None and len(dyn.states[0]) == n_t
    # collective excitation S+S- vanishes in the product ground state at t=0
    assert np.isclose(dyn.expectations[0][0, 0].real, 0.0, atol=1e-9)
    assert np.all(np.isfinite(dyn.expectations[0].real))


def test_single_run_without_states_custom_eops():
    s = make_solver("drude")
    dyn = s.single_run(4.0, e_ops=[s.ops.collective_exc], store_states=False)
    assert dyn.states is None
    assert dyn.expectations.shape == (1, 1, len(s.times))


# --- real HEOM run: parallel sweep (end-to-end pickling + reconstruction) --- #

def test_sweep_parallel_shapes_and_order():
    s = make_solver("drude")
    freqs = [3.95, 4.05]
    dyn = s.sweep(freqs, max_workers=2)
    assert dyn.expectations.shape == (2, len(freqs), len(s.times))
    assert dyn.states is None
    assert np.allclose(dyn.omegas, freqs)
    assert np.all(np.isfinite(dyn.expectations[0].real))
    # every frequency starts in the ground state -> zero excitation at t=0
    assert np.allclose(dyn.expectations[0][:, 0].real, 0.0, atol=1e-9)