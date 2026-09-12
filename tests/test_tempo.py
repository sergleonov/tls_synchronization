"""Tests for TempoSolver against a real oqupy process-tensor backend.

Everything here drives the *real* solver -- no stubs, mocks, or fakes. The full
pipeline is exercised on deliberately tiny problems (2 TLS, 3 time points, a
short-memory process tensor, a couple of drive frequencies):

* initialization      -- construction, grid, stored truncation knobs
* capability contract -- a missing bath and an unsupported ('drude') family are
                         both rejected (TEMPO renders only the 'power' family)
* process tensor      -- ``_prepare`` builds a real oqupy process tensor, and it
                         (plus the whole solver) survives a pickle round-trip --
                         which is what lets the inherited ``sweep`` ship work to
                         worker processes
* results             -- real ``oqupy.compute_dynamics`` via ``single_run`` and
                         the parallel ``sweep``

Use with an OqupyBackend (operators are dense, oqupy-compatible numpy arrays).
Run with `pytest test_tempo.py` in an environment with oqupy installed.
"""
import pickle

import numpy as np
import pytest

from tls_sync.backend import OqupyBackend
from tls_sync.model import Bath, TLSChainModel, SD_TYPES
from tls_sync.tempo import TempoSolver
from tls_sync.helpers import Dynamics


def make_model(sd="power"):
    baths = {
        "power": Bath("power", coupling=0.05, cutoff=1.0, temperature=0.5, ohmicity=1.0),
        "drude": Bath("drude", coupling=0.05, cutoff=0.5, temperature=0.5),
        None: None,
    }
    return TLSChainModel(omega_tls=[3.95, 4.05], J=0.001, Omega_amp=0.1,
                         T_drive=0.5, n_tls=2, bath=baths[sd])


def make_solver(sd="power", *, T_total=0.5, dt=0.25, tcut=1.0, epsrel=1e-3, **kw):
    # T_total=0.5, dt=0.25 -> times = [0.0, 0.25, 0.5]; short tcut + loose epsrel
    # keep the process tensor tiny so real runs finish quickly.
    return TempoSolver(make_model(sd), OqupyBackend(),
                       T_total=T_total, dt=dt, tcut=tcut, epsrel=epsrel, **kw)


# --- initialization -------------------------------------------------------- #

def test_init_builds_state_grid_and_knobs():
    s = make_solver(tcut=1.0, epsrel=1e-3)
    assert s.H.shape == (4, 4)
    assert np.allclose(s.H, s.H.conj().T)                   # Hermitian (dense array)
    assert np.isclose(np.trace(s.rho0), 1.0)                # normalized state
    assert len(s.times) == 3 and s.times[0] == 0.0 and np.isclose(s.times[-1], 0.5)
    assert s.tcut == 1.0 and s.epsrel == 1e-3


def test_supported_sd_is_power_only():
    assert set(TempoSolver.SUPPORTED_SD) == {"power"}
    assert set(TempoSolver.SUPPORTED_SD) <= set(SD_TYPES)   # a subset of the model's families


# --- capability contract (inherited _require_bath) ------------------------- #

def test_require_bath_missing_raises():
    # TEMPO is meaningless without a bath; rendering must refuse one that's absent.
    with pytest.raises(ValueError):
        make_solver(sd=None)._prepare()


def test_require_bath_rejects_unsupported_family():
    # TEMPO renders only 'power'; a 'drude' bath must be rejected (no conversion).
    with pytest.raises(ValueError):
        make_solver(sd="drude")._prepare()


# --- process tensor + pickling (what makes the parallel sweep possible) ---- #

def test_prepare_builds_picklable_process_tensor():
    s = make_solver("power")
    pt = s._prepare()
    assert pt is not None
    restored = pickle.loads(pickle.dumps(pt))               # crosses the boundary
    assert restored is not None


def test_solver_is_picklable_for_workers():
    # sweep pickles the whole solver to each worker; no oqupy objects are stored
    # on it (bath / params / tensor are all built inside _prepare), so it survives.
    s = make_solver("power")
    s2 = pickle.loads(pickle.dumps(s))
    assert s2.H.shape == s.H.shape
    assert np.isclose(np.trace(s2.rho0), 1.0)
    assert (s2.tcut, s2.epsrel) == (s.tcut, s.epsrel)
    assert s2.ops.collective_exc is not None


# --- real oqupy run: single_run -------------------------------------------- #

def test_single_run_from_ground_state():
    s = make_solver("power")
    dyn = s.single_run(4.0, store_states=True)              # default e_ops = [exc, S+]
    n_t = len(s.times)
    assert isinstance(dyn, Dynamics) and isinstance(dyn.expectations, np.ndarray)
    assert dyn.expectations.shape == (2, 1, n_t)
    assert dyn.states is not None and len(dyn.states[0]) == n_t
    # collective excitation S+S- vanishes in the product ground state at t=0
    assert np.isclose(dyn.expectations[0][0, 0].real, 0.0, atol=1e-8)
    assert np.all(np.isfinite(dyn.expectations[0].real))


def test_single_run_without_states_custom_eops():
    s = make_solver("power")
    dyn = s.single_run(4.0, e_ops=[s.ops.collective_exc], store_states=False)
    assert dyn.states is None
    assert dyn.expectations.shape == (1, 1, len(s.times))


# --- real oqupy run: parallel sweep (end-to-end pickling of tensor + solver) --- #

def test_sweep_parallel_shapes_and_order():
    s = make_solver("power")
    freqs = [3.95, 4.05]
    dyn = s.sweep(freqs, max_workers=2)
    assert dyn.expectations.shape == (2, len(freqs), len(s.times))
    assert dyn.states is None
    assert np.allclose(dyn.omegas, freqs)
    assert np.all(np.isfinite(dyn.expectations[0].real))
    # every frequency starts in the ground state -> zero excitation at t=0
    assert np.allclose(dyn.expectations[0][:, 0].real, 0.0, atol=1e-8)