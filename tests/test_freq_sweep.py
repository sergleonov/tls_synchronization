"""End-to-end frequency sweeps via Solver.run() for every backend.

These run the real integrators (mesolve / HEOMSolver / oqupy) over a tiny grid
through the parallel executor, so they also exercise the full pickling path.
"""
import numpy as np
import pytest

from _config import EXPECTED_N_TIME, EXPECTED_N_FREQS
from _helpers import state_trace

pytestmark = pytest.mark.slow


def test_run_output_shapes_and_dtypes(any_solver):
    exc, sp = any_solver.run()
    assert exc.shape == (EXPECTED_N_FREQS, EXPECTED_N_TIME)
    assert sp.shape == (EXPECTED_N_FREQS, EXPECTED_N_TIME)
    assert np.isrealobj(exc)
    assert np.iscomplexobj(sp)


def test_run_values_are_physical(any_solver):
    exc, sp = any_solver.run()
    assert np.isfinite(exc).all()
    assert np.isfinite(sp).all()
    # <S+S-> is a (collective) population: non-negative and bounded by n_tls
    assert exc.min() > -1e-6
    assert exc.max() < any_solver.n_tls + 1e-6


def test_run_is_deterministic(any_solver):
    exc1, sp1 = any_solver.run()
    exc2, sp2 = any_solver.run()
    assert np.allclose(exc1, exc2)
    assert np.allclose(sp1, sp2)


def test_run_store_states_shapes_and_traces(any_solver):
    exc, sp, states = any_solver.run(store_states=True)
    assert exc.shape == (EXPECTED_N_FREQS, EXPECTED_N_TIME)
    assert states.shape == (EXPECTED_N_FREQS, EXPECTED_N_TIME)
    assert states.dtype == object
    # every stored density matrix should be trace-preserving
    for f in range(EXPECTED_N_FREQS):
        for t in range(EXPECTED_N_TIME):
            # HEOM hierarchy truncation can allow a small trace drift
            assert np.isclose(state_trace(states[f, t]).real, 1.0, atol=1e-3)
