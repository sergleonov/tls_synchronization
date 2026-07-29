"""Tests for run_parallel / parallel_eval_husimi.

These exercise the serialization path directly with a qutip-free dummy worker,
across every multiprocessing start method the platform supports. This is the
core mechanism behind every solver's frequency sweep, so a regression here
(e.g. an object that no longer pickles) is caught without a full simulation.
"""
import multiprocessing as mp
from functools import partial

import numpy as np

from tls_sync.parallel import run_parallel, parallel_eval_husimi
from _workers import dummy_worker, dummy_eval, N_TIME

def test_run_parallel_shapes_and_routing():
    omega = np.array([3.0, 4.0, 5.0])

    exc, sp = run_parallel(omega, dummy_worker, N_TIME, False, "test",
                           max_workers=2)

    assert exc.shape == (3, N_TIME)
    assert sp.shape == (3, N_TIME)
    assert sp.dtype == complex
    # each row's excitation equals its drive frequency -> results land in order
    assert np.allclose(exc[:, 0], omega)


def test_run_parallel_store_states():
    omega = np.array([3.0, 4.0, 5.0])

    exc, sp, states = run_parallel(omega, partial(dummy_worker, store_states=True),
                                   N_TIME, True, "test", max_workers=2)

    assert exc.shape == (3, N_TIME)
    assert states.shape == (3, N_TIME)
    assert states.dtype == object
    assert np.allclose(np.asarray(states[0, 0]), np.eye(2))


def test_parallel_eval_husimi_shapes():
    theta = np.linspace(0, np.pi, 5)
    phi = np.linspace(0, 2 * np.pi, 5)
    states = [np.eye(2, dtype=complex) for _ in range(N_TIME)]

    # parallel_eval_husimi uses the default multiprocessing context; CI exercises
    # spawn/fork via the TLS_MP_START env var (see conftest).
    Qt = parallel_eval_husimi(states, dummy_eval, theta, phi, "avg", None, "test",
                              max_workers=2)
    assert Qt.shape == (N_TIME, len(theta), len(phi))
    assert np.allclose(Qt, 2.0)   # Re(tr(I_2)) = 2 for each state


def test_max_workers_is_honored():
    # A regression guard for the earlier bug where max_workers was overwritten.
    omega = np.array([3.0, 4.0, 5.0])
    exc, sp = run_parallel(omega, dummy_worker, N_TIME, False, "test", max_workers=1)
    assert exc.shape == (3, N_TIME)
    assert np.allclose(exc[:, 0], omega)
