"""Husimi-Q function evaluation.

The Husimi Q distribution is non-negative everywhere; that, plus grid shapes
and the per-method contract, is what we assert. Values themselves depend on the
dynamics, so we don't hard-code them.
"""
import numpy as np
import pytest

from _config import EXPECTED_N_TIME

OMEGA_D = 4.0
N_GRID = 5
THETA = np.linspace(0, np.pi, N_GRID)
PHI = np.linspace(0, 2 * np.pi, N_GRID)


def test_eval_husimi_on_initial_state_is_nonnegative(any_solver):
    Q = any_solver.eval_husimi(any_solver.rho0, THETA, PHI, method="avg")
    assert Q.shape == (N_GRID, N_GRID)
    assert np.isfinite(Q).all()
    assert Q.min() > -1e-9                    # Husimi-Q >= 0


def test_eval_husimi_invalid_method_raises(any_solver):
    with pytest.raises(ValueError):
        any_solver.eval_husimi(any_solver.rho0, THETA, PHI, method="bogus")


def test_eval_husimi_ptrace_requires_index(any_solver):
    with pytest.raises(ValueError):
        any_solver.eval_husimi(any_solver.rho0, THETA, PHI, method="ptrace", tls_idx=None)


@pytest.mark.parametrize("method", ["avg", "diff", "ptrace"])
def test_husimi_sim_shapes(any_solver, method):
    tls_idx = 0 if method == "ptrace" else None
    Qt = any_solver.husimi_sim(OMEGA_D, THETA, PHI, method, tls_idx=tls_idx)
    assert Qt.shape == (EXPECTED_N_TIME, N_GRID, N_GRID)
    assert np.isfinite(Qt).all()


@pytest.mark.parametrize("fixture_name", ["lindblad", "tempo"])
def test_husimi_sim_avg_is_nonnegative(request, fixture_name):
    # Lindblad and TEMPO produce completely-positive dynamics, so the reduced
    # states are valid density matrices
    solver = request.getfixturevalue(fixture_name)
    Qt = solver.husimi_sim(OMEGA_D, THETA, PHI, "avg")
    assert Qt.min() > -1e-9


def test_husimi_sim_heom_positivity_note(heom_drude):
    # A truncated HEOM hierarchy is not guaranteed completely positive, so the
    # reduced state can carry small negative eigenvalues and the Husimi Q can dip
    # slightly below zero. That is a property of the truncation, not a bug: we
    # only require the dip to be small. Deepen max_depth / Nk to shrink it.
    Qt = heom_drude.husimi_sim(OMEGA_D, THETA, PHI, "avg")
    assert np.isfinite(Qt).all()
    assert Qt.min() > -1e-2      # loose bound; tighten as the hierarchy deepens


@pytest.mark.xfail(reason="Husimi axis order for unequal len(theta)/len(phi) is "
                          "unverified; documents a known open question.",
                   strict=False)
def test_eval_husimi_unequal_grid_axes(lindblad):
    theta = np.linspace(0, np.pi, 4)
    phi = np.linspace(0, 2 * np.pi, 7)
    Q = lindblad.eval_husimi(lindblad.rho0, theta, phi, method="avg")
    # Intended convention: first axis theta, second axis phi.
    assert Q.shape == (len(theta), len(phi))