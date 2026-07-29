"""Phase and correlation observables.

Correctness here is checked via mathematical invariants that must hold
regardless of the underlying dynamics: bounded ranges, self-correlation
identities, and correct dictionary structure (one entry per TLS pair).
"""
import numpy as np
import pytest

from _config import EXPECTED_N_TIME

OMEGA_D = 4.0
WINDOW = 4


# --- pure helpers (deterministic, no dynamics) --------------------------------
def test_pearson_evolution_self_correlation_is_one(lindblad):
    t = np.arange(EXPECTED_N_TIME)
    x = np.sin(0.7 * t) + 0.1 * t          # non-constant signal
    C = lindblad._pearson_evolution(x, x, window_size=WINDOW)
    assert np.isclose(C[-1], 1.0)          # a signal is perfectly self-correlated


def test_plv_evolution_identical_phase_is_one(lindblad):
    t = np.arange(EXPECTED_N_TIME)
    z = np.exp(1j * (0.5 * t))
    plv = lindblad._plv_evolution(z, z, window_size=WINDOW)
    assert np.isclose(plv[-1], 1.0)        # zero phase difference -> PLV = 1


def test_pearson_evolution_rejects_bad_overlap(lindblad):
    x = np.ones(EXPECTED_N_TIME)
    with pytest.raises(ValueError):
        lindblad._pearson_evolution(x, x, window_size=2, overlap=5)


# --- via the public simulation API --------------------------------------------
@pytest.mark.slow
def test_phase_sim_shapes_and_range(qutip_solver):
    phases, t = qutip_solver.phase_sim(OMEGA_D)
    assert len(phases) == qutip_solver.n_tls
    assert len(t) == EXPECTED_N_TIME
    for ph in phases:
        assert len(ph) == EXPECTED_N_TIME
        assert np.isfinite(ph).all()
        assert np.all(ph >= -np.pi - 1e-9) and np.all(ph <= np.pi + 1e-9)


@pytest.mark.slow
def test_pearson_sim_structure_and_bounds(lindblad):
    corr, t = lindblad.pearson_sim(OMEGA_D, window_size=WINDOW, overlap=1)
    assert isinstance(corr, dict)
    assert len(corr) == 1                          # one pair for two TLS
    (series,) = corr.values()
    assert len(series) == EXPECTED_N_TIME
    valid = series[np.isfinite(series)]
    assert np.all(valid >= -1.0 - 1e-9) and np.all(valid <= 1.0 + 1e-9)


@pytest.mark.slow
def test_plv_sim_bounds(lindblad):
    corr, _ = lindblad.plv_sim(OMEGA_D, window_size=WINDOW, overlap=1)
    (series,) = corr.values()
    valid = series[np.isfinite(series)]
    assert np.all(valid >= -1e-9) and np.all(valid <= 1.0 + 1e-9)


@pytest.mark.slow
def test_phase_corr_sim_multiple_metrics(lindblad):
    names = ["pearson", "plv", "connected", "entropy"]
    phases, corrs, t = lindblad.phase_corr_sim(
        OMEGA_D, names, window_size=WINDOW, overlap=1
    )
    assert len(phases) == lindblad.n_tls
    assert len(corrs) == len(names)
    for c in corrs:
        assert isinstance(c, dict) and len(c) == 1

    # entropy (mutual information) is non-negative
    (entropy_series,) = corrs[names.index("entropy")].values()
    valid = entropy_series[np.isfinite(entropy_series)]
    assert np.all(valid >= -1e-9)


@pytest.mark.slow
def test_invalid_correlation_name_raises(lindblad):
    states = lindblad._get_states(OMEGA_D)
    with pytest.raises(ValueError):
        lindblad._cor_sim_helper(states, "not_a_metric", window_size=WINDOW, overlap=1)
