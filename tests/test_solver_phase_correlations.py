import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tls_sync import HEOM, Lindblad, TEMPO, TieredSolver


def make_common_kwargs():
    return dict(
        tls_freqs=np.array([3.1, 4.2]),
        J=0.02,
        Omega_amp=0.1,
        lam=0.02,
        T=0.5,
        T_total=10.0,
        T_drive=5.0,
        dt=0.5,
        n_tls=2,
        n_freqs=2,
    )


def test_lindblad_phase_and_correlation_methods():
    params = make_common_kwargs()
    solver = Lindblad(**params)

    assert pickle.loads(pickle.dumps(solver)).__str__() == solver.__str__()

    exc, sp, states = solver.run(store_states=True)
    assert exc.shape == (solver.n_freqs, solver.n_time)
    assert sp.shape == (solver.n_freqs, solver.n_time)
    assert len(states) == solver.n_freqs

    theta = np.linspace(0, np.pi, 4)
    phi = np.linspace(0, 2 * np.pi, 4)
    husimi = solver.husimi_sim(solver.omega_d_vals[0], theta, phi, method="avg")
    assert husimi.shape == (solver.n_time, len(theta), len(phi))

    phases, t = solver.phase_sim(solver.omega_d_vals[0])
    assert len(phases) == solver.n_tls
    assert np.array_equal(t, solver.tlist)

    pearson_corr, t_corr = solver.pearson_sim(solver.omega_d_vals[0], window_size=2, overlap=1)
    assert isinstance(pearson_corr, dict)
    assert np.array_equal(t_corr, solver.tlist)

    plv_corr, t_plv = solver.plv_sim(solver.omega_d_vals[0], window_size=2, overlap=1)
    assert isinstance(plv_corr, dict)
    assert np.array_equal(t_plv, solver.tlist)

    phases2, corr_dict, t_corr2 = solver.phase_corr_sim(solver.omega_d_vals[0], "pearson", window_size=2, overlap=1)
    assert len(phases2) == solver.n_tls
    assert isinstance(corr_dict, dict)
    assert np.array_equal(t_corr2, solver.tlist)


def test_heom_phase_and_correlation_methods():
    params = make_common_kwargs()
    params.update({"gamma_bath": 0.05, "Nk": 1, "max_depth": 1, "sd_type": "drude"})
    solver = HEOM(**params)

    assert pickle.loads(pickle.dumps(solver)).__str__() == solver.__str__()

    exc, sp, states = solver.run(store_states=True)
    assert exc.shape == (solver.n_freqs, solver.n_time)
    assert sp.shape == (solver.n_freqs, solver.n_time)
    assert len(states) == solver.n_freqs

    theta = np.linspace(0, np.pi, 4)
    phi = np.linspace(0, 2 * np.pi, 4)
    husimi = solver.husimi_sim(solver.omega_d_vals[0], theta, phi, method="avg")
    assert husimi.shape == (solver.n_time, len(theta), len(phi))

    phases, t = solver.phase_sim(solver.omega_d_vals[0])
    assert len(phases) == solver.n_tls
    assert np.array_equal(t, solver.tlist)

    pearson_corr, t_corr = solver.pearson_sim(solver.omega_d_vals[0], window_size=2, overlap=1)
    assert isinstance(pearson_corr, dict)
    assert np.array_equal(t_corr, solver.tlist)

    plv_corr, t_plv = solver.plv_sim(solver.omega_d_vals[0], window_size=2, overlap=1)
    assert isinstance(plv_corr, dict)
    assert np.array_equal(t_plv, solver.tlist)

    phases2, corr_dict, t_corr2 = solver.phase_corr_sim(solver.omega_d_vals[0], "pearson", window_size=2, overlap=1)
    assert len(phases2) == solver.n_tls
    assert isinstance(corr_dict, dict)
    assert np.array_equal(t_corr2, solver.tlist)


def test_tempo_phase_and_correlation_methods():
    params = make_common_kwargs()
    params.update({"gamma_bath": 0.05, "zeta": 1, "cutoff_type": "exponential", "tcut": 3.0, "epsrel": 1e-3})
    solver = TEMPO(**params)

    assert pickle.loads(pickle.dumps(solver)).__str__() == solver.__str__()

    exc, sp, states = solver.run(store_states=True)
    assert exc.shape == (solver.n_freqs, solver.n_time)
    assert sp.shape == (solver.n_freqs, solver.n_time)
    assert len(states) == solver.n_freqs

    theta = np.linspace(0, np.pi, 4)
    phi = np.linspace(0, 2 * np.pi, 4)
    husimi = solver.husimi_sim(solver.omega_d_vals[0], theta, phi, method="avg")
    assert husimi.shape == (solver.n_time, len(theta), len(phi))

    phases, t = solver.phase_sim(solver.omega_d_vals[0])
    assert len(phases) == solver.n_tls
    assert np.array_equal(t, solver.tlist)

    pearson_corr, t_corr = solver.pearson_sim(solver.omega_d_vals[0], window_size=2, overlap=1)
    assert isinstance(pearson_corr, dict)
    assert np.array_equal(t_corr, solver.tlist)

    plv_corr, t_plv = solver.plv_sim(solver.omega_d_vals[0], window_size=2, overlap=1)
    assert isinstance(plv_corr, dict)
    assert np.array_equal(t_plv, solver.tlist)

    phases2, corr_dict, t_corr2 = solver.phase_corr_sim(solver.omega_d_vals[0], "pearson", window_size=2, overlap=1)
    assert len(phases2) == solver.n_tls
    assert isinstance(corr_dict, dict)
    assert np.array_equal(t_corr2, solver.tlist)


def test_tiered_phase_and_correlation_methods():
    params = make_common_kwargs()
    params.update({"g": 0.02, "omega_c": 3.75, "Nb": 3})
    solver = TieredSolver(**params)

    assert pickle.loads(pickle.dumps(solver)).__str__() == solver.__str__()

    exc, sp, states = solver.run(store_states=True)
    assert exc.shape == (solver.n_freqs, solver.n_time)
    assert sp.shape == (solver.n_freqs, solver.n_time)
    assert len(states) == solver.n_freqs

    theta = np.linspace(0, np.pi, 4)
    phi = np.linspace(0, 2 * np.pi, 4)
    husimi = solver.husimi_sim(solver.omega_d_vals[0], theta, phi, method="avg")
    assert husimi.shape == (solver.n_time, len(theta), len(phi))

    phases, t = solver.phase_sim(solver.omega_d_vals[0])
    assert len(phases) == solver.n_tls
    assert np.array_equal(t, solver.tlist)

    pearson_corr, t_corr = solver.pearson_sim(solver.omega_d_vals[0], window_size=2, overlap=1)
    assert isinstance(pearson_corr, dict)
    assert np.array_equal(t_corr, solver.tlist)

    plv_corr, t_plv = solver.plv_sim(solver.omega_d_vals[0], window_size=2, overlap=1)
    assert isinstance(plv_corr, dict)
    assert np.array_equal(t_plv, solver.tlist)

    phases2, corr_dict, t_corr2 = solver.phase_corr_sim(solver.omega_d_vals[0], "pearson", window_size=2, overlap=1)
    assert len(phases2) == solver.n_tls
    assert isinstance(corr_dict, dict)
    assert np.array_equal(t_corr2, solver.tlist)
