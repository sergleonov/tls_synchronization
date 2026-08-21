"""Smoke tests for the plotting module.

We don't assert on pixels; we check that each figure-producing function runs
under the Agg backend with mathtext (no system LaTeX) and writes its file.
Synthetic arrays stand in for solver output so these stay fast and qutip-free.
"""
import numpy as np
import pytest

from tls_sync import plotting

N_FREQ = 4
N_TIME = 5
N_GRID = 6
LABELS = ["A", "B"]


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run inside a temp dir so output folders don't pollute the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def omega_and_time():
    return np.linspace(3.0, 5.0, N_FREQ), np.linspace(0.0, 4.0, N_TIME)


def test_plot_exc_map(in_tmp_cwd, omega_and_time):
    omega, tlist = omega_and_time
    res = [np.random.rand(N_FREQ, N_TIME) for _ in LABELS]
    plotting.plot_exc_map(res, omega, tlist, LABELS, save=True, filename="exc")
    assert (in_tmp_cwd / "figs" / "bctds_figures" / "exc.png").exists()


def test_plot_sp_map(in_tmp_cwd, omega_and_time):
    omega, tlist = omega_and_time
    res = [np.random.rand(N_FREQ, N_TIME) + 1j * np.random.rand(N_FREQ, N_TIME)
           for _ in LABELS]
    plotting.plot_sp_map(res, omega, tlist, LABELS, save=True, filename="sp")
    assert (in_tmp_cwd / "figs" / "bctds_figures" / "sp.png").exists()


def test_plot_diff_map(in_tmp_cwd, omega_and_time):
    omega, tlist = omega_and_time
    exc = [np.random.rand(N_FREQ, N_TIME) for _ in LABELS]
    sp = [np.random.rand(N_FREQ, N_TIME) + 1j * np.random.rand(N_FREQ, N_TIME)
          for _ in LABELS]
    plotting.plot_diff_map(exc, sp, omega, tlist, LABELS, save=True, filename="diff")
    assert (in_tmp_cwd / "figs" / "bctds_figures" / "diff.png").exists()


def test_plot_fft_map(in_tmp_cwd, omega_and_time):
    omega, _ = omega_and_time
    fft_freqs = [np.linspace(0, 0.1, 8) for _ in LABELS]
    fft_data = [np.random.rand(N_FREQ, 8) for _ in LABELS]
    omega_tls = np.array([3.5, 4.0])
    plotting.plot_fft_map(fft_freqs, fft_data, omega, omega_tls, LABELS,
                          save=True, filename="fft")
    assert (in_tmp_cwd / "figs" / "bctds_figures" / "fft.png").exists()


def test_plot_phase_evolution(in_tmp_cwd, omega_and_time):
    _, tlist = omega_and_time
    phases = [np.sin(tlist), np.cos(tlist)]
    plotting.plot_phase_evolution(phases, tlist, [3.5, 4.0], "Markovian",
                                  save=True, filename="phase")
    assert (in_tmp_cwd / "figs" / "phase_plots" / "phase.png").exists()


def test_plot_correlations(in_tmp_cwd, omega_and_time):
    _, tlist = omega_and_time
    corr = {"TLS 3.5, 4.0": np.linspace(-1, 1, N_TIME)}
    plotting.plot_correlations(corr, tlist, "Markovian", "Pearson",
                               save=True, filename="corr")
    assert (in_tmp_cwd / "figs" / "correlation_plots" / "Pearson_corr.png").exists()


def test_plot_phase_corr_evolution(in_tmp_cwd, omega_and_time):
    _, tlist = omega_and_time
    phases = [np.sin(tlist), np.cos(tlist)]
    corrs = [{"TLS 3.5, 4.0": np.linspace(-1, 1, N_TIME)}]
    plotting.plot_phase_corr_evolution(phases, corrs, ["pearson"], tlist,
                                       [3.5, 4.0], "Markovian",
                                       filename="pc", save=True)
    assert (in_tmp_cwd / "figs" / "correlation_plots" / "corrs_pc.png").exists()


def test_plot_corr_J_sweep(in_tmp_cwd):
    J_list = np.linspace(0, 0.1, N_FREQ)
    ratios = np.linspace(0.8, 1.2, N_TIME)
    corr_map = np.random.uniform(-1, 1, (N_FREQ, N_TIME))
    plotting.plot_corr_J_sweep(corr_map, J_list, ratios, "Markovian",
                               save=True, filename="jsweep")
    assert (in_tmp_cwd / "figs" / "correlation_plots" / "jsweep.png").exists()


def test_plot_husimi_snapshots(in_tmp_cwd, omega_and_time):
    _, tlist = omega_and_time
    theta = np.linspace(0, np.pi, N_GRID)
    phi = np.linspace(0, 2 * np.pi, N_GRID)
    Qts = [np.random.rand(N_TIME, N_GRID, N_GRID) for _ in LABELS]
    fig, axes = plotting.plot_husimi_snapshots(
        Qts, tlist, snapshots=[tlist[0], tlist[-1]], theta=theta, phi=phi,
        labels=LABELS, omega_d=4.0, tls_freqs=[3.5, 4.0],
        filename="snap", save=True,
    )
    assert (in_tmp_cwd / "figs" / "husimi_snapshots" / "snap.png").exists()


def test_set_plot_format_updates_rcparams_then_restore():
    import matplotlib.pyplot as plt
    plotting.set_plot_format(scale=2)
    assert plt.rcParams["font.size"] == pytest.approx(24)
    # set_plot_format enables usetex; undo it so later tests keep mathtext
    plt.rcParams["text.usetex"] = False
