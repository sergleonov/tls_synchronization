from tls_sync import SemiclassicalSolver, MarkovianSolver
from tls_sync.model import TLSCavityModel, SemiclassicalCavityModel
from tls_sync.utils import compute_fft
from tls_sync.plotting import (
    plot_cavity_map, plot_cavity_iq, plot_exc_map, plot_sp_map, plot_fft_map,
)
import numpy as np
import os


def plot_cavity_field(alpha, omegas, tlist, label, drive_off):
    """Amplitude, phase, and raw quadratures of one solver's complex cavity field."""
    plot_cavity_map(np.abs(alpha), omegas, tlist,
                    fr"{label}: cavity amplitude $|\alpha|$", r"$|\alpha|$",
                    cmap="inferno", drive_off=drive_off,
                    filename=f"{label}_amplitude")
    plot_cavity_map(np.angle(alpha), omegas, tlist,
                    fr"{label}: cavity phase $\arg(\alpha)$", "Phase (rad)",
                    cmap="twilight", vmin=-np.pi, vmax=np.pi, drive_off=drive_off,
                    filename=f"{label}_phase")
    plot_cavity_iq(alpha, omegas, tlist,
                   title=fr"{label}: cavity quadratures $I,\,Q$",
                   filename=f"{label}_iq")


def main():
    # model params
    tls_freqs = [3.75, 3.82]
    J = 0.02
    Omega_amp = 0.1
    g = 0.02
    T = 0.5
    T_drive = 10.0
    n_tls = len(tls_freqs)
    omega_c = float(np.mean(tls_freqs))
    dissipation = 0.002

    # solver params
    n_freqs = 300
    T_total = 100
    dt = 0.1

    # init tiered
    tiered_model = TLSCavityModel(
        omega_tls=tls_freqs,
        J=J,
        Omega_amp=Omega_amp,
        T_drive=T_drive,
        omega_c=omega_c,
        g=g,                       # mode coupling
        Nb=10,                     # fock number
        gamma=dissipation,         # tls relaxation
        temperature=T,
        n_tls=n_tls,
    )

    # init semiclassical
    semiclassical_model = SemiclassicalCavityModel(
        omega_tls=tls_freqs,
        J=J,
        Omega_amp=Omega_amp,
        T_drive=T_drive,
        omega_c=omega_c,
        g=g,
        kappa=0.001,               # cavity leakage
        eta=0.0,
        gamma=dissipation,         # TLS relaxation rate
        gamma_phi=dissipation / 10.0,
        temperature=T,
        n_tls=n_tls,
    )

    # init solvers
    markov = MarkovianSolver(tiered_model, T_total=T_total, dt=dt)
    semi = SemiclassicalSolver(semiclassical_model, T_total=T_total, dt=dt / 10.0, dt_output=dt)

    # freq resolution
    omegas = np.linspace(3.0, 5.0, n_freqs)

    # expectation ops
    e_ops_tier = [markov.ops.collective_exc, markov.ops.collective_sp, markov.ops.aux["a"]]
    e_ops_semi = [semi.ops.collective_exc, semi.ops.collective_sp]

    # run sweeps
    tiered_res = markov.single_run(omegas, e_ops=e_ops_tier)
    semi_res = semi.single_run(omegas, e_ops=e_ops_semi)

    # check the output time match
    tlist = markov.times
    assert np.allclose(tlist, semi.times), "solvers must share the output time grid"

    # tiered res
    tier_exc = tiered_res.expectations[0].real
    tier_sp = tiered_res.expectations[1]
    alpha_tier = tiered_res.expectations[2]
    # semiclassical res
    semi_exc = semi_res.expectations[0].real
    semi_sp = semi_res.expectations[1]
    alpha_semi = semi_res.extra["alpha"]

    # postprocessing: FFT of <S+> along time, per drive frequency
    fft_freqs_tier, fft_data_tier = compute_fft(tier_sp, omegas, tlist, dt, n_time=len(tlist))
    fft_freqs_semi, fft_data_semi = compute_fft(semi_sp, omegas, tlist, dt, n_time=len(tlist))

    # save results
    print("Saving data...")
    os.makedirs("data/bctds_data/", exist_ok=True)
    np.savez(
        "data/bctds_data/tiered_vs_semiclassical.npz",
        omegas=omegas, tlist=tlist,
        tier_exc=tier_exc, tier_sp=tier_sp, alpha_tier=alpha_tier,
        semi_exc=semi_exc, semi_sp=semi_sp, alpha_semi=alpha_semi,
        fft_freqs_tier=fft_freqs_tier, fft_data_tier=fft_data_tier,
        fft_freqs_semi=fft_freqs_semi, fft_data_semi=fft_data_semi,
        tls_freqs=tls_freqs, J=J, Omega_amp=Omega_amp, g=g, T=T,
        T_total=T_total, T_drive=T_drive, dt=dt, n_tls=n_tls,
        n_freqs=n_freqs, omega_c=omega_c, Nb=10, dissipation=dissipation,
    )

    # plot
    print("Plotting...")
    labels = ["Tiered", "Semiclassical"]

    # cavity field plots
    plot_cavity_field(alpha_tier, omegas, tlist, "tiered", T_drive)
    plot_cavity_field(alpha_semi, omegas, tlist, "semiclassical", T_drive)

    # tls excitation
    plot_exc_map([tier_exc, semi_exc], omegas, tlist, labels, filename="exc_map")
    plot_sp_map([tier_sp, semi_sp], omegas, tlist, labels, filename="sp_map")

    # fft maps
    plot_fft_map([fft_freqs_tier, fft_freqs_semi], [fft_data_tier, fft_data_semi],
                 omegas, tls_freqs, labels, filename="fft_map")

    print("Done.")


if __name__ == "__main__":
    main()
