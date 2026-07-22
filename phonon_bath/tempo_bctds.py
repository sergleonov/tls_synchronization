import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tls_sync.solver import TEMPO, Lindblad
from tls_sync.plotting import plot_diff_map, plot_exc_map, plot_fft_map, plot_sp_map, compute_fft


def main():
    ap = argparse.ArgumentParser(description="TEMPO Simulation for single TLS")
    ap.add_argument("--tag", type=str, default="", help="Tag for output files")
    args = ap.parse_args()

    tls_freqs = [3.75, 3.82]
    J = 0.02
    Omega_amp = 0.1
    lam = 0.02
    gamma_bath = 0.05
    T = 0.5
    T_total = 1600
    T_drive = 100.0
    dt = 0.1
    n_tls = len(tls_freqs)
    n_freqs = 300
    
    zeta = 1
    cutoff_type = "exponential"
    tcut = 5.0
    epsrel = 1e-4

    tempo = TEMPO(tls_freqs=tls_freqs, 
                J=J, 
                Omega_amp=Omega_amp, 
                lam=lam, 
                gamma_bath=gamma_bath, 
                T=T, 
                T_total=T_total, 
                T_drive=T_drive, 
                dt=dt, 
                n_tls=n_tls,
                n_freqs=n_freqs,
                zeta=zeta,
                cutoff_type=cutoff_type,
                tcut=tcut,
                epsrel=epsrel)
    
    mark = Lindblad(tls_freqs=tls_freqs, 
                    J=J, 
                    Omega_amp=Omega_amp, 
                    lam=lam, 
                    T=T, 
                    T_total=T_total, 
                    T_drive=T_drive, 
                    dt=dt, 
                    n_tls=n_tls,
                    n_freqs=n_freqs)

    tempo_exc, tempo_sp = tempo.run()
    mark_exc, mark_sp = mark.run()

    exc = [mark_exc, tempo_exc]
    sp  = [mark_sp, tempo_sp]

    fft_freqs_tempo, fft_data_tempo = compute_fft(tempo_sp, tempo.omega_d_vals, tempo.tlist, tempo.dt, tempo.n_time)
    fft_freqs_mark, fft_data_mark = compute_fft(mark_sp, mark.omega_d_vals, mark.tlist, mark.dt, mark.n_time)

    # save results
    print("Saving data...")
    os.makedirs("bctds_data", exist_ok=True)
    np.savez(f"bctds_data/data_{tempo}_{args.tag}.npz",
            results_tempo=(tempo_exc, tempo_sp), results_mark=(mark_exc, mark_sp),
            fft_freqs_tempo=fft_freqs_tempo, fft_data_tempo=fft_data_tempo,
            fft_freqs_mark=fft_freqs_mark, fft_data_mark=fft_data_mark,
            tls_freqs=tls_freqs, 
            J=J, 
            Omega_amp=Omega_amp, 
            lam=lam, 
            gamma_bath=gamma_bath, 
            T=T, 
            T_total=T_total, 
            T_drive=T_drive, 
            dt=dt, 
            n_tls=n_tls,
            n_freqs=n_freqs,
            zeta=zeta,
            tcut=tcut,
            cutoff_type=cutoff_type,
            epsrel=epsrel)

    print("Plotting results...")
    plot_exc_map(exc, tempo.omega_d_vals, tempo.tlist, labels=[mark._name, tempo._name], filename=f"exc_map_{tempo}_{args.tag}")
    plot_sp_map(sp, tempo.omega_d_vals, tempo.tlist, labels=[mark._name, tempo._name], filename=f"sp_map_{tempo}_{args.tag}")
    plot_diff_map(exc, sp, tempo.omega_d_vals, tempo.tlist, labels=[mark._name, tempo._name], filename=f"diff_map_{tempo}_{args.tag}")
    plot_fft_map([fft_freqs_mark, fft_freqs_tempo], [fft_data_mark, fft_data_tempo], tempo.omega_d_vals, tempo.omega_tls, labels=[mark._name, tempo._name], filename=f"fft_map_{tempo}_{args.tag}")
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()