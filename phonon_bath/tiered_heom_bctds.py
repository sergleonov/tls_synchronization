import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../")
from solvers import HEOM, Lindblad, TieredSolver
from bctds_plots import plot_diff_map, plot_exc_map, plot_fft_map, plot_sp_map, compute_fft


def main():
    ap = argparse.ArgumentParser(description="tier Simulation for single TLS")
    ap.add_argument("--tag", type=str, default="", help="Tag for output files")
    args = ap.parse_args()

    tls_freqs = [3.75, 3.82]
    J = 0.02
    Omega_amp = 0.1
    lam = 0.002
    g = 0.02
    T = 0.5
    T_total = 1600
    T_drive = 100.0
    dt = 0.1
    n_tls = len(tls_freqs)
    n_freqs = 300

    Nk = 3
    max_depth = 5
    gamma_bath = 0.05
    ohmicity = None
    sd_type = "drude"

    omega_c = np.mean(tls_freqs)
    Nb = 10

    heom = HEOM(tls_freqs=tls_freqs, 
                J=J, 
                Omega_amp=Omega_amp, 
                lam=g, 
                gamma_bath=gamma_bath, 
                T=T, 
                Nk=Nk, 
                max_depth=max_depth, 
                T_total=T_total, 
                T_drive=T_drive, 
                dt=dt, 
                n_tls=n_tls,
                n_freqs=n_freqs,
                sd_type=sd_type,
                ohmicity=ohmicity)

    tier = TieredSolver(tls_freqs=tls_freqs, 
                        J=J, 
                        Omega_amp=Omega_amp, 
                        lam=lam, 
                        g=g,
                        T=T, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt, 
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        omega_c=omega_c,
                        Nb=Nb)
    
    mark = Lindblad(tls_freqs=tls_freqs, 
                    J=J, 
                    Omega_amp=Omega_amp, 
                    lam=g, 
                    T=T, 
                    T_total=T_total, 
                    T_drive=T_drive, 
                    dt=dt, 
                    n_tls=n_tls,
                    n_freqs=n_freqs)
    
    heom_exc, heom_sp = heom.run()
    tier_exc, tier_sp = tier.run()
    mark_exc, mark_sp = mark.run()

    exc = [mark_exc, tier_exc, heom_exc]
    sp  = [mark_sp, tier_sp, heom_sp]

    fft_freqs_heom, fft_data_heom = compute_fft(heom_sp, heom.omega_d_vals, heom.tlist, heom.dt, heom.n_time)
    fft_freqs_tier, fft_data_tier = compute_fft(tier_sp, tier.omega_d_vals, tier.tlist, tier.dt, tier.n_time)
    fft_freqs_mark, fft_data_mark = compute_fft(mark_sp, mark.omega_d_vals, mark.tlist, mark.dt, mark.n_time)

    labels = [mark._name, tier._name, heom._name]
    
    # save results
    print("Saving data...")
    os.makedirs("bctds_data", exist_ok=True)
    np.savez(f"bctds_data/data_{tier}_{args.tag}.npz",
            results_heom=(heom_exc, heom_sp),
            results_tier=(tier_exc, tier_sp), results_mark=(mark_exc, mark_sp), 
            fft_freqs_heom=fft_freqs_heom, fft_data_heom=fft_data_heom,
            fft_freqs_tier=fft_freqs_tier, fft_data_tier=fft_data_tier,
            fft_freqs_mark=fft_freqs_mark, fft_data_mark=fft_data_mark,
            tls_freqs=tls_freqs, 
            J=J, 
            Omega_amp=Omega_amp, 
            lam=lam, 
            T=T, 
            T_total=T_total, 
            T_drive=T_drive, 
            dt=dt, 
            n_tls=n_tls,
            n_freqs=n_freqs, 
            omega_c=omega_c,
            Nb=Nb,
            gamma_bath=gamma_bath,
            Nk=Nk,
            max_depth=max_depth,
            sd_type=sd_type,
            ohmicity=ohmicity)

    print("Plotting results...")
    plot_exc_map(exc, tier.omega_d_vals, tier.tlist, labels=labels, filename=f"exc_map_{tier}_{args.tag}")
    plot_sp_map(sp, tier.omega_d_vals, tier.tlist, labels=labels, filename=f"sp_map_{tier}_{args.tag}")
    plot_diff_map(exc, sp, tier.omega_d_vals, tier.tlist, labels=labels, filename=f"diff_map_{tier}_{args.tag}")
    plot_fft_map([fft_freqs_mark, fft_freqs_tier, fft_freqs_heom], [fft_data_mark, fft_data_tier, fft_data_heom], tier.omega_d_vals, tier.omega_tls, labels=labels, filename=f"fft_map_{tier}_{args.tag}")
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()