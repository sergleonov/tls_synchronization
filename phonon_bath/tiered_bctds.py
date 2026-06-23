import sys
import argparse
sys.path.append("../")
from solvers import *
from bctds_plots import *


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
    T_total = 400
    T_drive = 100.0
    dt = 0.5
    n_tls = len(tls_freqs)
    n_freqs = 100

    omega_c = 3.80
    Nb = 10

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
    
    tier_exc, tier_sp = tier.run()
    mark_exc, mark_sp = mark.run()

    exc = [mark_exc, tier_exc]
    sp  = [mark_sp, tier_sp]

    fft_freqs_tier, fft_data_tier = compute_fft(tier_sp, tier.omega_d_vals, tier.tlist, tier.dt, tier.n_time)
    fft_freqs_mark, fft_data_mark = compute_fft(mark_sp, mark.omega_d_vals, mark.tlist, mark.dt, mark.n_time)

    # save results
    print("Saving data...")
    os.makedirs("bctds_data", exist_ok=True)
    np.savez(f"bctds_data/data_{tier}_{args.tag}.npz",
            results_tier=(tier_exc, tier_sp), results_mark=(mark_exc, mark_sp),
            fft_freqs_tier=fft_freqs_tier, fft_data_tempo=fft_data_tier,
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
            Nb=Nb)

    print("Plotting results...")
    plot_exc_map(exc, tier.omega_d_vals, tier.tlist, labels=[mark._name, tier._name], filename=f"exc_map_{tier}_{args.tag}")
    plot_sp_map(sp, tier.omega_d_vals, tier.tlist, labels=[mark._name, tier._name], filename=f"sp_map_{tier}_{args.tag}")
    plot_diff_map(exc, sp, tier.omega_d_vals, tier.tlist, labels=[mark._name, tier._name], filename=f"diff_map_{tier}_{args.tag}")
    plot_fft_map([fft_freqs_mark, fft_freqs_tier], [fft_data_mark, fft_data_tier], tier.omega_d_vals, tier.omega_tls, labels=[mark._name, tier._name], filename=f"fft_map_{tier}_{args.tag}")
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()