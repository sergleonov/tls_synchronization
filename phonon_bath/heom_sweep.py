import sys
import argparse
sys.path.append("../")
from solvers import *
from bctds_plots import *


def main():
    ap = argparse.ArgumentParser(description="HEOM Simulation for single TLS")
    ap.add_argument("--tag", type=str, default="", help="Tag for output files")
    args = ap.parse_args()

    tls_freqs = [3.75, 3.82]
    J = 0.02
    Omega_amp = 0.1
    lam = 0.02
    gamma_bath = 0.5
    T = 0.5
    Nk = 3
    max_depth = 5
    T_total = 1600
    T_drive = 100.0
    dt = 0.5
    n_tls = len(tls_freqs)
    n_freqs = 300

    sd_type = "power"
    ohms = [0.5, 1, 3]

    labels = []

    exc, sp = [], []

    fft_freqs, fft_data = [], []


    for ohmicity in ohms:
        heom = HEOM(tls_freqs=tls_freqs, 
                    J=J, 
                    Omega_amp=Omega_amp, 
                    lam=lam, 
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
    
        heom_exc, heom_sp = heom.run()
        exc.append(heom_exc)
        sp.append(heom_sp)
        labels.append(heom._name + f" {ohmicity}")

        fft_freqs_heom, fft_data_heom = compute_fft(heom_sp, heom.omega_d_vals, heom.tlist, heom.dt, heom.n_time)
        fft_freqs.append(fft_freqs_heom)
        fft_data.append(fft_data_heom)

        # save results
        print("Saving data...")
        os.makedirs("bctds_data", exist_ok=True)
        np.savez(f"bctds_data/sweep_data_{heom}_{args.tag}.npz",
                results_heom=(heom_exc, heom_sp),
                fft_freqs_heom=fft_freqs_heom, fft_data_tempo=fft_data_heom,
                tls_freqs=tls_freqs, 
                J=J, 
                Omega_amp=Omega_amp, 
                lam=lam, 
                gamma_bath=gamma_bath, 
                T=T, 
                Nk=Nk, 
                max_depth=max_depth, 
                T_total=T_total, 
                T_drive=T_drive, 
                dt=dt, 
                n_tls=n_tls,
                n_freqs=n_freqs, 
                ohmicity=ohmicity,
                sd_type=sd_type)

    print("Plotting results...")
    plot_exc_map(exc, heom.omega_d_vals, heom.tlist, labels=labels, filename=f"sweep_exc_map_{heom}_{args.tag}")
    plot_sp_map(sp, heom.omega_d_vals, heom.tlist, labels=labels, filename=f"sweep_sp_map_{heom}_{args.tag}")
    plot_diff_map(exc, sp, heom.omega_d_vals, heom.tlist, labels=labels, filename=f"sweep_diff_map_{heom}_{args.tag}")
    plot_fft_map(fft_freqs, fft_data, heom.omega_d_vals, heom.omega_tls, labels=labels, filename=f"sweep_fft_map_{heom}_{args.tag}")
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()