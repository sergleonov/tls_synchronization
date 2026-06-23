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
    gamma_bath = 0.05
    T = 0.5
    Nk = 3
    max_depth = 5
    T_total = 1600
    T_drive = 100.0
    dt = 0.5
    n_tls = len(tls_freqs)
    n_freqs = 300

    sd_type = "power"
    ohmicity = 3

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
    
    heom_exc, heom_sp = heom.run()
    mark_exc, mark_sp = mark.run()

    exc = [mark_exc, heom_exc]
    sp  = [mark_sp, heom_sp]

    fft_freqs_heom, fft_data_heom = compute_fft(heom_sp, heom.omega_d_vals, heom.tlist, heom.dt, heom.n_time)
    fft_freqs_mark, fft_data_mark = compute_fft(mark_sp, mark.omega_d_vals, mark.tlist, mark.dt, mark.n_time)

    # save results
    print("Saving data...")
    os.makedirs("bctds_data", exist_ok=True)
    np.savez(f"bctds_data/data_{heom}_{args.tag}.npz",
            results_heom=(heom_exc, heom_sp), results_mark=(mark_exc, mark_sp),
            fft_freqs_heom=fft_freqs_heom, fft_data_tempo=fft_data_heom,
            fft_freqs_mark=fft_freqs_mark, fft_data_mark=fft_data_mark,
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
    plot_exc_map(exc, heom.omega_d_vals, heom.tlist, labels=[mark._name, heom._name], filename=f"exc_map_{heom}_{args.tag}")
    plot_sp_map(sp, heom.omega_d_vals, heom.tlist, labels=[mark._name, heom._name], filename=f"sp_map_{heom}_{args.tag}")
    plot_diff_map(exc, sp, heom.omega_d_vals, heom.tlist, labels=[mark._name, heom._name], filename=f"diff_map_{heom}_{args.tag}")
    plot_fft_map([fft_freqs_mark, fft_freqs_heom], [fft_data_mark, fft_data_heom], heom.omega_d_vals, heom.omega_tls, labels=[mark._name, heom._name], filename=f"fft_map_{heom}_{args.tag}")
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()