import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../")
from solvers import HEOM, Lindblad, TieredSolver, TEMPO
from bctds_plots import plot_corr_J_sweep


def main():
    tls_freqs = [3.75, 3.82]
    J = 0.002
    Omega_amp = 0.1
    lam = 0.002
    g = 0.02
    gamma_bath = 0.05
    T = 0.5
    Nk = 3
    max_depth = 5
    T_total = 100
    T_drive = 10.0
    dt = 0.1
    n_tls = len(tls_freqs)
    n_freqs = 300

    # Tiered params
    omega_c = np.mean(tls_freqs)
    Nb = 10

    # HEOM params
    sd_type = "drude"
    ohmicity = None

    # TEMPO params
    zeta = 1
    cutoff_type = "exponential"
    tcut = 5.0
    epsrel = 1e-4


    print("Initializing solvers...")
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
    
    window_size = 9
    overlap = 6
    omega_d = tls_freqs[0]
    print(f"Drive frequency: {omega_d}")

    print("Computing dynamics...")

    # resolutions
    n_J, n_ratio = 50, 50
    J_list = np.linspace(0, 0.05, n_J)
    freq_ratios = np.linspace(1, 1.10, n_ratio)

    # sweep
    correlation_map = np.zeros((n_J, n_ratio))

    for j_idx, J_new in enumerate(J_list):
        for r_idx, ratio in enumerate(freq_ratios):
            tls_freqs = [omega_d, omega_d * ratio]
            tier = TieredSolver(tls_freqs=tls_freqs, 
                        J=J_new, 
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
            exit()
            corr_dict, t = tier.correlation_sim(omega_d, window_size, overlap)
            corrs = next(iter(corr_dict.values())) # first correlation
            correlation_map[j_idx, r_idx] = corrs[-1]
            print(j_idx, J_new, r_idx, ratio)

    # plot
    plot_corr_J_sweep(correlation_map, J_list, freq_ratios, tier._name)
    print("Showing plots...")
    plt.show()
    
if __name__ == "__main__":
    status = main()
    sys.exit(status)