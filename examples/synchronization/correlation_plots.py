import matplotlib.pyplot as plt
import numpy as np
import sys
from tls_sync import HEOM, Lindblad, TieredSolver, TEMPO
from tls_sync.plotting import plot_correlations


def main():
    tls_freqs = [3.75, 3.82]
    J = 0.02
    Omega_amp = 0.1
    lam = 0.002
    g = 0.02
    gamma_bath = 0.05
    T = 0.5
    Nk = 3
    max_depth = 5
    T_total = 1000
    T_drive = 100.0
    dt = 0.1
    n_tls = len(tls_freqs)

    # Tiered params
    omega_c = np.mean(tls_freqs)
    Nb = 10

    # HEOM params
    sd_type = "power"
    ohmicity = 1.0

    # TEMPO params
    zeta = 1
    cutoff_type = "exponential"
    tcut = 2.5
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
                    n_tls=n_tls)
    
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
                        omega_c=omega_c,
                        Nb=Nb)

    tempo = TEMPO(tls_freqs=tls_freqs, 
                J=J, 
                Omega_amp=Omega_amp, 
                lam=g, 
                gamma_bath=gamma_bath, 
                T=T, 
                T_total=T_total, 
                T_drive=T_drive, 
                dt=dt, 
                n_tls=n_tls,
                zeta=zeta,
                cutoff_type=cutoff_type,
                tcut=tcut,
                epsrel=epsrel)
    
    window_size = 150
    overlap = 149
    omega_d = tls_freqs[0]
    print(f"Drive frequency: {omega_d}")

    print("Computing dynamics...")
    solvers = [mark, tier, heom]
    for solver in solvers:
        correlations, t = solver.pearson_sim(omega_d, window_size, overlap)
        plot_correlations(correlations, t, solver.get_name(), corr_name="pearson", filename=f"correlation_drive_{omega_d}_{solver}")
    
    print("Showing plots...")
    plt.show()
    
if __name__ == "__main__":
    status = main()
    sys.exit(status)