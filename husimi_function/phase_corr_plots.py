import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../")
from solvers import HEOM, Lindblad, TieredSolver, TEMPO
from bctds_plots import plot_phase_corr_evolution


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
    dt = 0.5
    n_tls = len(tls_freqs)
    n_freqs = 300

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
                lam=g, 
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
    
    omega_d = tls_freqs[0]
    print(f"Drive frequency: {omega_d}")

    print("Computing dynamics...")
    corr_name = "quantum"
    solvers = [mark, tier, heom, tempo]

    phases = []
    correlations = []
    labels = []
    files = []

    for solver in solvers:
        labels.append(solver._name)
        p, c, t = solver.phase_corr_sim(omega_d, corr_name=corr_name, window_size=9, overlap=8)
        phases.append(p)
        correlations.append(c)
        files.append(f"drive_{omega_d}_{solver}")
    
    plot_phase_corr_evolution(phases, correlations, corr_name, t, solver.omega_tls, labels, filenames=files)

    print("Showing plots...")
    plt.show()
    
if __name__ == "__main__":
    status = main()
    sys.exit(status)