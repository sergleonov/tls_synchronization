import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../")
from solvers import HEOM, Lindblad, TieredSolver, TEMPO
from bctds_plots import plot_phase_evolution


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
    T_total = 400
    T_drive = 100.0
    dt = 0.5
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
    
    omega_d = tls_freqs[0]
    print(f"Drive frequency: {omega_d}")

    print("Computing dynamics...")
    solvers = [mark, tier, heom, tempo]
    for solver in solvers:
        phases, t = solver.phase_sim(omega_d)
        plot_phase_evolution(phases, t, solver.omega_tls, solver._name, filename=f"phase_drive_{omega_d}_{solver}")
    
    print("Showing plots...")
    plt.show()
    
if __name__ == "__main__":
    status = main()
    sys.exit(status)