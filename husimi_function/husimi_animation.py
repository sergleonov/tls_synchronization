import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../")
from solvers import HEOM, Lindblad, TieredSolver
from bctds_plots import generate_husimi_anim


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

    omega_c = np.mean(tls_freqs)
    Nb = 10

    sd_type = "drude"
    ohmicity = None

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
    
    solvers = [mark, heom, tier]
    
    tls_idx = 0
    method = "ptrace"
    omega_d = tls_freqs[0]
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 80)
    
    Qts = []
    labels = []
    for solver in solvers:
        Qts.append(solver.husimi_sim(omega_d, theta, phi, method=method, tls_idx=tls_idx))
        labels.append(solver._name)

    sd = ""
    if sd_type == "power":
        sd = sd_type + f"_ohm{ohmicity}"
    filename = f"sim_J{J}_Om{Omega_amp}_omega_c{omega_c}_cutoff{gamma_bath}_g{g}_{sd}_lam{lam}_dt{dt}_Nb{Nb}_Nk{Nk}_depth{max_depth}_T{T_total}"
    generate_husimi_anim(Qts=Qts, tlist=heom.tlist, theta=theta, phi=phi, T_drive=heom.T_drive, labels=labels, filename=filename)

if __name__ == "__main__":
    status = main()
    sys.exit(status)