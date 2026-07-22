import numpy as np
import sys
from tls_sync import HEOM, Lindblad, TieredSolver, TEMPO
from tls_sync.plotting import generate_husimi_anim

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
    tcut = 2.5
    epsrel = 1e-4

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
    
    solvers = [mark, heom]
    
    tls_idx = 0
    method = "diff"

    omega_d = 3.75
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 80)
    
    Qts = []
    labels = []
    for solver in solvers:
        Qts.append(solver.husimi_sim(omega_d, theta, phi, method=method, tls_idx=tls_idx))
        labels.append(solver.get_name())

    sd = ""
    if sd_type == "power":
        sd = sd_type + f"_ohm{ohmicity}"
    filename = f"sim_omega_d{omega_d}_{method}_J{J}_Om{Omega_amp}_omega_c{omega_c}_cutoff{gamma_bath}_g{g}_{sd}_lam{lam}_dt{dt}_Nb{Nb}_Nk{Nk}_depth{max_depth}"
    generate_husimi_anim(Qts=Qts, tlist=heom.tlist, theta=theta, phi=phi, T_drive=heom.T_drive, labels=labels, method=method, omega_d=omega_d, filename=filename)

if __name__ == "__main__":
    status = main()
    sys.exit(status)