import numpy as np
import sys
from tls_sync import HEOM, Lindblad, TieredSolver, TEMPO
from tls_sync.plotting import plot_husimi_snapshots, set_plot_format

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
    T_total = 300
    T_drive = 100.0
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
    method = "ptrace"

    omega_d = 3.45
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 80)
    
    Qts = []
    labels = []
    for solver in solvers:
        Qts.append(solver.husimi_sim(omega_d, theta, phi, method=method, tls_idx=tls_idx))
        labels.append(solver.get_name())

    filename = f"husimi_snap_omega_d{omega_d}_tls_{tls_freqs[0]}_{tls_freqs[1]}_mark_heom"

    snapshots = np.linspace(0, T_total, 6)
    set_plot_format(scale=1.5, title_scale=1.5)
    plot_husimi_snapshots(
        Qts=Qts,
        tlist=heom.tlist,
        snapshots=snapshots,
        theta=theta,
        phi=phi,
        labels=labels,
        method=method,
        omega_d=omega_d,
        filename=filename,
    )


if __name__ == "__main__":
    status = main()
    sys.exit(status)