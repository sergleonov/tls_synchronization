import numpy as np
import matplotlib.pyplot as plt

from tls_sync import HeomSolver, MarkovianSolver, SemiclassicalSolver, TempoSolver
from tls_sync.model import TLSChainModel, Bath, TLSCavityModel, SemiclassicalCavityModel
from tls_sync.plotting import plot_bloch_sphere

def main():
    # model params
    tls_freq = [1.0]  # GHZ frequency of the bath modes (normalized)
    Omega_amp = 60*tls_freq[0]  # GHz frequency of the driving field
    T_drive = 3.0

    # bath params
    cutoff = tls_freq[0] # GHz cut off frequency
    lam = tls_freq[0]  # coupling strength
    T = 0.5  # K temperature

    # solver params
    T_total = T_drive
    dt = T_drive / 1e3

    omegad = Omega_amp/2.4048  # GHz frequency of the driving field (resonant)

    # cavity mode
    omega_c = 4.0

    # decay
    dissipation = 0.002

    bath = Bath(
        sd_type="drude",
        # ohmicity=3.0,
        coupling=lam,
        cutoff=cutoff,
        temperature=T
    )

    # init HEOM/Lindblad model
    model = TLSChainModel(
        omega_tls=tls_freq,
        Omega_amp=Omega_amp,
        T_drive=T_total,
        n_tls=1,
        bath=bath
    )

    # init tiered
    tiered_model = TLSCavityModel(
            omega_tls=tls_freq,
            Omega_amp=Omega_amp,
            T_drive=T_drive,
            omega_c=omega_c,
            g=lam,                       # mode coupling
            Nb=10,                     # fock number
            gamma=dissipation,         # tls relaxation
            temperature=T,
            n_tls=1,
    )

    # init semiclassical
    semiclassical_model = SemiclassicalCavityModel(
        omega_tls=tls_freq,
        Omega_amp=Omega_amp,
        T_drive=T_drive,
        omega_c=omega_c,
        g=lam,
        kappa=0.001,               # cavity leakage
        eta=0.0,
        gamma=dissipation,         # TLS relaxation rate
        gamma_phi=dissipation / 10.0,
        temperature=T,
        n_tls=1,
    )

    # build solvers
    markov = MarkovianSolver(model, T_total=T_total, dt=dt)
    tempo = TempoSolver(model, T_total=T_total, dt=dt, tcut=2.5, epsrel=1e-4)
    heom = HeomSolver(model, T_total=T_total, dt=dt, Nk=3)
    tiered = MarkovianSolver(tiered_model, T_total=T_total, dt=dt)
    semi = SemiclassicalSolver(semiclassical_model, T_total=T_total, dt=dt/10.0, dt_output=dt)

    # expectation operators (identical for both)
    e_ops = [sum(markov.ops.sx), sum(markov.ops.sy),sum(markov.ops.sz)]
    e_ops_tempo = [sum(tempo.ops.sx), sum(tempo.ops.sy),sum(tempo.ops.sz)]
    e_ops_tier = [sum(tiered.ops.sx), sum(tiered.ops.sy),sum(tiered.ops.sz)]
    e_ops_semi = [sum(semi.ops.sx), sum(semi.ops.sy),sum(semi.ops.sz)]

    import qutip as qt
    psis = [
        (qt.basis(2,1)+qt.basis(2,0)).unit(),
        (qt.basis(2,1)+2.*qt.basis(2,0)).unit(),
        (2.*qt.basis(2,1)+qt.basis(2,0)).unit(),
        ((2j+.3)*qt.basis(2,1) + np.sqrt(2*2 + .3*.3)*qt.basis(2,0)).unit(),
        ((2j+.3)*qt.basis(2,0) + np.sqrt(2*2 + .3*.3)*qt.basis(2,1)).unit()
    ]
    fig, axs= None, None

    for psi in psis:
        # psi = qt.tensor(psi, qt.basis(10,0))
        rho0 = qt.ket2dm(psi)
        # run dynamics
        # markov_res = markov.single_run(omegad, rho0=rho0, e_ops=e_ops)
        # tempo_res = tempo.single_run(omegad, rho0=rho0.full(), e_ops=e_ops_tempo)
        heom_res = heom.single_run(omegad, rho0=rho0, e_ops=e_ops)
        # tiered_res = tiered.single_run(omegad, rho0=rho0, e_ops=e_ops_tier)
        # semi_res = semi.single_run(omegad, rho0=rho0.full(), e_ops=e_ops_semi)

        fig, axs = plot_bloch_sphere([heom_res], title="HEOM (Drude) Varying Initial State Dynamics", tmax=T_total, fig=fig, axs=axs)

    plt.tight_layout()
    plt.savefig("figs/cycle_figures/bloch_initial_states_heom_drude.png", dpi=300)


if __name__ == "__main__":
    main()
