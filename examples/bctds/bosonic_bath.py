from qutip import Bloch
import numpy as np
import matplotlib.pyplot as plt

from tls_sync import HeomSolver, MarkovianSolver, SemiclassicalSolver, TempoSolver
from tls_sync.model import TLSChainModel, Bath, TLSCavityModel, SemiclassicalCavityModel

# -----------------Plotting the Bloch sphere-----------------
def plot_bloch_sphere(results, tmax, labels):

    # plot two bloch spheres
    fig = plt.figure(figsize=(12, 6))
    axs = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]

    fig.suptitle("TLS Dynamics on the Bloch Sphere", fontsize=16)

    for idx, result in enumerate(results):
        # extract data
        xs = result.expectations[0][0]
        ys = result.expectations[1][0]
        zs = result.expectations[2][0]
        t = result.times

        b = Bloch(fig, axs[idx])
        b.view = [-45, 30]
        cmap = plt.get_cmap('inferno')  # choose a colormap
        colors = cmap(np.divide(t, tmax))  # normalize time to [0, 1] for color mapping

        for i in range(len(xs) - 1):
            segment = [[xs[i], xs[i+1]],
                    [ys[i], ys[i+1]],
                    [zs[i], zs[i+1]]]
            b.add_points(segment, meth='l') # 'l' tells QuTiP to render as a line
        colors[0] = [0, 0.5, 1, 1] # color first point differently for better visibility
        b.point_color = colors
        b.render()

        axs[idx].set_title(labels[idx], fontsize=14)

def main():
    # model params
    tls_freq = [3.75]  # GHZ frequency of the bath modes (normalized)
    Omega_amp = 20*tls_freq[0]  # GHz frequency of the driving field
    T_drive = 30/tls_freq[0]

    # bath params
    cutoff = tls_freq[0] # GHz cut off frequency
    lam = 1.0  # coupling strength
    T = 0.5  # K temperature

    # solver params
    T_total = T_drive
    dt = T_drive / 1e4

    omegad = Omega_amp/2.4048  # GHz frequency of the driving field (resonant)

    # cavity mode
    omega_c = 4.0

    # decay
    dissipation = 0.002

    bath = Bath(
        sd_type="power",
        ohmicity=1.0,
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
    heom = HeomSolver(model, T_total=T_total, dt=dt, Nk=6)
    tiered = MarkovianSolver(tiered_model, T_total=T_total, dt=dt)
    semi = SemiclassicalSolver(semiclassical_model, T_total=T_total, dt=dt/10.0, dt_output=dt)

    # expectation operators (identical for both)
    e_ops = [sum(markov.ops.sx), sum(markov.ops.sy),sum(markov.ops.sz)]
    e_ops_tempo = [sum(tempo.ops.sx), sum(tempo.ops.sy),sum(tempo.ops.sz)]
    e_ops_tier = [sum(tiered.ops.sx), sum(tiered.ops.sy),sum(tiered.ops.sz)]
    e_ops_semi = [sum(semi.ops.sx), sum(semi.ops.sy),sum(semi.ops.sz)]

    # run dynamics
    markov_res = markov.single_run(omegad, e_ops=e_ops)
    tempo_res = tempo.single_run(omegad, e_ops=e_ops_tempo)
    # heom_res = heom.single_run(omegad, e_ops=e_ops)
    # tiered_res = tiered.single_run(omegad, e_ops=e_ops_tier)
    # semi_res = semi.single_run(omegad, e_ops=e_ops_semi)

    plot_bloch_sphere([markov_res, tempo_res], tmax=T_total, labels=["Markovian", "TEMPO"])
    plt.tight_layout()
    plt.savefig("figs/cycle_figures/bloch.png")


if __name__ == "__main__":
    main()
