import matplotlib.pyplot as plt
import numpy as np
import sys

from tls_sync import HEOM, Lindblad
from tls_sync.plotting import set_plot_format


def main():
    omega1 = 3.75
    J = 0.02
    Omega_amp = 0.1
    g = 0.02
    gamma_bath = 0.05
    T = 0.5
    Nk = 3
    max_depth = 50
    T_total = 400
    T_drive = 100.0
    dt = 0.1
    n_tls = 2
    n_freqs = 5

    sd_type = "drude"
    ohmicity = None
    window_size = 150

    freq_list = np.linspace(3.0, 5.0, n_freqs)
    if omega1 not in freq_list:
        idx = np.searchsorted(freq_list, omega1)
        freq_list = np.concatenate((freq_list[:idx], [omega1], freq_list[idx:]))
        n_freqs += 1

    heatmaps = {}
    for omega2 in freq_list:
        tls_freqs = [omega1, omega2]
        heom = HEOM(
            tls_freqs=tls_freqs,
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
            ohmicity=ohmicity,
        )
        mark = Lindblad(
            tls_freqs=tls_freqs,
            J=J,
            Omega_amp=Omega_amp,
            lam=g,
            T=T,
            T_total=T_total,
            T_drive=T_drive,
            dt=dt,
            n_tls=n_tls,
            n_freqs=n_freqs,
        )

        for solver in (mark, heom):
            solver_name = solver.get_name()
            if solver_name not in heatmaps:
                heatmaps[solver_name] = []

            _, _, states = solver.run(store_states=True)
            corr_values = []
            for drive_idx in range(len(freq_list)):
                corr_values.append(
                    solver.final_pearson_from_states(
                        list(states[drive_idx]),
                        window_size=window_size,
                    )
                )
            heatmaps[solver_name].append(np.asarray(corr_values))

    set_plot_format(2, 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (solver_name, heatmap) in zip(axes, heatmaps.items()):
        data = np.vstack(heatmap)
        image = ax.imshow(
            data,
            origin="lower",
            extent=[freq_list[0], freq_list[-1], freq_list[0], freq_list[-1]],
            aspect="auto",
            cmap="inferno",
            vmin=-1,
            vmax=1,
        )
        ax.set_title(solver_name)
        ax.set_xlabel(r"Drive frequency $\omega_d$")
        ax.set_ylabel(r"TLS frequency $\omega_2$")
        fig.colorbar(image, ax=ax, orientation="horizontal")

    plt.tight_layout()
    plt.savefig("pearson_heatmap.png")
    plt.show()



if __name__ == "__main__":
    status = main()
    sys.exit(status)
