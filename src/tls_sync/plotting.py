import matplotlib.pyplot as plt
import matplotlib.animation as anim
from .utils import find_max, find_min
import numpy as np
import os


def set_plot_format(scale=1, title_scale=1):
    """Increase the default font sizes used by the plotting functions."""
    plt.rcParams.update({
        "font.size": 12 * scale,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.titlesize": 16 * title_scale,
        "axes.labelsize": 14 * scale,
        "xtick.labelsize": 12 * scale,
        "ytick.labelsize": 12 * scale,
        "legend.fontsize": 11 * scale,
        "figure.titlesize": 16 * title_scale,
        "axes.titleweight": "normal",
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{braket}"
            r"\usepackage{amsmath}"
            r"\usepackage{sfmath}"
            r"\renewcommand{\familydefault}{\sfdefault}"
        ),
    })


def plot_exc_map(res_exc, omega_d_vals, tlist, labels, save=True, filename="exc_map"):
    """Plot excitation maps for one or more datasets.

    Parameters
    ----------
    res_exc : list of ndarray
        A list of 2D excitation arrays indexed by drive frequency and time.
    omega_d_vals : array_like
        Drive frequency values corresponding to the first axis of each excitation array.
    tlist : array_like
        Time values corresponding to the second axis of each excitation array.
    labels : list of str
        Title labels for each subplot.
    save : bool, optional
        Whether to save the figure to the ``bctds_figures`` directory (default is True).
    filename : str, optional
        The output filename (without extension) used when saving the figure.
    """
    n_plots = len(res_exc)
    # check shape
    for i in range(n_plots):
        assert(len(omega_d_vals) == len(res_exc[i]))
        assert(len(tlist) == len(res_exc[i][0]))
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]}
    fig, ax = plt.subplots(1, n_plots + 1, figsize=(6*n_plots,6), gridspec_kw=gridspec)

    # normalize cmaps
    vmin = find_min(res_exc)
    vmax = find_max(res_exc)

    # plot
    images = []
    
    for i in range(n_plots):
        images.append(ax[i].imshow(np.transpose(res_exc[i]),
                     extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                     origin='lower', aspect='auto', cmap='inferno',
                     vmin=vmin,
                     vmax=vmax))
        ax[i].set_title(r"$ \langle S_+S_- \rangle $ " + labels[i])
        ax[i].set_xlabel("Drive Frequency (GHz)")
        ax[i].set_ylabel("Time (ns)")

    # colorbar
    cb1 = fig.colorbar(images[-1], cax=ax[n_plots])
    cb1.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if save:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/{filename}.png")

def plot_sp_map(res_sp, omega_d_vals, tlist, labels, save=True, filename="sp_map"):
    """Plot sigma plus excitation maps for one or more datasets.

    Parameters
    ----------
    res_sp : list of ndarray
        A list of expectation values of total excitation indexed by drive frequency and time.
    omega_d_vals : array_like
        Drive frequency values corresponding to the first axis of each data array.
    tlist : array_like
        Time values corresponding to the second axis of each data array.
    labels : list of str
        Title labels for each subplot.
    save : bool, optional
        Whether to save the figure to the ``bctds_figures`` directory (default is True).
    filename : str, optional
        The output filename (without extension) used when saving the figure.
    """
    n_plots = len(res_sp)
    # check shape
    for i in range(n_plots):
        assert(len(omega_d_vals) == len(res_sp[i]))
        assert(len(tlist) == len(res_sp[i][0]))
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]}
    fig, ax = plt.subplots(1, n_plots + 1, figsize=(6*n_plots,6), gridspec_kw=gridspec)

    # normalize cmaps
    vmin = find_min(np.abs(res_sp))
    vmax = find_max(np.abs(res_sp))

    # plot
    images = []
    
    for i in range(n_plots):
        images.append(ax[i].imshow(np.transpose(np.abs(res_sp[i])),
                     extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                     origin='lower', aspect='auto', cmap='inferno',
                     vmin=vmin,
                     vmax=vmax))
        ax[i].set_title(r"$ | \langle S_+ \rangle | $ " + labels[i])
        ax[i].set_xlabel("Drive Frequency (GHz)")
        ax[i].set_ylabel("Time (ns)")

    # colorbar
    cb1 = fig.colorbar(images[-1], cax=ax[n_plots])
    cb1.set_label(r"$ | \langle \sigma^{+}\rangle | $ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if save:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/{filename}.png")

def plot_diff_map(res_exc, res_sp, omega_d_vals, tlist, labels, save=True, filename="diff_map"):
    """Plot difference maps comparing total and sigma plus excitation data across label pairs.

    Parameters
    ----------
    res_exc : list of ndarray
        A list of 2D excitation arrays indexed by drive frequency and time.
    res_sp : list of ndarray
        A list of complex spin expectation arrays indexed by drive frequency and time.
    omega_d_vals : array_like
        Drive frequency values corresponding to the first axis of each data array.
    tlist : array_like
        Time values corresponding to the second axis of each data array.
    labels : list of str
        Labels used to name each dataset and each difference plot.
    save : bool, optional
        Whether to save the figure to the ``bctds_figures`` directory (default is True).
    filename : str, optional
        The output filename (without extension) used when saving the figure.
    """

    assert(len(res_exc) == len(labels))
    assert(len(res_sp) == len(labels))
    
    # check shape
    for i in range(len(res_exc)):
        assert(len(omega_d_vals) == len(res_exc[i]))
        assert(len(tlist) == len(res_exc[i][0]))

    # compute diffs
    n_plots = 0
    exc_diffs = {}
    sp_diffs = {}
    for i in range(len(res_exc)):
        for j in range(i+1, len(res_exc)):
            exc_diffs[r"$ \langle S_+S_- \rangle $ Difference " + f"({labels[i]} - {labels[j]})"] = np.subtract(res_exc[i], res_exc[j])
            sp_diffs[r"$ | \langle S_+ \rangle | $ Difference " + f"({labels[i]} - {labels[j]})"] = np.subtract(np.abs(res_sp[i]), np.abs(res_sp[j]))
            n_plots += 1
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]}
    fig, ax = plt.subplots(2, n_plots + 1, figsize=(6*n_plots, 10), gridspec_kw=gridspec)

    # plot
    images = []
    for j, diffs in enumerate([exc_diffs, sp_diffs]):
        vmin = find_min(list(diffs.values()))
        vmax = find_max(list(diffs.values()))
        for i, key in enumerate(diffs.keys()):
            images.append(ax[j][i].imshow(np.transpose(diffs[key]),
                        extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                        origin='lower', aspect='auto', cmap='bwr',
                        vmin=vmin,
                        vmax=vmax))
            ax[j][i].set_title(key)
            ax[j][i].set_xlabel("Drive Frequency (GHz)")
            ax[j][i].set_ylabel("Time (ns)")

        # colorbar
        cb1 = fig.colorbar(images[-1], cax=ax[j][n_plots])
        if j == 0:
            cb1.set_label(r"$ \langle S_+S_- \rangle $ Difference", labelpad=14)
        else:
            cb1.set_label(r"$ | \langle S_+ \rangle | $ Difference", labelpad=14)
        plt.tight_layout()

    # save
    if save:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/{filename}.png")

def plot_fft_map(fft_freqs, fft_data, omega_d_vals, omega_tls, labels, save=True, filename="fft_map"):
    """Plot FFT magnitude maps across drive frequency and spectral frequency.

    Parameters
    ----------
    fft_freqs : list of array_like
        A list of frequency grids for each FFT dataset.
    fft_data : list of ndarray
        A list of 2D FFT magnitude arrays indexed by drive frequency and FFT frequency.
    omega_d_vals : array_like
        Drive frequency values corresponding to the first axis of each FFT array.
    omega_tls : array_like
        Bare two-level system eigenfrequencies plotted as reference lines.
    labels : list of str
        Title labels for each subplot.
    save : bool, optional
        Whether to save the figure to the ``bctds_figures`` directory (default is True).
    filename : str, optional
        The output filename (without extension) used when saving the figure.
    """
    n_plots = len(fft_freqs)
    # check shape
    for i in range(n_plots):
        assert(len(omega_d_vals) == len(fft_data[i]))
        assert(len(fft_freqs[i]) == len(fft_freqs[0]))
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]}
    fig, ax = plt.subplots(1, n_plots + 1, figsize=(6*n_plots, 6), gridspec_kw=gridspec)

    # normalize cmaps
    vmin = find_min(fft_data)
    vmax = find_max(fft_data)

    # plot
    images = []
    
    for i in range(n_plots):
        images.append(ax[i].imshow(fft_data[i].T,
                      extent=[omega_d_vals[0], omega_d_vals[-1],
                              fft_freqs[i][0], fft_freqs[i][-1]],
                      origin='lower', aspect='auto', cmap='Oranges',
                      vmin=vmin,
                      vmax=vmax))
        ax[i].set_title(f"FFT Data {labels[i]}")
        ax[i].set_xlabel("Drive Frequency (GHz)")
        ax[i].set_ylabel("FFT Frequency (GHz)")
        # add bare eigenfrequencies as vertical lines
        ax[i].vlines(x=omega_tls, color='black', ymin=fft_freqs[i][0], ymax=fft_freqs[i][-1], linestyle='--', linewidth=0.9)

    # colorbar
    cb1 = fig.colorbar(images[-1], cax=ax[n_plots])
    cb1.set_label(r"$|\mathrm{FFT}(\phi)|$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if save:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/{filename}.png")

def plot_husimi_snapshots(Qts, tlist, snapshots, theta, phi, labels, omega_d, tls_freqs, filename="husimi_snapshots", save=True):
    """Plot selected Husimi Q-function snapshots for multiple solvers.

    Parameters
    ----------
    Qts : list of ndarray
        Time-series Husimi Q-function arrays for one or more datasets. Each element must
        have shape (len(tlist), len(theta), len(phi)).
    tlist : array_like
        Time values corresponding to the first axis of each Q-function dataset.
    snapshots : array_like
        Time values at which snapshots should be extracted and plotted.
    theta : array_like
        Polar angle grid for the Husimi function.
    phi : array_like
        Azimuthal angle grid for the Husimi function.
    labels : list of str, optional
        Title labels for each solver subplot.
    method : str, optional
        Name of the simulation method used in the figure title.
    omega_d : float, optional
        Drive frequency value used in the figure title.
    filename : str, optional
        Output filename (without extension) used when saving the figure.
    save : bool, optional
        Whether to save the figure to the ``husimi_snapshots`` directory.
    cmap : str, optional
        Matplotlib colormap used for the plots.

    Returns
    -------
    tuple
        The matplotlib figure and axes objects for the generated snapshot plot.
    """
    snapshots = np.atleast_1d(np.asarray(snapshots))

    for i in range(len(Qts)):
        assert len(Qts[i]) == len(tlist)
        assert np.shape(Qts[i][0]) == (len(theta), len(phi))

    snapshot_indices = np.array([np.argmin(np.abs(tlist - snapshot)) for snapshot in snapshots], dtype=int)

    n_plots = len(Qts)
    n_snapshots = len(snapshots)
    fig = plt.figure(figsize=(8 * n_plots + 2, 5 * n_snapshots + 1.5))
    gs = fig.add_gridspec(n_snapshots + 1, n_plots, height_ratios=[1] * n_snapshots + [0.12])

    axes = np.empty((n_snapshots, n_plots), dtype=object)
    for row_idx in range(n_snapshots):
        for col_idx in range(n_plots):
            axes[row_idx, col_idx] = fig.add_subplot(gs[row_idx, col_idx])

    cax = fig.add_subplot(gs[n_snapshots, :])

    vmax = find_max(Qts)
    vmin = find_min(Qts)

    images = []
    for row_idx, snap_idx in enumerate(snapshot_indices):
        for col_idx in range(n_plots):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                Qts[col_idx][snap_idx],
                origin='lower',
                extent=[phi[0], phi[-1], theta[0], theta[-1]],
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{labels[col_idx]} @ t={tlist[snap_idx]:.1f}")
            ax.set_xlabel(r"$\phi$")
            ax.set_ylabel(r"$\theta$")
            ax.set_aspect('auto')
            images.append(image)

    fig.colorbar(images[-1], cax=cax, orientation="horizontal").set_label(r"$Q(\theta,\phi)$")
    fig.suptitle(f"Husimi Q Snapshots | Drive {omega_d} GHz\n"
                 + f"TLSs: {tls_freqs[0]} GHz and {tls_freqs[1]} GHz")
    plt.tight_layout()

    if save:
        os.makedirs("husimi_snapshots", exist_ok=True)
        plt.savefig(f"husimi_snapshots/{filename}.png")

    return fig, axes


def husimi_snapshots(Qts, tlist, snapshots, theta, phi, labels=None, method=None, omega_d=None, filename="husimi_snapshots", save=True, cmap="inferno"):
    """Backward-compatible wrapper for plotting selected Husimi Q snapshots."""
    return plot_husimi_snapshots(
        Qts=Qts,
        tlist=tlist,
        snapshots=snapshots,
        theta=theta,
        phi=phi,
        labels=labels,
        method=method,
        omega_d=omega_d,
        filename=filename,
        save=save,
        cmap=cmap,
    )


def plot_husimi_anim(Qts, tlist, theta, phi, T_drive, labels, method, omega_d, filename="husimi_animation"):
    """Generate and save a Husimi Q-function animation.

    Parameters
    ----------
    Qts : list of ndarray
        Time-series Husimi Q-function arrays for one or more datasets. Each element must
        have shape (len(tlist), len(theta), len(phi)).
    tlist : array_like
        List of time points corresponding to the first axis of each Q-function dataset.
    theta : array_like
        Polar angle grid for the Husimi function.
    phi : array_like
        Azimuthal angle grid for the Husimi function.
    T_drive : float
        Drive duration used to label frames as DRIVE ON/OFF.
    labels : list of str
        Title labels for each subplot.
    method : str
        Name of the simulation method used in the animation title.
    omega_d : float
        Drive frequency value used in the animation title.
    filename : str, optional
        Output filename (without extension) used when saving the animation.
    """
    for i in range(len(Qts)):
        assert len(Qts[i]) == len(tlist)
        assert np.shape(Qts[i][0]) == (len(theta), len(phi))

    print("Generating Husimi animaton...")
    n_plots = len(Qts)
    # check shape
    for i in range(n_plots):
        assert len(Qts[0]) == len(Qts[i])
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]} 
    fig, ax = plt.subplots(1, n_plots + 1, figsize=(6 * n_plots, 4), gridspec_kw=gridspec)

    vmax = find_max(Qts)
    vmin = find_min(Qts)
    t_idx = 0

    # add labels
    ax[0].set_ylabel(r"$\theta$")
    # add data
    images = []
    
    for i in range(len(Qts)):
        ax[i].set_title(labels[i])
        ax[i].set_xlabel(r"$\phi$")
        ax[i].set_aspect('auto')

        images.append(
            ax[i].imshow(
                Qts[i][t_idx],
                origin='lower',
                extent=[phi[0], phi[-1], theta[0], theta[-1]],
                cmap='inferno',
                vmin=vmin, vmax=vmax
            )
        )

    # colorbar
    cb1 = fig.colorbar(images[-1], cax=ax[n_plots])
    cb1.set_label(r"$Q(\theta,\phi)$")
    plt.tight_layout()

    def update(frame):
        t_idx = frame

        for i in range(len(Qts)):
            # clear frame
            ax[i].clear()
            if i == 0:
                ax[i].set_ylabel(r"$\theta$")
            ax[i].set_title(labels[i])
            ax[i].set_xlabel(r"$\phi$")

            ax[i].imshow(
                Qts[i][t_idx],
                origin='lower',
                extent=[phi[0], phi[-1], theta[0], theta[-1]],
                cmap='inferno',
                vmin=vmin, vmax=vmax
            )

        # label time of the frame
        t_now = tlist[t_idx]

        if t_now <= T_drive:
            phase = "DRIVE ON"
            color = 'green'
        else:
            phase = "DRIVE OFF"
            color = 'red'

        fig.suptitle(f"Drive: {omega_d} GHz, Method: {method}, t = {t_now:.1f} ns | {phase}", color=color)

    ani = anim.FuncAnimation(fig=fig, func=update, frames=len(tlist), interval=30)
    os.makedirs("husimi_animation",exist_ok=True)
    ani.save(f"husimi_animation/{filename}.mp4", writer='ffmpeg', fps=10)
    print("Animation saved.")

def plot_phase_evolution(phases, tlist, tls_freqs, solver_name, save=True, filename="phase_plot"):
    """Plot TLS phase evolution and pairwise phase differences over time.

    Parameters
    ----------
    phases : list of ndarray
        Phase trajectories for each TLS.
    tlist : array_like
        Time values corresponding to the phase trajectories.
    tls_freqs : array_like
        TLS frequencies used for legend labels.
    solver_name : str
        Name of the solver used, shown in the figure title.
    save : bool, optional
        Whether to save the plot to the ``phase_plots`` directory (default is True).
    filename : str, optional
        The output filename (without extension) used when saving the figure.
    """
    fig, ax = plt.subplots(1, 2, figsize=(6*2,6))

    for i in range(len(phases)):
        ax[0].plot(tlist, phases[i])

    ax[0].legend([f"TLS {tls_freqs[i]}" for i in range(len(phases))])
    ax[0].set_xlabel("Time [us]")
    ax[0].set_ylabel("Phase (rad.)")
    ax[0].set_title("Phase Time Evolution")

    # plot differences
    leg_strs = []
    for i in range(len(phases)):
        for j in range(i+1, len(phases)):
            ax[1].plot(tlist, phases[i] - phases[j])
            leg_strs.append(f"TLS {tls_freqs[i]} - TLS {tls_freqs[j]}")

    ax[1].set_xlabel("Time [us]")
    ax[1].set_ylabel("Phase (rad.)")
    ax[1].set_title("Phase Differences Over Time")
    ax[1].legend(leg_strs)
    fig.suptitle(f"{solver_name} Phase Time Evolution")
    plt.tight_layout()

    if save:
        os.makedirs("phase_plots", exist_ok=True)
        plt.savefig(f"phase_plots/{filename}.png")

def plot_correlations(correlations, tlist, solver_name, corr_name, save=True, filename="correlations_plot"):
    """Plot correlation time series for one or more TLS pairs.

    Parameters
    ----------
    correlations : dict
        Mapping of pair labels to correlation trajectories.
    tlist : array_like
        Time values corresponding to the correlation series.
    solver_name : str
        Name of the solver used, shown in the figure title.
    corr_name : str
        Correlation type name used in axis labels and title.
    save : bool, optional
        Whether to save the plot to the ``correlation_plots`` directory (default is True).
    filename : str, optional
        The output filename (without extension) used when saving the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6,6))

    lgd_strs = []
    for label, C_t in correlations.items():
        assert len(C_t) == len(tlist)
        ax.plot(tlist, C_t)
        lgd_strs.append(f"{label}")
    
    ax.set_xlabel("Time [us]")
    ax.set_ylabel(f"{corr_name}")
    ax.set_title(f"{corr_name} Over Time ({solver_name})")
    ax.legend(lgd_strs)
    plt.tight_layout()

    if save:
        os.makedirs("correlation_plots", exist_ok=True)
        plt.savefig(f"correlation_plots/{corr_name}_{filename}.png")

def plot_phase_corr_evolution(phases, correlations, corr_names, tlist, tls_freqs, solver_name, filename, save=True):
    """Plot phase differences and multiple correlation evolutions in one figure.

    Parameters
    ----------
    phases : list of ndarray
        Phase trajectories for each TLS.
    correlations : list of dict
        A list of correlation dictionaries, one per correlation type.
    corr_names : list of str
        Names of the correlation types in ``correlations``.
    tlist : array_like
        Shared time values for phase and correlation plots.
    tls_freqs : array_like
        TLS frequencies used for legend labels.
    solver_name : str
        Solver name shown in the figure titles.
    filename : str
        Output filename (without extension) used when saving the figure.
    save : bool, optional
        Whether to save the figure to the ``correlation_plots`` directory (default is True).
    """
    n_plots = len(correlations) + 1

    fig, ax = plt.subplots(n_plots, 1, figsize=(16,4*n_plots))

    y_limits = {
        "connected": [-1.1, 1.1],
        "pearson": [-1.1, 1.1],
        "plv": [-0.1, 1.1],
        "entropy": [-0.1, 1.1]
    }

    labels = {
        "connected": "Connected Quantum Correlation",
        "pearson": "Pearson Correlation",
        "plv": "Phase Locking Value",
        "entropy": "Mutual Information"
    }

    # plot phase diffs
    leg_strs = []
    for k in range(len(tls_freqs)):
        for j in range(k+1, len(tls_freqs)):
            ax[0].plot(tlist, phases[k] - phases[j])
            leg_strs.append(f"TLS {tls_freqs[k]} - TLS {tls_freqs[j]} ({solver_name})")

    ax[0].set_xlabel("Time [us]")
    ax[0].set_ylabel("Phase (rad.)")
    ax[0].set_title("Phase Differences Over Time")
    ax[0].legend(leg_strs)

    for i in range(len(correlations)):
        # plot correlations
        leg_strs = []
        for label, C_t in correlations[i].items():
            assert len(C_t) == len(tlist)
            ax[i+1].plot(tlist, C_t)
            leg_strs.append(label)

        ax[i+1].set_xlabel("Time [us]")
        ax[i+1].set_ylabel(f"{labels[corr_names[i]]}")
        ax[i+1].set_title(f"{labels[corr_names[i]]} Over Time")
        ax[i+1].legend(leg_strs)
        ax[i+1].set_ylim(y_limits[corr_names[i]])
    plt.suptitle(f"{solver_name} Phase Difference and TLS Correlations", y=0.99)
    
    plt.tight_layout()

    if save:
        os.makedirs("correlation_plots", exist_ok=True)
        plt.savefig(f"correlation_plots/corrs_{filename}.png")

def plot_corr_J_sweep(corr_map, J_list, freq_ratios, solver_name, save=True, filename="corr_J_sweep"):
    """Plot a correlation sweep as a function of coupling strength and frequency ratio.

    Parameters
    ----------
    corr_map : ndarray
        2D correlation map indexed by J strength and frequency ratio.
    J_list : array_like
        Coupling strength values for the vertical axis.
    freq_ratios : array_like
        Frequency ratio values for the horizontal axis.
    solver_name : str
        Name of the solver used, shown in the plot title.
    save : bool, optional
        Whether to save the figure to the ``correlation_plots`` directory (default is True).
    filename : str, optional
        The output filename (without extension) used when saving the figure.
    """
    gridspec = {'width_ratios': [1, 0.1]} 
    fig, ax = plt.subplots(1, 2, figsize=(6, 8), gridspec_kw=gridspec)

    assert len(corr_map) == len(J_list)
    assert len(corr_map[0]) == len(freq_ratios)

    im = ax[0].imshow(corr_map,
                    extent=[freq_ratios[0], freq_ratios[-1],
                            J_list[0], J_list[-1]],
                    origin='lower', aspect='auto', cmap='inferno',
                    vmin=-1,
                    vmax=1)
    
    ax[0].set_title(f"Correlation Sweep ({solver_name})")
    ax[0].set_xlabel(r"$ \omega_2 / \omega_1$")
    ax[0].set_ylabel("J")
    
    # colorbar
    cb1 = fig.colorbar(im, cax=ax[1])
    cb1.set_label(r"$ C_{12} $", labelpad=14)
    plt.tight_layout()

    # save
    if save:
        os.makedirs("correlation_plots",exist_ok=True)
        plt.savefig(f"correlation_plots/{filename}.png")
    
