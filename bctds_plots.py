import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.signal import windows
import os

def smooth_envelope(amplitude, fraction=0.02):
    N = len(amplitude)
    win_len = int(max(3, min(N-1, np.round(N * fraction))))
    if win_len % 2 == 0:
        win_len += 1
    kernel = np.ones(win_len) / win_len
    return np.convolve(amplitude, kernel, mode='same')

def compute_fft(sp_t, omega_d_vals, tlist, dt, n_time, fmax=0.1): 

    window_fn = windows.hann(n_time)
    window_rms = np.sqrt(np.mean(window_fn**2))
    N_pad = 2**13

    fft_data = []

    for idx, omega_d in enumerate(omega_d_vals):
        Splus_t = sp_t[idx, :]

        LO = np.exp(-1j * omega_d * tlist)
        demod = Splus_t * LO

        phi = np.angle(demod)
        amp = np.abs(demod)
        env = smooth_envelope(amp)

        phi_weighted = phi * env
        phi_win = phi_weighted * window_fn

        fft_vals = np.fft.rfft(phi_win, n=N_pad)
        fft_amp = np.abs(fft_vals) / window_rms

        fft_data.append(fft_amp)

    fft_data = np.array(fft_data)
    fft_freqs = np.fft.rfftfreq(N_pad, d=dt)

    # limit the plot to observe the features
    idx_max = np.searchsorted(fft_freqs, fmax) 

    fft_data = fft_data[:, :idx_max]
    fft_freqs = fft_freqs[:idx_max]

    return fft_freqs, fft_data  

# ------------- Plots -------------
def find_max(mats):
    res = np.max(mats[0])
    for i in range(1, len(mats)):
        res = max(res, np.max(mats[i]))
    return res


def find_min(mats):
    res = np.min(mats[0])
    for i in range(1, len(mats)):
        res = min(res, np.min(mats[i]))
    return res

def plot_exc_map(res_exc, omega_d_vals, tlist, labels, save=True, filename="exc_map"):
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

def generate_husimi_anim(Qts, tlist, theta, phi, T_drive, labels, filename="husimi_animation"):
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

        fig.suptitle(f"t = {t_now:.1f} ns | {phase}", color=color)

    ani = anim.FuncAnimation(fig=fig, func=update, frames=len(tlist), interval=30)
    os.makedirs("husimi_animation",exist_ok=True)
    ani.save(f"husimi_animation/{filename}.mp4", writer='ffmpeg', fps=10)
    print("Animation saved.")

def plot_phase_evolution(phases, tlist, tls_freqs, solver_name, save=True, filename="phase_plot"):
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
    n_plots = len(correlations) + 1

    fig, ax = plt.subplots(n_plots, 1, figsize=(16,4*n_plots))

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
        ax[i+1].set_ylabel(f"{corr_names[i].capitalize()}")
        ax[i+1].set_title(f"{corr_names[i].capitalize()} Over Time")
        ax[i+1].legend(leg_strs)
    plt.suptitle(f"{solver_name} Phase Difference and TLS Correlations", y=0.99)
    
    plt.tight_layout()

    if save:
        os.makedirs("correlation_plots", exist_ok=True)
        plt.savefig(f"correlation_plots/corrs_{filename}.png")

def plot_corr_J_sweep(corr_map, J_list, freq_ratios, solver_name, save=True, filename="corr_J_sweep"):
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
    
