import numpy as np
import matplotlib.pyplot as plt
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

