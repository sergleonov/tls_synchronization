from scipy.signal import windows
import numpy as np

def smooth_envelope(amplitude, fraction=0.02):
    N = len(amplitude)
    win_len = int(max(3, min(N-1, np.round(N * fraction))))
    if win_len % 2 == 0:
        win_len += 1
    kernel = np.ones(win_len) / win_len
    return np.convolve(amplitude, kernel, mode='same')

def compute_fft(sp_t, omega_d_vals, tlist, dt, n_time, fmax=0.1):
    """Compute demodulated FFT amplitudes from sigma plus excitation time series.

    Parameters
    ----------
    sp_t : ndarray
        Complex spin coherence data with shape (n_drive_freqs, n_time).
    omega_d_vals : array_like
        Drive frequency values corresponding to the first axis of ``sp_t``.
    tlist : array_like
        Time points corresponding to the second axis of ``sp_t``.
    dt : float
        Time step used for FFT frequency scaling.
    n_time : int
        Number of time points in each time series.
    fmax : float, optional
        Maximum FFT frequency to retain for plotting (default is 0.1).

    Returns
    -------
    fft_freqs : ndarray
        Frequency grid for the FFT output.
    fft_data : ndarray
        FFT amplitude data with shape (n_drive_freqs, n_fft_bins).
    """

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

def find_max(mats):
    """Return the maximum element among a list of matrices.

    Parameters
    ----------
    mats : list of ndarray
        A list of NumPy matrices whose maximum values are compared.

    Returns
    -------
    float
        The maximum value found in any of the input matrices.
    """
    res = np.max(mats[0])
    for i in range(1, len(mats)):
        res = max(res, np.max(mats[i]))
    return res


def find_min(mats):
    """Return the minimum element among a list of matrices.

    Parameters
    ----------
    mats : list of ndarray
        A list of NumPy matrices whose minimum values are compared.

    Returns
    -------
    float
        The minimum value found in any of the input matrices.
    """
    res = np.min(mats[0])
    for i in range(1, len(mats)):
        res = min(res, np.min(mats[i]))
    return res