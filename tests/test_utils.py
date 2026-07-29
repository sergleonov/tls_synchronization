"""Unit tests for tls_sync.utils (pure NumPy/SciPy; no solver needed)."""
import numpy as np
import pytest

from tls_sync import utils


def test_find_max_and_min():
    mats = [np.array([[1.0, 2.0]]), np.array([[5.0, -3.0]]), np.array([[0.0]])]
    assert utils.find_max(mats) == 5.0
    assert utils.find_min(mats) == -3.0


def test_find_max_min_reject_empty():
    with pytest.raises(ValueError):
        utils.find_max([])
    with pytest.raises(ValueError):
        utils.find_min([])


def test_smooth_envelope_preserves_length_and_interior():
    x = np.ones(50) * 3.0
    env = utils.smooth_envelope(x)
    assert env.shape == x.shape
    # boxcar with mode='same' tapers at the edges; interior is preserved
    assert np.allclose(env[1:-1], 3.0)
    assert env.max() <= 3.0 + 1e-9


def test_compute_fft_shapes_and_frequency_cap():
    n_drive, n_time, dt, fmax = 2, 64, 0.5, 0.1
    tlist = np.arange(n_time) * dt
    omega_d_vals = np.array([3.0, 5.0])
    rng = np.random.default_rng(0)
    sp_t = rng.random((n_drive, n_time)) + 1j * rng.random((n_drive, n_time))

    fft_freqs, fft_data = utils.compute_fft(sp_t, omega_d_vals, tlist, dt, n_time, fmax=fmax)

    assert fft_data.shape == (n_drive, len(fft_freqs))
    assert np.all(fft_freqs <= fmax + 1e-12)
    assert np.all(fft_freqs >= 0.0)
    assert np.isfinite(fft_data).all()
    assert np.all(fft_data >= 0.0)          # magnitudes


def test_compute_fft_is_deterministic():
    n_time, dt = 32, 0.5
    tlist = np.arange(n_time) * dt
    omega = np.array([4.0])
    sp = (np.sin(0.3 * tlist) + 1j * np.cos(0.3 * tlist))[None, :]
    f1, d1 = utils.compute_fft(sp, omega, tlist, dt, n_time)
    f2, d2 = utils.compute_fft(sp, omega, tlist, dt, n_time)
    assert np.array_equal(f1, f2)
    assert np.array_equal(d1, d2)
