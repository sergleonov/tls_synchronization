#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from Juan Sebastián Salcedo-Gallo 
adjusted to compare markovian vs non-markovian dynamics


generate_two_panel_data.py

Generates data for a 2-column figure:
  Column 0: Amp-1
  Column 1: Amp-2

Each column has 4 stacked rows:
  1) Tail metric (post-pulse tail area)
  2) Ringdown heatmap (population vs time and omega_d)
  3) Phase-V map (FFT of phase weighted by envelope)
  4) Floquet quasi-energies (from propagator over pulse duration)

Key update (per request):
  The phase FFT is computed ONLY on the post-pulse region (t > t_drive_ns).
  The drive region is excluded from the FFT entirely.

Additional update:
  The Floquet quasienergy wrapping is now done robustly by:
    * computing principal eigenphases from the propagator at each omega_d
    * tracking branches across omega_d by nearest wrapped continuation
  This avoids fake overlaps / branch swaps in the high-amplitude regime.

Outputs a single NPZ file with all arrays, so plotting can be done separately.

Run:
  python3 generate_two_panel_data.py
"""

import os
import argparse
import itertools
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from scipy.signal import windows
from tqdm import tqdm

from qutip import (
    QobjEvo, tensor, qeye, sigmax, sigmaz, sigmam, sigmap,
    mesolve, propagator
)
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver

# ---------------------- Defaults (your regime) ----------------------
DEFAULTS = dict(
    omega1=3.75,
    omega2=3.82,
    j_coupling=0.02,
    gamma_collective=0.001,
    t_drive_ns=30.0,
    t_total_ns=100.0,
    dt_ns=0.5,
    omega_d_min=3.0,
    omega_d_max=4.5,
    n_omega_d=400,
    amp1=0.2,
    amp2=0.2,
    fft_view_mhz=100.0,
    n_pad=2**13,
    tail_start_buffer_ns=5.0,
    tail_baseline_fraction=0.25,
    freq_smooth_points=9,
    outdir="data",
    Nk = 3,
    max_depth = 5,
    T = 0.5,
    gamma_bath = 3.75,
)
# ---------------------- Worker globals ----------------------
WG = {}

def _build_qutip_objects(omega1, omega2, j_coupling, gamma_collective):
    sx1 = tensor(sigmax(), qeye(2));  sx2 = tensor(qeye(2), sigmax())
    sz1 = tensor(sigmaz(), qeye(2));  sz2 = tensor(qeye(2), sigmaz())
    sp1 = tensor(sigmap(),  qeye(2)); sp2 = tensor(qeye(2), sigmap())
    sm1 = tensor(sigmam(),  qeye(2)); sm2 = tensor(qeye(2), sigmam())

    sp_tot, sm_tot = sp1 + sp2, sm1 + sm2
    pop_op = sp_tot * sm_tot
    sx_tot = sx1 + sx2

    h_static = 0.5 * omega1 * sz1 + 0.5 * omega2 * sz2 + j_coupling * sx1 * sx2
    psi0 = h_static.eigenstates()[1][0]  # keep your original choice
    c_ops = [np.sqrt(gamma_collective) * sm_tot]

    return dict(
        sx_tot=sx_tot,
        sp_tot=sp_tot,
        pop_op=pop_op,
        h_static=h_static,
        psi0=psi0,
        c_ops=c_ops,
        Q_bath=sx1 + sx2,
    )

def _worker_init(omega1, omega2, j_coupling, gamma_collective):
    global WG
    WG = _build_qutip_objects(omega1, omega2, j_coupling, gamma_collective)

def _nearest_wrapped_phase(phi, phi_ref):
    """
    Return phi + 2*pi*m closest to phi_ref.
    """
    m = np.round((phi_ref - phi) / (2.0 * np.pi))
    return phi + 2.0 * np.pi * m

def _best_phase_permutation(curr_phases, prev_phases):
    """
    Given current principal phases in (-pi, pi] and previously tracked
    unwrapped phases, choose the permutation that minimizes total wrapped jump.

    Since this is a 4-level system, brute force over 4! = 24 permutations
    is cheap and robust.
    """
    best_perm = None
    best_unwrapped = None
    best_cost = np.inf

    n = len(curr_phases)
    for perm in itertools.permutations(range(n)):
        trial = np.array([curr_phases[j] for j in perm], dtype=float)
        trial_unwrapped = np.array(
            [_nearest_wrapped_phase(trial[k], prev_phases[k]) for k in range(n)],
            dtype=float
        )
        cost = float(np.sum((trial_unwrapped - prev_phases) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
            best_unwrapped = trial_unwrapped

    return best_perm, best_unwrapped

def track_quasi_branches_from_principal_phases(phases_principal, t_drive_ns):
    """
    Track quasi-energy branches across omega_d using only the principal
    eigenphases computed at each frequency.

    Parameters
    ----------
    phases_principal : ndarray, shape (n_freq, n_states)
        Principal phases in (-pi, pi], one row per omega_d.
    t_drive_ns : float
        Pulse duration in ns.

    Returns
    -------
    quasi_tracked : ndarray, shape (n_freq, n_states)
        Tracked quasi-energy branches in GHz.
    """
    phases_principal = np.asarray(phases_principal, dtype=float)
    n_freq, n_states = phases_principal.shape

    phases_tracked = np.full((n_freq, n_states), np.nan, dtype=float)

    # Initialize first point by sorting principal phases
    idx0 = np.argsort(phases_principal[0])
    phases_tracked[0, :] = phases_principal[0, idx0]

    for i in range(1, n_freq):
        curr = phases_principal[i]
        prev = phases_tracked[i - 1]

        if not np.all(np.isfinite(curr)) or not np.all(np.isfinite(prev)):
            continue

        _, best_unwrapped = _best_phase_permutation(curr, prev)
        phases_tracked[i, :] = best_unwrapped

    return phases_tracked / t_drive_ns  # GHz

def run_single(amp_idx, freq_idx, omega_d, drive_amp, t_drive_ns, t_ns, gamma_collective, Nk, max_depth, gamma_bath, T):
    """
    One (amp, omega_d) simulation:
      - mesolve for pop(t) and <S+>(t)
      - propagator for quasi-energies over t_drive

    Returns principal eigenphases, which are tracked later across omega_d.
    """
    h_static = WG["h_static"]
    sx_tot   = WG["sx_tot"]
    sp_tot   = WG["sp_tot"]
    pop_op   = WG["pop_op"]
    psi0     = WG["psi0"]
    c_ops    = WG["c_ops"]
    Q_bath = WG["Q_bath"]

    def drive_coeff(t, args):
        if t <= t_drive_ns:
            return drive_amp * np.cos(omega_d * t)
        return 0.0

    H = [h_static, [sx_tot, drive_coeff]]

    rho0 = psi0 * psi0.dag()
    bath = DrudeLorentzBath(Q_bath, lam=gamma_collective, gamma=gamma_bath, T=T, Nk=Nk)

    heom_solver = HEOMSolver(
        QobjEvo(H),
        [bath],
        max_depth=max_depth,
        options={"nsteps": 5000, "progress_bar": ''},
    )

    if amp_idx == 0:
        # first run is markovian
        res = mesolve(
            H, psi0, t_ns, c_ops,
            e_ops=[pop_op, sp_tot],
            options={"nsteps": 5000}
        )
    else: 
        res = heom_solver.run(
            rho0,
            t_ns,
            e_ops=[pop_op, sp_tot]
        )

    pop = np.real(res.expect[0])
    sp  = np.array(res.expect[1], dtype=complex)

    # principal eigenphases from propagator over pulse duration
    try:
        U = propagator(H, t_drive_ns, c_ops=[], options={"nsteps": 5000})
        phases = np.angle(np.linalg.eigvals(U.full()))
        phases = np.asarray(phases, dtype=float)
    except Exception:
        phases = np.full(4, np.nan, dtype=float)

    return amp_idx, freq_idx, pop, sp, phases

# ---------------------- Helpers (same as your pipeline) ----------------------
def smooth_envelope(amplitude, fraction=0.02):
    N = len(amplitude)
    win_len = int(max(3, min(N - 1, int(np.round(N * fraction)))))
    if win_len % 2 == 0:
        win_len += 1
    kernel = np.ones(win_len, dtype=float) / float(win_len)
    return np.convolve(amplitude, kernel, mode="same")

def _baseline_tail_median(y, frac=0.25):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 30:
        return np.nan
    n_tail = max(30, int(round(frac * len(y))))
    return float(np.median(y[-n_tail:]))

def tail_area_metric_us(pop_trace, t_ns, t_drive_ns, tail_start_buffer_ns, tail_baseline_fraction):
    """
    Post-pulse tail area:
      A_us = ∫ max(pop(t) - C, 0) dt, over t > t_drive + buffer
    Returns:
      A_us (float), emax (float), baseline C (float)
    """
    y_full = np.asarray(pop_trace, float)

    post = t_ns > (t_drive_ns + tail_start_buffer_ns)
    if post.sum() < 10:
        return np.nan, 0.0, np.nan

    tt = t_ns[post]
    yy = y_full[post]

    C = _baseline_tail_median(yy, frac=tail_baseline_fraction)
    if not np.isfinite(C):
        return np.nan, 0.0, np.nan

    y = yy - C
    y[~np.isfinite(y)] = 0.0
    y = np.maximum(y, 0.0)

    emax = float(np.max(y)) if len(y) else 0.0
    A_ns = float(np.trapezoid(y, tt))
    A_us = A_ns / 1e3
    return A_us, emax, C

def smooth_over_freq(x, points=9):
    x = np.asarray(x, float)
    if points is None or points <= 1:
        return x
    if points % 2 == 0:
        points += 1
    k = np.ones(points, dtype=float) / float(points)
    pad = points // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")

def phase_fft_map_from_sp_postpulse(sp_mat, omega_d_vals, t_ns, dt_ns, t_drive_ns, fft_view_mhz, n_pad):
    """
    Same protocol as before, but FFT is computed ONLY on post-pulse samples.

    Protocol:
      demod -> phi(t) -> env(t) -> FFT(phi*env) with Hann window

    Post-pulse region:
      use only indices where t_ns > t_drive_ns

    Returns:
      fft_map: shape (n_freq, n_fft_bins_kept)
      fft_freqs_mhz_kept
    """
    n_freq, n_time = sp_mat.shape

    # Post-pulse mask (strictly exclude drive region)
    post = t_ns > t_drive_ns
    n_post = int(np.count_nonzero(post))
    if n_post < 8:
        raise RuntimeError(
            f"Not enough post-pulse points for FFT: n_post={n_post}. "
            f"Increase t_total_ns or dt_ns, or reduce t_drive_ns."
        )

    t_post = t_ns[post]

    # Window over the post-pulse segment only
    window_fn = windows.hann(n_post)
    window_rms = np.sqrt(np.mean(window_fn**2))

    # FFT frequency axis (based on dt_ns), then keep view band
    fft_freqs_ghz = np.fft.rfftfreq(n_pad, d=dt_ns)
    fft_freqs_mhz = fft_freqs_ghz * 1e3
    mask = fft_freqs_mhz <= (fft_view_mhz + 1e-9)

    out = np.zeros((n_freq, int(mask.sum())), dtype=float)

    for i in tqdm(range(n_freq), desc="Phase FFT (post-pulse)", leave=False):
        omega_d = omega_d_vals[i]

        # Slice to post-pulse only
        Splus_post = sp_mat[i, post]

        # Demodulate using the post-pulse time stamps
        LO = np.exp(-1j * omega_d * t_post)
        demod = Splus_post * LO

        phi_t = np.angle(demod)
        amp = np.abs(demod)

        env = smooth_envelope(amp, fraction=0.02)
        phi_weighted = phi_t * env

        # Window and FFT
        phi_win = phi_weighted * window_fn
        F = np.fft.rfft(phi_win, n=n_pad)
        A = np.abs(F) / window_rms
        out[i, :] = A[mask]

    return out, fft_freqs_mhz[mask]

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()

    omega1 = float(args.omega1)
    omega2 = float(args.omega2)
    j_coupling = float(args.j_coupling)
    gamma_collective = float(args.gamma_collective)  # decay rate

    t_drive_ns = float(args.t_drive_ns)
    t_total_ns = float(args.t_total_ns)
    dt_ns = float(args.dt_ns)

    omega_d_vals = np.linspace(float(args.omega_d_min), float(args.omega_d_max), int(args.n_omega_d))
    amps = [float(args.amp1), float(args.amp2)]

    fft_view_mhz = float(args.fft_view_mhz)
    n_pad = int(args.n_pad)

    Nk = int(args.Nk)
    max_depth = int(args.max_depth)
    gamma_bath = float(args.gamma_bath)
    T = float(args.T)

    tail_start_buffer_ns = float(args.tail_start_buffer_ns)
    tail_baseline_fraction = float(args.tail_baseline_fraction)
    freq_smooth_points = int(args.freq_smooth_points)

    outdir = str(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # time grid
    t_ns = np.arange(0.0, t_total_ns, dt_ns)
    t_us = t_ns / 1e3

    n_amp = len(amps)
    n_freq = len(omega_d_vals)
    Nt = len(t_ns)

    # outputs
    heat = np.zeros((n_amp, Nt, n_freq), dtype=float)        # pop(t) for each amp, freq
    sp_mat = np.zeros((n_amp, n_freq, Nt), dtype=complex)    # <S+>(t)
    quasi_phases = np.full((n_amp, n_freq, 4), np.nan, dtype=float)
    quasi = np.full((n_amp, n_freq, 4), np.nan, dtype=float)

    # multiprocessing setup
    try:
        mp.set_start_method("fork", force=True)
        ctx = mp.get_context("fork")
    except RuntimeError:
        ctx = mp.get_context()

    max_workers = min(os.cpu_count() or 1, n_amp * n_freq)
    print(f"Using max_workers={max_workers} (start_method={ctx.get_start_method()})")

    # Stage 1: simulations over (amp, omega_d)
    futures = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(omega1, omega2, j_coupling, gamma_collective),
    ) as pool:
        for a_idx, A in enumerate(amps):
            for f_idx, w in enumerate(omega_d_vals):
                futures.append(pool.submit(run_single, a_idx, f_idx, w, A, t_drive_ns, t_ns, gamma_collective, Nk, max_depth, gamma_bath, T))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Drive sweep (all amps)"):
            a_idx, f_idx, pop, sp, phases = fut.result()
            heat[a_idx, :, f_idx] = pop
            sp_mat[a_idx, f_idx, :] = sp
            quasi_phases[a_idx, f_idx, :] = phases

    # Robust quasi branch tracking per amplitude
    for a_idx in range(n_amp):
        quasi[a_idx, :, :] = track_quasi_branches_from_principal_phases(
            quasi_phases[a_idx, :, :],
            t_drive_ns
        )

    # Stage 2: tail metric (threaded, cheap)
    tail_area_us = np.full((n_amp, n_freq), np.nan, dtype=float)
    emax_post = np.zeros((n_amp, n_freq), dtype=float)
    baseline_C = np.full((n_amp, n_freq), np.nan, dtype=float)

    n_threads = min(32, os.cpu_count() or 1, n_amp * n_freq)

    def _metric_one(a_idx, f_idx):
        A_us, emax, C = tail_area_metric_us(
            heat[a_idx, :, f_idx],
            t_ns=t_ns,
            t_drive_ns=t_drive_ns,
            tail_start_buffer_ns=tail_start_buffer_ns,
            tail_baseline_fraction=tail_baseline_fraction,
        )
        return a_idx, f_idx, A_us, emax, C

    with ThreadPoolExecutor(max_workers=n_threads) as tp:
        metric_futs = []
        for a_idx in range(n_amp):
            for f_idx in range(n_freq):
                metric_futs.append(tp.submit(_metric_one, a_idx, f_idx))

        for fut in tqdm(as_completed(metric_futs), total=len(metric_futs), desc="Tail area", leave=False):
            a_idx, f_idx, A_us, emax, C = fut.result()
            tail_area_us[a_idx, f_idx] = A_us
            emax_post[a_idx, f_idx] = emax
            baseline_C[a_idx, f_idx] = C

    tail_area_us_plot = np.zeros_like(tail_area_us)
    for a_idx in range(n_amp):
        tail_area_us_plot[a_idx, :] = smooth_over_freq(tail_area_us[a_idx, :], points=freq_smooth_points)

    # Stage 3: Phase FFT per amp (POST-PULSE ONLY)
    fft_map = []
    fft_freqs_mhz = None
    for a_idx in range(n_amp):
        fmap, fmhz = phase_fft_map_from_sp_postpulse(
            sp_mat[a_idx, :, :],
            omega_d_vals=omega_d_vals,
            t_ns=t_ns,
            dt_ns=dt_ns,
            t_drive_ns=t_drive_ns,
            fft_view_mhz=fft_view_mhz,
            n_pad=n_pad,
        )
        fft_map.append(fmap)
        if fft_freqs_mhz is None:
            fft_freqs_mhz = fmhz

    fft_map = np.stack(fft_map, axis=0)  # (n_amp, n_freq, n_fft)

    # Save
    outfile = os.path.join(outdir, f"two_tls_mark_nonmark_data.npz")
    np.savez(
        outfile,
        omega_d_vals=omega_d_vals,
        t_ns=t_ns,
        t_us=t_us,
        heat=heat,
        sp_mat=sp_mat,
        quasi=quasi,
        quasi_phases=quasi_phases,
        amps=np.array(amps, dtype=float),
        tail_area_us=tail_area_us,
        tail_area_us_plot=tail_area_us_plot,
        baseline_C=baseline_C,
        emax_postpulse=emax_post,
        fft_map=fft_map,
        fft_freqs_mhz=fft_freqs_mhz,
        omega1=omega1,
        omega2=omega2,
        t_drive_ns=t_drive_ns,
        dt_ns=dt_ns,
        j_coupling=j_coupling,
        gamma_collective=gamma_collective,
        T=T,
        Nk=Nk,
        max_depth=max_depth,
        gamma_bath=gamma_bath,
        fft_view_mhz=fft_view_mhz,
        n_pad=n_pad,
        tail_start_buffer_ns=tail_start_buffer_ns,
        tail_baseline_fraction=tail_baseline_fraction,
        freq_smooth_points=freq_smooth_points,
    )
    print("✓ Data saved →", outfile)

if __name__ == "__main__":
    main()