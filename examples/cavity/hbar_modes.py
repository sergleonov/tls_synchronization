# Cavity-QED simulation with HBAR modes
# Code from Salil Bedkihal

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from tqdm import tqdm

# ============================================================
# UNITS
# ============================================================
# Frequency: GHz, Time: ns, so 1 GHz = 1/ns and omega = 2*pi*f
# ============================================================

PI2 = 2.0 * np.pi

# ============================================================
# TLS PARAMETERS
# ============================================================

f_tls1 = 3.95
f_tls2 = 4.05
J_xx = 0.00
J_zz = 0.00
gamma1 = 0.0005
gamma2 = 0.0005
gamma_phi1 = 0.00005
gamma_phi2 = 0.00005
Omega_drive = 0.2

# ============================================================
# HBAR PARAMETERS
# ============================================================

f_mode_center = 4.000
N_modes = 10
mode_spacing = 0.010       # 10 MHz = 0.010 GHz
mode_indices = np.arange(N_modes) - N_modes // 2
mode_freqs = f_mode_center + mode_indices * mode_spacing
kappa_modes = np.full(N_modes, 0.001)
g_mode_tls1 = np.full(N_modes, 0.02)
g_mode_tls2 = np.full(N_modes, 0.02)

# ============================================================
# TIME PARAMETERS
# ============================================================

T_drive = 200.0
T_total = 1600.0
dt = 0.01
dt_output = 0.1

steps_per_output = int(round(dt_output / dt))
if not np.isclose(steps_per_output * dt, dt_output):
    raise ValueError("dt_output must be an integer multiple of dt.")

tlist = np.arange(0.0, T_total + 0.5 * dt_output, dt_output)
Nt = len(tlist)

# ============================================================
# DRIVE FREQUENCIES
# ============================================================

drive_freqs = np.linspace(3.0, 4.5, 300)
Nf = len(drive_freqs)

# ============================================================
# PAULI MATRICES
# ============================================================

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 1], [0, 0]], dtype=complex)
sp = sm.conj().T

# ============================================================
# TWO-TLS OPERATORS
# ============================================================

sx1 = np.kron(sx, I2)
sx2 = np.kron(I2, sx)
sy1 = np.kron(sy, I2)
sy2 = np.kron(I2, sy)
sz1 = np.kron(sz, I2)
sz2 = np.kron(I2, sz)
sm1 = np.kron(sm, I2)
sm2 = np.kron(I2, sm)
sp1 = sm1.conj().T
sp2 = sm2.conj().T

# ============================================================
# COLLECTIVE OPERATORS
# ============================================================

Sx = sx1 + sx2
S_plus = sp1 + sp2

# ============================================================
# INITIAL TLS STATE
# ============================================================

g_state = np.array([1, 0], dtype=complex)
psi_g = np.kron(g_state, g_state)
rho0 = np.outer(psi_g, psi_g.conj())

# ============================================================
# DISSIPATION
# ============================================================

Gamma1 = PI2 * gamma1
Gamma2 = PI2 * gamma2
Gamma_phi1 = PI2 * gamma_phi1
Gamma_phi2 = PI2 * gamma_phi2

C1 = np.sqrt(Gamma1) * sm1
C2 = np.sqrt(Gamma2) * sm2
Cphi1 = np.sqrt(Gamma_phi1 / 2.0) * sz1
Cphi2 = np.sqrt(Gamma_phi2 / 2.0) * sz2

collapse_data = []
for C in [C1, C2, Cphi1, Cphi2]:
    Cd = C.conj().T
    CdC = Cd @ C
    collapse_data.append((C, Cd, CdC))

# ============================================================
# BARE TLS HAMILTONIAN
# ============================================================

H_tls_bare = (
    -0.5 * PI2 * f_tls1 * sz1
    - 0.5 * PI2 * f_tls2 * sz2
    + PI2 * J_xx * (sx1 @ sx2)
    + PI2 * J_zz * (sz1 @ sz2)
)

# ============================================================
# SQUARE DRIVE
# ============================================================

def square_drive(t):
    if t < T_drive:
        return 1.0
    return 0.0

# ============================================================
# EXTERNAL TLS DRIVE
# ============================================================
# The drive acts ONLY on the TLSs. There is NO direct HBAR drive.
# ============================================================

def low_q_field_batch(t):
    envelope = square_drive(t)
    if envelope == 0.0:
        return np.zeros(Nf, dtype=float)
    return np.cos(PI2 * drive_freqs * t)

# ============================================================
# RIGHT-HAND SIDE
# ============================================================

def rhs_batch(t, alpha_modes, rho):
    # External drive
    E_low = low_q_field_batch(t)

    # TLS polarization
    sx_exp1 = np.einsum("ij,nji->n", sx1, rho)
    sx_exp2 = np.einsum("ij,nji->n", sx2, rho)

    # HBAR equations: no direct external HBAR drive; modes sourced by TLS polarization
    omega_modes = PI2 * mode_freqs
    kappa_rad = PI2 * kappa_modes

    dalpha_modes = (
        -1j * omega_modes[None, :] * alpha_modes
        - 0.5 * kappa_rad[None, :] * alpha_modes
        - 1j * PI2 * g_mode_tls1[None, :] * sx_exp1[:, None]
        - 1j * PI2 * g_mode_tls2[None, :] * sx_exp2[:, None]
    )

    # TLS Hamiltonian
    H = np.broadcast_to(H_tls_bare, (Nf, 4, 4)).copy()

    # Direct TLS drive
    H += PI2 * Omega_drive * E_low[:, None, None] * Sx[None, :, :]

    # HBAR field acting back on TLSs
    mode_quadrature = alpha_modes + alpha_modes.conjugate()
    E_mode_tls1 = np.sum(g_mode_tls1[None, :] * mode_quadrature, axis=1)
    E_mode_tls2 = np.sum(g_mode_tls2[None, :] * mode_quadrature, axis=1)
    H += PI2 * E_mode_tls1[:, None, None] * sx1[None, :, :]
    H += PI2 * E_mode_tls2[:, None, None] * sx2[None, :, :]

    # Unitary TLS evolution
    Hrho = np.matmul(H, rho)
    rhoH = np.matmul(rho, H)
    drho = -1j * (Hrho - rhoH)

    # Lindblad terms
    for C, Cd, CdC in collapse_data:
        C_rho = np.matmul(C[None, :, :], rho)
        C_rho_Cd = np.matmul(C_rho, Cd[None, :, :])
        CdC_rho = np.matmul(CdC[None, :, :], rho)
        rho_CdC = np.matmul(rho, CdC[None, :, :])
        drho += C_rho_Cd - 0.5 * (CdC_rho + rho_CdC)

    return dalpha_modes, drho

# ============================================================
# SIMULATION
# ============================================================

def simulate_batch():
    print()
    print("=" * 70)
    print("TLS + MANY HBAR MODES")
    print("=" * 70)
    print(f"TLS frequencies: {f_tls1:.4f}, {f_tls2:.4f} GHz")
    print(f"HBAR modes: {N_modes}")
    print(f"HBAR range: {mode_freqs[0]:.4f} - {mode_freqs[-1]:.4f} GHz")
    print(f"HBAR spacing: {mode_spacing:.4f} GHz")
    print(f"Drive frequencies: {drive_freqs[0]:.4f} - {drive_freqs[-1]:.4f} GHz")
    print(f"T_drive = {T_drive:.1f} ns")
    print(f"T_total = {T_total:.1f} ns")
    print("=" * 70)
    print()

    # Initial conditions
    alpha_modes = np.zeros((Nf, N_modes), dtype=complex)
    rho = np.broadcast_to(rho0, (Nf, 4, 4)).copy()

    # Output arrays
    alpha_modes_out = np.zeros((Nf, N_modes, Nt), dtype=complex)
    S_plus_out = np.zeros((Nf, Nt), dtype=complex)

    # Initial S+
    S_plus_out[:, 0] = np.einsum("ij,nji->n", S_plus, rho)

    # Number of RK4 steps
    n_steps = int(round(T_total / dt))
    output_index = 1

    # RK4
    for step in tqdm(range(n_steps), desc="RK4"):
        t = step * dt

        # K1
        k1_a, k1_r = rhs_batch(t, alpha_modes, rho)

        # K2
        k2_a, k2_r = rhs_batch(t + 0.5 * dt, alpha_modes + 0.5 * dt * k1_a, rho + 0.5 * dt * k1_r)

        # K3
        k3_a, k3_r = rhs_batch(t + 0.5 * dt, alpha_modes + 0.5 * dt * k2_a, rho + 0.5 * dt * k2_r)

        # K4
        k4_a, k4_r = rhs_batch(t + dt, alpha_modes + dt * k3_a, rho + dt * k3_r)

        # Update HBAR modes
        alpha_modes += dt / 6.0 * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)

        # Update TLS density matrices
        rho += dt / 6.0 * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)

        # Save
        if (step + 1) % steps_per_output == 0:
            if output_index < Nt:
                alpha_modes_out[:, :, output_index] = alpha_modes
                S_plus_out[:, output_index] = np.einsum("ij,nji->n", S_plus, rho)
                output_index += 1

    return alpha_modes_out, S_plus_out

# ============================================================
# RUN
# ============================================================

alpha_modes, S_plus_time = simulate_batch()

# ============================================================
# HBAR MODE MAGNITUDES
# ============================================================

mode_magnitudes = np.abs(alpha_modes)

# ============================================================
# COLLECTIVE HBAR FIELD (alpha_total = sum_m alpha_m)
# ============================================================

alpha_total = np.sum(alpha_modes, axis=1)
alpha_total_mag = np.abs(alpha_total)

# ============================================================
# SELECTED DRIVE
# ============================================================

f_drive_selected = 4.000
idx_selected = np.argmin(np.abs(drive_freqs - f_drive_selected))
print(f"\nSelected drive frequency = {drive_freqs[idx_selected]:.6f} GHz")

# ============================================================
# 1. COLLECTIVE TLS DYNAMICS
# ============================================================

S_plus_selected = S_plus_time[idx_selected, :]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tlist, np.real(S_plus_selected), label="Re[S+]")
ax.plot(tlist, np.imag(S_plus_selected), label="Im[S+]")
ax.plot(tlist, np.abs(S_plus_selected), label="|S+|")
ax.axvline(T_drive, linestyle="--")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Collective TLS coherence")
ax.set_title(f"Collective TLS dynamics, fd = {drive_freqs[idx_selected]:.4f} GHz")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 2. COLLECTIVE TLS S+ MAGNITUDE HEATMAP
# ============================================================
# |S+| vs drive frequency (y) and time (x)
# ============================================================

S_plus_magnitude = np.abs(S_plus_time)

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    S_plus_magnitude,
    extent=[tlist[0], tlist[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
)
ax.axvline(T_drive, linestyle="--")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title("Collective TLS $|S_+|$")
fig.colorbar(im, ax=ax, label="$|S_+|$")
plt.tight_layout()
plt.show()

# ============================================================
# 3. INDIVIDUAL HBAR MODE DYNAMICS
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
for m in range(N_modes):
    ax.plot(tlist, mode_magnitudes[idx_selected, m, :], linewidth=1.0, label=f"{mode_freqs[m]:.3f} GHz")
ax.axvline(T_drive, linestyle="--")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("|alpha_m|")
ax.set_title(f"Individual HBAR modes, fd = {drive_freqs[idx_selected]:.4f} GHz")
plt.tight_layout()
plt.show()

# ============================================================
# 4. HBAR MODE HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    mode_magnitudes[idx_selected, :, :],
    extent=[tlist[0], tlist[-1], mode_freqs[0], mode_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
)
ax.axvline(T_drive, linestyle="--")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("HBAR mode frequency (GHz)")
ax.set_title(f"HBAR mode magnitudes, fd = {drive_freqs[idx_selected]:.4f} GHz")
fig.colorbar(im, ax=ax, label="|alpha_m|")
plt.tight_layout()
plt.show()

# ============================================================
# 5. COLLECTIVE HBAR MAGNITUDE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tlist, alpha_total_mag[idx_selected, :])
ax.axvline(T_drive, linestyle="--")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("|sum_m alpha_m|")
ax.set_title(f"Collective HBAR field, fd = {drive_freqs[idx_selected]:.4f} GHz")
plt.tight_layout()
plt.show()

# ============================================================
# POST-DRIVE DATA
# ============================================================

post_mask = tlist >= T_drive
t_post = tlist[post_mask] - T_drive

# ============================================================
# COLLECTIVE HBAR DEMODULATED SIGNAL
# ============================================================

alpha_total_post = alpha_total[:, post_mask]

# HBAR alpha free evolution: alpha ~ exp(-i omega t), so demodulate with exp(+i omega_drive t)
rotator_HBAR = np.exp(+1j * PI2 * drive_freqs[:, None] * t_post[None, :])
alpha_total_demod = alpha_total_post * rotator_HBAR

# HBAR DEMODULATED MAGNITUDE / PHASE
alpha_total_demod_mag = np.abs(alpha_total_demod)
alpha_total_demod_phase = np.angle(alpha_total_demod)

# ============================================================
# 6. HBAR DEMODULATED PHASE HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    alpha_total_demod_phase,
    extent=[t_post[0], t_post[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi,
)
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title("Drive-frame collective HBAR phase")
fig.colorbar(im, ax=ax, label="Phase (rad)")
plt.tight_layout()
plt.show()

# ============================================================
# HBAR PHASE × ENVELOPE
# ============================================================

alpha_total_phase_weighted = alpha_total_demod_mag * alpha_total_demod_phase

# HANN WINDOW
window_HBAR = windows.hann(len(t_post))
alpha_total_windowed = alpha_total_phase_weighted * window_HBAR[None, :]

# ============================================================
# 7. HBAR PHASE FFT
# ============================================================

N_fft_HBAR = 2**15
FFT_HBAR_phase = np.fft.rfft(alpha_total_windowed, n=N_fft_HBAR, axis=1)
FFT_HBAR_phase_amp = np.abs(FFT_HBAR_phase)
fft_freqs_HBAR = np.fft.rfftfreq(N_fft_HBAR, d=dt_output)

fft_max_HBAR = max(1.5 * Omega_drive, 0.01)
fft_mask_HBAR = fft_freqs_HBAR <= fft_max_HBAR
fft_freq_plot_HBAR = fft_freqs_HBAR[fft_mask_HBAR]
FFT_HBAR_plot = FFT_HBAR_phase_amp[:, fft_mask_HBAR]

# HBAR FFT ROW NORMALIZATION
FFT_HBAR_norm = np.zeros_like(FFT_HBAR_plot)
for i in range(Nf):
    row_max = np.max(FFT_HBAR_plot[i, :])
    if row_max > 0.0:
        FFT_HBAR_norm[i, :] = FFT_HBAR_plot[i, :] / row_max

# ============================================================
# 8. HBAR PHASE FFT HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    FFT_HBAR_norm.T,
    extent=[drive_freqs[0], drive_freqs[-1], fft_freq_plot_HBAR[0], fft_freq_plot_HBAR[-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
    vmin=0.0,
    vmax=1.0,
)
ax.set_xlabel("Drive frequency (GHz)")
ax.set_ylabel("FFT frequency (GHz)")
ax.set_title("FFT of demodulated collective HBAR phase")
ax.set_ylim(0.0, fft_max_HBAR)
fig.colorbar(im, ax=ax, label="Normalized FFT amplitude")
plt.tight_layout()
plt.show()

# ============================================================
# 9. HBAR FFT LINE CUTS
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
for fd in [f_tls1, f_mode_center, f_tls2]:
    idx = np.argmin(np.abs(drive_freqs - fd))
    ax.plot(fft_freq_plot_HBAR, FFT_HBAR_norm[idx, :], linewidth=1.5, label=f"fd = {drive_freqs[idx]:.4f} GHz")
ax.set_xlim(0.0, fft_max_HBAR)
ax.set_xlabel("FFT frequency (GHz)")
ax.set_ylabel("Normalized FFT amplitude")
ax.set_title("HBAR phase FFT line cuts")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 10. SELECTED-DRIVE HBAR DEMODULATED PHASE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t_post, alpha_total_demod_phase[idx_selected, :], label="HBAR demodulated phase")
ax.axhline(0.0, linestyle="--")
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Phase (rad)")
ax.set_title(f"Demodulated collective HBAR phase, fd = {drive_freqs[idx_selected]:.4f} GHz")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# COLLECTIVE TLS S+ DEMODULATION
# ============================================================

S_plus_post = S_plus_time[:, post_mask]

# S+ convention: S+ ~ exp(+i omega t), so S+_demod = S+ exp(-i omega_drive t)
rotator_Splus = np.exp(-1j * PI2 * drive_freqs[:, None] * t_post[None, :])
S_plus_demod = S_plus_post * rotator_Splus

# TLS DEMODULATED MAGNITUDE / PHASE
S_plus_demod_mag = np.abs(S_plus_demod)
S_plus_demod_phase = np.angle(S_plus_demod)

# ============================================================
# 11. S+ DEMODULATED PHASE HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    S_plus_demod_phase,
    extent=[t_post[0], t_post[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi,
)
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title("Drive-frame collective TLS S+ phase")
fig.colorbar(im, ax=ax, label="Phase (rad)")
plt.tight_layout()
plt.show()

# ============================================================
# S+ PHASE × ENVELOPE
# ============================================================

S_plus_phase_weighted = S_plus_demod_mag * S_plus_demod_phase

# HANN WINDOW
window_Splus = windows.hann(len(t_post))
S_plus_windowed = S_plus_phase_weighted * window_Splus[None, :]

# ============================================================
# 12. S+ PHASE FFT
# ============================================================

N_fft_Splus = 2**15
FFT_Splus_phase = np.fft.rfft(S_plus_windowed, n=N_fft_Splus, axis=1)
FFT_Splus_phase_amp = np.abs(FFT_Splus_phase)
fft_freqs_Splus = np.fft.rfftfreq(N_fft_Splus, d=dt_output)

fft_max_Splus = max(1.5 * Omega_drive, 0.01)
fft_mask_Splus = fft_freqs_Splus <= fft_max_Splus
fft_freq_plot_Splus = fft_freqs_Splus[fft_mask_Splus]
FFT_Splus_plot = FFT_Splus_phase_amp[:, fft_mask_Splus]

# S+ FFT ROW NORMALIZATION
FFT_Splus_norm = np.zeros_like(FFT_Splus_plot)
for i in range(Nf):
    row_max = np.max(FFT_Splus_plot[i, :])
    if row_max > 0.0:
        FFT_Splus_norm[i, :] = FFT_Splus_plot[i, :] / row_max

# ============================================================
# 13. S+ PHASE FFT HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    FFT_Splus_norm.T,
    extent=[drive_freqs[0], drive_freqs[-1], fft_freq_plot_Splus[0], fft_freq_plot_Splus[-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
    vmin=0.0,
    vmax=1.0,
)
ax.set_xlabel("Drive frequency (GHz)")
ax.set_ylabel("FFT frequency (GHz)")
ax.set_title("FFT of demodulated collective TLS S+ phase")
ax.set_ylim(0.0, fft_max_Splus)
fig.colorbar(im, ax=ax, label="Normalized FFT amplitude")
plt.tight_layout()
plt.show()

# ============================================================
# 14. S+ FFT LINE CUTS
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
for fd in [f_tls1, f_mode_center, f_tls2]:
    idx = np.argmin(np.abs(drive_freqs - fd))
    ax.plot(fft_freq_plot_Splus, FFT_Splus_norm[idx, :], linewidth=1.5, label=f"fd = {drive_freqs[idx]:.4f} GHz")
ax.set_xlim(0.0, fft_max_Splus)
ax.set_xlabel("FFT frequency (GHz)")
ax.set_ylabel("Normalized FFT amplitude")
ax.set_title("S+ phase FFT line cuts")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 15. SELECTED-DRIVE DEMODULATED S+ PHASE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t_post, S_plus_demod_phase[idx_selected, :], label="S+ demodulated phase")
ax.axhline(0.0, linestyle="--")
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Phase (rad)")
ax.set_title(f"Demodulated collective S+ phase, fd = {drive_freqs[idx_selected]:.4f} GHz")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 16. COLLECTIVE HBAR REAL/IMAGINARY FIELD
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tlist, np.real(alpha_total[idx_selected, :]), label="Re[alpha_HBAR]")
ax.plot(tlist, np.imag(alpha_total[idx_selected, :]), label="Im[alpha_HBAR]")
ax.axvline(T_drive, linestyle="--")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Collective HBAR field")
ax.set_title(f"Collective HBAR field, fd = {drive_freqs[idx_selected]:.4f} GHz")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"TLS frequencies       : {f_tls1:.4f}, {f_tls2:.4f} GHz")
print(f"HBAR modes            : {N_modes}")
print(f"HBAR range            : {mode_freqs[0]:.4f} - {mode_freqs[-1]:.4f} GHz")
print(f"HBAR spacing          : {mode_spacing:.4f} GHz")
print(f"Drive range           : {drive_freqs[0]:.4f} - {drive_freqs[-1]:.4f} GHz")
print(f"Drive amplitude       : {Omega_drive:.4f} GHz")
print(f"Drive duration        : {T_drive:.1f} ns")
print(f"Total time            : {T_total:.1f} ns")
print()
print("External HBAR drive   : NONE")
print("HBAR source            : TLS polarization")
print("Collective HBAR       : sum_m alpha_m")
print()
print("HBAR demodulation     : alpha_HBAR * exp(+i 2*pi fd t)")
print("TLS S+ demodulation   : S+ * exp(-i 2*pi fd t)")
print()
print("HBAR phase FFT        : DONE")
print("TLS S+ phase FFT      : DONE")
print("=" * 70)