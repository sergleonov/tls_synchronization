# Single-mode cavity-QED simulation
# Code from Salil Bedkihal

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import windows

# ============================================================
# 0. UNITS
# ============================================================
# Frequency : GHz
# Time      : ns
# 1 GHz = 1/ns, omega = 2*pi*f
# ============================================================

PI2 = 2.0 * np.pi

# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

# CAVITY
f_c = 4.000                 # GHz
kappa_c = 0.001             # GHz

# TLS FREQUENCIES
f_tls1 = 3.95               # GHz
f_tls2 = 4.05               # GHz

# CAVITY-TLS COUPLINGS
g_cav1 = 0.01               # GHz
g_cav2 = 0.01               # GHz

# TLS-TLS INTERACTION
J_xx = 0.001                # GHz
J_zz = 0.001                # GHz

# TLS RELAXATION / DEPHASING
gamma1 = 0.0005             # GHz
gamma2 = 0.0005             # GHz
gamma_phi1 = 0.00005        # GHz
gamma_phi2 = 0.00005        # GHz

# COLLECTIVE TLS DRIVE
Omega_drive = 0.2           # GHz

# DIRECT CAVITY DRIVE
eta_cavity = 0.0001         # GHz

# ============================================================
# 2. TIME PARAMETERS
# ============================================================

T_drive = 50.0              # ns
T_total = 1600.0            # ns
dt = 0.01                   # ns
dt_output = 0.1             # ns

if not np.isclose(dt_output / dt, round(dt_output / dt)):
    raise ValueError("dt_output must be an integer multiple of dt.")

steps_per_output = int(round(dt_output / dt))
tlist = np.arange(0.0, T_total + 0.5 * dt_output, dt_output)
Nt = len(tlist)

# ============================================================
# 3. DRIVE FREQUENCIES
# ============================================================

drive_freqs = np.linspace(3.5, 4.5, 300)
Nf = len(drive_freqs)

# ============================================================
# 4. PAULI MATRICES
# ============================================================

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm = np.array([[0, 1], [0, 0]], dtype=complex)
sp = sm.conj().T

# ============================================================
# 5. TWO-TLS OPERATORS
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
# 6. COLLECTIVE TLS OPERATOR
# ============================================================

Sx = sx1 + sx2

# ============================================================
# 7. INITIAL TLS STATE
# ============================================================

g_state = np.array([1, 0], dtype=complex)
psi_g = np.kron(g_state, g_state)
rho0 = np.outer(psi_g, psi_g.conj())

# ============================================================
# 8. DISSIPATION
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
for C in (C1, C2, Cphi1, Cphi2):
    Cd = C.conj().T
    CdC = Cd @ C
    collapse_data.append((C, Cd, CdC))

# ============================================================
# 9. BARE TLS HAMILTONIAN
# ============================================================

H_tls_bare = (
    -0.5 * PI2 * f_tls1 * sz1
    - 0.5 * PI2 * f_tls2 * sz2
    + PI2 * J_xx * (sx1 @ sx2)
    + PI2 * J_zz * (sz1 @ sz2)
)

# ============================================================
# 10. SQUARE DRIVE
# ============================================================

def square_drive(t):
    if t < T_drive:
        return 1.0
    return 0.0

# ============================================================
# 11. COLLECTIVE TLS DRIVE
# ============================================================

def low_q_field_batch(t):
    envelope = square_drive(t)
    if envelope == 0.0:
        return np.zeros(Nf, dtype=float)
    return np.cos(PI2 * drive_freqs * t)

# ============================================================
# 12. MAXWELL-BLOCH RHS
# ============================================================

def rhs_batch(t, alpha, rho):
    # EXTERNAL TLS DRIVE
    E_low = low_q_field_batch(t)

    # TLS POLARIZATIONS
    sx_exp1 = np.einsum("ij,nji->n", sx1, rho)
    sx_exp2 = np.einsum("ij,nji->n", sx2, rho)

    # CAVITY
    omega_c = PI2 * f_c
    kappa = PI2 * kappa_c

    dalpha = (
        -1j * omega_c * alpha
        - 0.5 * kappa * alpha
        - 1j * PI2 * eta_cavity * E_low
        - 1j * PI2 * g_cav1 * sx_exp1
        - 1j * PI2 * g_cav2 * sx_exp2
    )

    # TLS HAMILTONIAN
    H = np.broadcast_to(H_tls_bare, (Nf, 4, 4)).copy()

    # COLLECTIVE EXTERNAL TLS DRIVE
    H += PI2 * Omega_drive * E_low[:, None, None] * Sx[None, :, :]

    # CAVITY FIELD ACTING BACK ON TLSs
    E_high = alpha + alpha.conjugate()
    H += PI2 * g_cav1 * E_high[:, None, None] * sx1[None, :, :]
    H += PI2 * g_cav2 * E_high[:, None, None] * sx2[None, :, :]

    # UNITARY TLS EVOLUTION
    Hrho = np.matmul(H, rho)
    rhoH = np.matmul(rho, H)
    drho = -1j * (Hrho - rhoH)

    # LINDBLAD DISSIPATION
    for C, Cd, CdC in collapse_data:
        C_rho = np.matmul(C[None, :, :], rho)
        C_rho_Cd = np.matmul(C_rho, Cd[None, :, :])
        CdC_rho = np.matmul(CdC[None, :, :], rho)
        rho_CdC = np.matmul(rho, CdC[None, :, :])
        drho += C_rho_Cd - 0.5 * (CdC_rho + rho_CdC)

    return dalpha, drho

# ============================================================
# 13. VECTORIZED RK4
# ============================================================

def simulate_batch():
    print()
    print("=" * 70)
    print("COLLECTIVELY DRIVEN CAVITY + TWO TLS MAXWELL-BLOCH")
    print("=" * 70)
    print(f"Drive frequencies : {Nf}")
    print(f"Frequency range   : {drive_freqs[0]:.3f} - {drive_freqs[-1]:.3f} GHz")
    print(f"Cavity frequency  : {f_c:.3f} GHz")
    print(f"TLS frequencies    : {f_tls1:.3f}, {f_tls2:.3f} GHz")
    print(f"Cavity-TLS g      : {g_cav1:.4f}, {g_cav2:.4f} GHz")
    print(f"TLS drive amplitude: {Omega_drive:.4f} GHz")
    print(f"Cavity kappa      : {kappa_c:.6f} GHz")
    print(f"Internal dt       : {dt:.4f} ns")
    print(f"Output dt         : {dt_output:.4f} ns")
    print(f"Total time        : {T_total:.1f} ns")
    print(f"Drive duration    : {T_drive:.1f} ns")
    print("TLS drive         : COLLECTIVE")
    print("Cavity drive      : NONE")
    print("=" * 70)
    print()

    # INITIAL CONDITIONS
    alpha = np.zeros(Nf, dtype=complex)
    rho = np.broadcast_to(rho0, (Nf, 4, 4)).copy()

    # OUTPUT
    alpha_out = np.zeros((Nf, Nt), dtype=complex)
    alpha_out[:, 0] = alpha

    # NUMBER OF INTERNAL STEPS
    n_steps = int(round(T_total / dt))
    output_index = 1

    # RK4
    for step in tqdm(range(n_steps), desc="Maxwell-Bloch RK4"):
        t = step * dt

        # K1
        k1_a, k1_r = rhs_batch(t, alpha, rho)

        # K2
        k2_a, k2_r = rhs_batch(t + 0.5 * dt, alpha + 0.5 * dt * k1_a, rho + 0.5 * dt * k1_r)

        # K3
        k3_a, k3_r = rhs_batch(t + 0.5 * dt, alpha + 0.5 * dt * k2_a, rho + 0.5 * dt * k2_r)

        # K4
        k4_a, k4_r = rhs_batch(t + dt, alpha + dt * k3_a, rho + dt * k3_r)

        # UPDATE CAVITY
        alpha += dt / 6.0 * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)

        # UPDATE TLS
        rho += dt / 6.0 * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)

        # STORE OUTPUT
        if (step + 1) % steps_per_output == 0:
            if output_index < Nt:
                alpha_out[:, output_index] = alpha
                output_index += 1

    return alpha_out

# ============================================================
# 14. RUN SIMULATION
# ============================================================

alpha = simulate_batch()

# ============================================================
# 15. RAW CAVITY I-Q
# ============================================================

I = np.real(alpha)
Q = np.imag(alpha)
IQ_magnitude = np.abs(alpha)
IQ_phase = np.angle(alpha)

# ============================================================
# 16. FULL CAVITY MAGNITUDE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    IQ_magnitude,
    extent=[tlist[0], tlist[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
)
ax.axvline(T_drive, linestyle="--", linewidth=1.5)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title(r"Cavity output $|\alpha(t)|$")
fig.colorbar(im, ax=ax, label=r"$|\alpha|$")
plt.tight_layout()
plt.show()

# ============================================================
# 17. RAW CAVITY PHASE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    IQ_phase,
    extent=[tlist[0], tlist[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi,
)
ax.axvline(T_drive, linestyle="--", linewidth=1.5)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title(r"Raw cavity phase $\arg[\alpha(t)]$")
fig.colorbar(im, ax=ax, label="Phase (rad)")
plt.tight_layout()
plt.show()

# ============================================================
# 18. POST-DRIVE DATA
# ============================================================

post_mask = tlist >= T_drive
t_post = tlist[post_mask] - T_drive
alpha_post = alpha[:, post_mask]

# ============================================================
# 19. DRIVE-FRAME DEMODULATION
# ============================================================
# Cavity eqn: d alpha / dt = -i omega_c alpha + ...
# Free field: alpha(t) ~ exp(-i omega_c t).
# Drive frame: alpha_demod = alpha * exp(+i omega_d t).
# Gives alpha_demod ~ exp[-i(omega_c - omega_d)t].
# ============================================================

rotator_drive = np.exp(1j * PI2 * drive_freqs[:, None] * t_post[None, :])
alpha_demod = alpha_post * rotator_drive

# ============================================================
# 20. DEMODULATED SIGNAL
# ============================================================

alpha_demod_mag = np.abs(alpha_demod)
alpha_demod_phase = np.angle(alpha_demod)

# ============================================================
# 21. PLOT DEMODULATED COMPLEX SIGNAL MAGNITUDE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    alpha_demod_mag,
    extent=[t_post[0], t_post[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
)
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title(r"Demodulated cavity signal $|\alpha_{\rm demod}(t;f_d)|$")
fig.colorbar(im, ax=ax, label=r"$|\alpha_{\rm demod}|$")
plt.tight_layout()
plt.show()

# ============================================================
# 22. DEMODULATED PHASE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    alpha_demod_phase,
    extent=[t_post[0], t_post[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi,
)
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title(r"Drive-frame cavity phase $\phi_{\rm demod}=\arg(\alpha_{\rm demod})$")
fig.colorbar(im, ax=ax, label="Phase (rad)")
plt.tight_layout()
plt.show()

# ============================================================
# 23. DEMODULATED I AND Q
# ============================================================

I_demod = np.real(alpha_demod)
Q_demod = np.imag(alpha_demod)

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    I_demod,
    extent=[t_post[0], t_post[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
)
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title(r"Drive-frame $I_{\rm demod}=\mathrm{Re}(\alpha_{\rm demod})$")
fig.colorbar(im, ax=ax, label=r"$I_{\rm demod}$")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    Q_demod,
    extent=[t_post[0], t_post[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
)
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title(r"Drive-frame $Q_{\rm demod}=\mathrm{Im}(\alpha_{\rm demod})$")
fig.colorbar(im, ax=ax, label=r"$Q_{\rm demod}$")
plt.tight_layout()
plt.show()

# ============================================================
# 24. PHASE × ENVELOPE
# ============================================================
# phi(t) = angle(alpha_demod)
# envelope(t) = |alpha_demod|
# signal(t) = phi(t) * envelope(t)
# ============================================================

phase_demod = alpha_demod_phase
envelope_demod = alpha_demod_mag
phase_weighted = phase_demod * envelope_demod

# ============================================================
# 25. PLOT PHASE × ENVELOPE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    phase_weighted,
    extent=[t_post[0], t_post[-1], drive_freqs[0], drive_freqs[-1]],
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
)
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title(r"Demodulated phase weighted by cavity envelope $|\alpha_{\rm demod}|\phi_{\rm demod}$")
fig.colorbar(im, ax=ax, label=r"$|\alpha_{\rm demod}|\phi_{\rm demod}$")
plt.tight_layout()
plt.show()

# ============================================================
# 26. SINGLE PHASE FFT
# ============================================================
# Fourier transform of |alpha_demod(t)| * angle(alpha_demod(t)).
# ============================================================

window = windows.hann(len(t_post))
windowed_signal = phase_weighted * window[None, :]

# Zero padding
N_fft = 2**15
FFT_phase = np.fft.rfft(windowed_signal, n=N_fft, axis=1)
FFT_phase_amp = np.abs(FFT_phase)
fft_freqs = np.fft.rfftfreq(N_fft, d=dt_output)

# ============================================================
# 27. PHYSICALLY RELEVANT FFT RANGE
# ============================================================
# Relevant low-frequency scale is the TLS drive amplitude Omega_drive.
# Display up to a small multiple of Omega_drive.
# ============================================================

fft_max = max(1.5 * Omega_drive, 0.01)
fft_mask = fft_freqs <= fft_max
fft_freq_plot = fft_freqs[fft_mask]
FFT_phase_plot = FFT_phase_amp[:, fft_mask]

# ============================================================
# 28. NORMALIZE EACH DRIVE-FREQUENCY TRACE
# ============================================================
# Normalization is ONLY for visualization.
# ============================================================

FFT_phase_norm = np.zeros_like(FFT_phase_plot)
for i in range(Nf):
    row_max = np.max(FFT_phase_plot[i, :])
    if row_max > 0.0:
        FFT_phase_norm[i, :] = FFT_phase_plot[i, :] / row_max

# ============================================================
# 29. FINAL DRIVE FREQUENCY vs FFT FREQUENCY HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    FFT_phase_norm.T,
    extent=[drive_freqs[0], drive_freqs[-1], fft_freq_plot[0], fft_freq_plot[-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
    vmin=0.0,
    vmax=1.0,
)
ax.set_xlabel(r"Drive frequency $f_d$ (GHz)")
ax.set_ylabel(r"FFT frequency (GHz)")
ax.set_title(r"FFT of drive-frame cavity phase $|\alpha_{\rm demod}|\arg(\alpha_{\rm demod})$")
ax.set_ylim(0.0, fft_max)
fig.colorbar(im, ax=ax, label="Normalized FFT amplitude")
plt.tight_layout()
plt.show()

# ============================================================
# 30. LINE CUTS THROUGH FFT
# ============================================================

selected_drive_freqs = [f_tls1, f_c, f_tls2]

fig, ax = plt.subplots(figsize=(10, 6))
for fd in selected_drive_freqs:
    idx = np.argmin(np.abs(drive_freqs - fd))
    spectrum = FFT_phase_norm[idx, :]
    ax.plot(fft_freq_plot, spectrum, linewidth=1.5, label=f"$f_d={drive_freqs[idx]:.4f}$ GHz")
ax.axvline(Omega_drive, linestyle="--", linewidth=1.2, label=r"$\Omega_{\rm drive}$")
ax.set_xlim(0.0, fft_max)
ax.set_xlabel("FFT frequency (GHz)")
ax.set_ylabel("Normalized FFT amplitude")
ax.set_title("Phase-FFT line cuts")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 31. DIRECT CAVITY RINGDOWN
# ============================================================

idx_c = np.argmin(np.abs(drive_freqs - f_c))

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(tlist, IQ_magnitude[idx_c], linewidth=2.0)
ax.axvline(T_drive, linestyle="--", linewidth=1.5)
ax.set_xlabel("Time (ns)")
ax.set_ylabel(r"$|\alpha|$")
ax.set_title(f"Cavity-output ringdown at $f_d={drive_freqs[idx_c]:.4f}$ GHz")
ax.set_xlim(T_drive - 20, T_total)
plt.tight_layout()
plt.show()

# ============================================================
# 32. DIRECT DEMODULATED PHASE AT CAVITY RESONANCE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_post, alpha_demod_phase[idx_c], linewidth=1.2)
ax.set_xlabel("Time after drive switch-off (ns)")
ax.set_ylabel(r"$\arg(\alpha_{\rm demod})$")
ax.set_title(f"Drive-frame cavity phase at $f_d={drive_freqs[idx_c]:.4f}$ GHz")
plt.tight_layout()
plt.show()

# ============================================================
# 33. TRUE ZOOM OF CAVITY MAGNITUDE
# ============================================================

zoom_mask = (drive_freqs >= 3.85) & (drive_freqs <= 4.15)

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    IQ_magnitude[zoom_mask, :],
    extent=[tlist[0], tlist[-1], drive_freqs[zoom_mask][0], drive_freqs[zoom_mask][-1]],
    origin="lower",
    aspect="auto",
    cmap="inferno",
    vmin=np.min(IQ_magnitude),
    vmax=np.max(IQ_magnitude),
)
ax.axvline(T_drive, linestyle="--", linewidth=1.5)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Drive frequency (GHz)")
ax.set_title(r"Zoom: cavity-output magnitude $|\alpha|$")
fig.colorbar(im, ax=ax, label=r"$|\alpha|$")
plt.tight_layout()
plt.show()

# ============================================================
# 34. PRINT PARAMETERS
# ============================================================

tau_amp = 1.0 / (PI2 * kappa_c / 2.0)
tau_power = 1.0 / (PI2 * kappa_c)

print()
print("=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"Cavity frequency       = {f_c:.4f} GHz")
print(f"TLS 1 frequency        = {f_tls1:.4f} GHz")
print(f"TLS 2 frequency        = {f_tls2:.4f} GHz")
print(f"Cavity-TLS coupling 1  = {g_cav1:.4f} GHz")
print(f"Cavity-TLS coupling 2  = {g_cav2:.4f} GHz")
print(f"TLS-TLS Jxx            = {J_xx:.4f} GHz")
print(f"TLS-TLS Jzz            = {J_zz:.4f} GHz")
print(f"TLS drive amplitude    = {Omega_drive:.4f} GHz")
print(f"Cavity kappa           = {kappa_c:.6f} GHz")
print(f"Cavity amplitude tau   = {tau_amp:.2f} ns")
print(f"Cavity power tau       = {tau_power:.2f} ns")
print(f"Number of drive points = {Nf}")
print(f"Internal timestep      = {dt:.4f} ns")
print(f"Output timestep        = {dt_output:.4f} ns")
print("Measured observable    = alpha")
print("Demodulated observable = alpha_demod")
print("Demodulation frame     = drive frequency")
print("Demodulation           = alpha * exp(+i*omega_d*t)")
print("Phase                  = angle(alpha_demod)")
print("Envelope               = abs(alpha_demod)")
print("FFT signal             = envelope * phase")
print(f"FFT display limit      = {fft_max:.4f} GHz")
print("Phase FFTs             = ONE")
print("Direct cavity drive    = NONE")
print("=" * 70)