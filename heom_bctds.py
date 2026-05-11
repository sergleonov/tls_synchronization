import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import windows
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver
import os
# ---------------------- System Parameters ----------------------
omega_tls_1 = 3.75
omega_tls_2 = 3.82
J = 0.02 # interaction strength

Omega_amp = 0.2

T_total = 150 # ns
T_drive = 100.0   # ns
dt = 0.5 # ns
tlist = np.arange(0, T_total, dt)
n_time = len(tlist)
omega_d_vals = np.linspace(3.0, 5.0, 200)


# ---------------------- Disorder ----------------------
np.random.seed(17)
sigma_disorder = 0.1
delta1 = np.random.normal(0.0, sigma_disorder)
delta2 = np.random.normal(0.0, sigma_disorder)
J_dis = J + np.random.normal(0.0, 0.1)

omega_tls_1_dis = omega_tls_1 + delta1
omega_tls_2_dis = omega_tls_2 + delta2

print(f"Disorder: Δ1={delta1:+.3f}, Δ2={delta2:+.3f}, ΔJ={J_dis-J:+.3f}")

# ---------------------- Operators ----------------------
sx1 = tensor(sigmax(), qeye(2))
sx2 = tensor(qeye(2), sigmax())
sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())

sm1 = tensor(sigmam(), qeye(2))
sm2 = tensor(qeye(2), sigmam())
sp1 = tensor(sigmap(), qeye(2))
sp2 = tensor(qeye(2), sigmap())

collective_sp = sp1 + sp2
collective_sm = sm1 + sm2
collective_excitation = collective_sp * collective_sm

# ---------------------- Hamiltonian ----------------------
H0 = 0.5 * omega_tls_1_dis * sz1 + 0.5 * omega_tls_2_dis * sz2
Hint = J_dis * sx1 * sx2
H_static = H0 + Hint

# ---------------------- HEOM Bath ----------------------
Q_bath = sx1 + sx2 # coupling operator

lam = 0.001   # coupling strength
gamma_bath = max(omega_tls_1_dis, omega_tls_2_dis) # bath cutoff frequency
T = 0.5   # temperature

Nk = 3
max_depth = 5

bath = DrudeLorentzBath(Q_bath, lam=lam, gamma=gamma_bath, T=T, Nk=Nk)

# ---------------------- Square pulse ----------------------
def drive_coeff(t, args):
    if 0.0 <= t <= args['T_drive']:
        return args['Omega'] * np.cos(args['omega_d'] * t)
    else:
        return 0.0

# ---------------------- Simulation ----------------------
def compute_heom(omega_d):

    H_full = QobjEvo(
        [H_static, [sx1 + sx2, drive_coeff]],
        args={
            'omega_d': omega_d,
            'Omega': Omega_amp,
            'T_drive': T_drive
        }
    )

    # initial state
    evals, evecs = H_static.eigenstates()
    rho0 = ket2dm(evecs[0])

    solver = HEOMSolver(
        H_full,
        [bath],
        max_depth=max_depth,
        options={"nsteps": 5000, "progress_bar": ''},
    )

    result = solver.run(
        rho0,
        tlist,
        e_ops=[collective_excitation, collective_sp]
    )

    return np.real(result.expect[0]), result.expect[1]

print(f"\nRunning Ω = {Omega_amp} GHz ...")

exc_data = np.zeros((len(omega_d_vals), n_time))
sp_data = np.zeros((len(omega_d_vals), n_time), dtype=complex)

with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

    for idx, (exc, sp) in enumerate(tqdm(executor.map(compute_heom, omega_d_vals),
                                         total=len(omega_d_vals),
                                         desc="HEOM simulations")):
        exc_data[idx, :] = exc
        sp_data[idx, :] = sp

results_all = (exc_data, sp_data)

print("\nHEOM simulation complete.\n")

# ---------------------- Plot ⟨S+S-⟩ ----------------------
os.makedirs("bctds",exist_ok=True)

fig, ax = plt.subplots(figsize=(7,5))
im = ax.imshow(np.transpose(results_all[0]),
               extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
               origin='lower', aspect='auto', cmap='inferno')
ax.set_title("⟨S₊S₋⟩ (HEOM, square pulse)")
ax.set_xlabel("Drive Frequency (GHz)")
ax.set_ylabel("Time (ns)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("bctds/heom_exc_map.png")

# ---------------------- Plot |⟨S+⟩| ----------------------
fig, ax = plt.subplots(figsize=(7,5))
im = ax.imshow(np.transpose(np.abs(results_all[1])),
               extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
               origin='lower', aspect='auto', cmap='inferno')
ax.set_title("|⟨S₊⟩| (HEOM, square pulse)")
ax.set_xlabel("Drive Frequency (GHz)")
ax.set_ylabel("Time (ns)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("bctds/heom_sp_map.png")

# ---------------------- FFT prep ----------------------
window_fn = windows.hann(n_time)
window_rms = np.sqrt(np.mean(window_fn**2))
N_pad = 2**13

def smooth_envelope(amplitude, fraction=0.02):
    N = len(amplitude)
    win_len = int(max(3, min(N-1, np.round(N * fraction))))
    if win_len % 2 == 0:
        win_len += 1
    kernel = np.ones(win_len) / win_len
    return np.convolve(amplitude, kernel, mode='same')

# ---------------------- FFT ----------------------
fft_data = []
sp_data = results_all[1]

for idx, omega_d in enumerate(omega_d_vals):
    Splus_t = sp_data[idx, :]

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
fmax = 1 # GHz
idx_max = np.searchsorted(fft_freqs, fmax) 

fft_data = fft_data[:, :idx_max]
fft_freqs = fft_freqs[:idx_max]

# plot cropped data
fig, ax = plt.subplots(figsize=(7,5))
im = ax.imshow(fft_data.T,
               extent=[omega_d_vals[0], omega_d_vals[-1],
                       fft_freqs[0], fft_freqs[-1]],
               origin='lower', aspect='auto', cmap='Oranges')
ax.set_title("FFT of phase*env (HEOM, square pulse)")
ax.set_xlabel("Drive Frequency (GHz)")
ax.set_ylabel("FFT Frequency (GHz)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("bctds/heom_fft_map.png")
plt.show()
print("Done.")