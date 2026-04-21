'''
This script generates snapshot at the final time of the evolution of the Husimi function
 for a hamiltonian from eq. 14 in https://arxiv.org/pdf/2511.04339 with markovian assumption
'''

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
from scipy.special import jv

# ---------------------- Parameters ----------------------
omega_tls = 3.75
detuning = 0.0

k_max = 10

Omega_amp = 0.5
omega_d = 3.0

T_drive = 50.0
T_total = 100.0
dt = 0.5

tlist = np.arange(0, T_total, dt)
omega_d_vals = np.linspace(3.75, 4.0, 20)

# ---------------------- Disorder ----------------------
np.random.seed(18)
omega_tls += np.random.normal(0, 0.05)

print("Disorder applied.")

# ---------------------- Operators ----------------------
sx = sigmax()
sy = sigmay()
sz = sigmaz()

sm = sigmam()

# ---------------------- Dissipation ----------------------
gamma = 0.002
c_ops = [np.sqrt(gamma)*sm]

# ---------------------- Pulse envelope ----------------------
def pulse_env(t, args):
    return 1.0 if t <= T_drive else 0.0

#-------------Hamiltonian--------------
def drive_z(t, args):
    z_drive = 0
    for k in range(1, k_max+1):
        z_drive += omega_tls * jv(2*k, Omega_amp/omega_d) * np.cos(2*k * omega_d * t) * pulse_env(t, args)
    return z_drive

def drive_y(t, args):
    y_drive = 0
    for k in range(0, k_max+1):
        y_drive += omega_tls * jv(2*k + 1, Omega_amp/omega_d) * np.sin((2*k + 1) * omega_d * t) * pulse_env(t, args)
    return y_drive

H_static = 0.5 * omega_tls * jv(0, Omega_amp/omega_d) * sz + 0.5* detuning * sx

H = [H_static, [sz, drive_z], [sy, drive_y]]

#-------------Initial state--------------
psi0 = H_static.groundstate()[1]

#-------------Solve--------------
result = mesolve(
    H,
    psi0,
    tlist,
    c_ops,
    [],
    options=Options(store_states=True)
)

# ---------------------- Spin coherent ----------------------
def spin_coherent(theta, phi):
    return (np.cos(theta/2) * basis(2,0) +
            np.exp(1j*phi) * np.sin(theta/2) * basis(2,1))

def husimi_Q(rho, theta_grid, phi_grid):
    Q = np.zeros((len(theta_grid), len(phi_grid)))

    for i, th in enumerate(theta_grid):
        for j, ph in enumerate(phi_grid):
            psi = spin_coherent(th, ph)
            Q[i, j] = (ket2dm(psi) * rho).tr().real / (2*np.pi)

    return Q

theta_grid = np.linspace(0, np.pi, 60)
phi_grid = np.linspace(0, 2*np.pi, 60)

# ---------------------- Compute Q(t) ----------------------
print("Computing Husimi-Q(t)...")

Q_time = np.zeros((len(tlist), len(theta_grid), len(phi_grid)))

for t_idx, state in enumerate(tqdm(result.states)):

    rho = state
    if rho.isket:
        rho = ket2dm(rho)

    rho1 = ptrace(rho, 0)
    Q_time[t_idx] = husimi_Q(rho1, theta_grid, phi_grid)

# ---------------------- Animation ----------------------
print("Starting animation...")

plt.figure(figsize=(7,6))

frames = np.arange(0, len(tlist), 1)

vmax = np.max(Q_time)

pause_time = 0.1

for t_idx in frames:

    plt.clf()

    plt.imshow(
        Q_time[t_idx],
        origin='lower',
        extent=[0, 2*np.pi, 0, np.pi],
        cmap='inferno',
        vmin=0,
        vmax=vmax
    )

    plt.colorbar(label="Q(θ,φ)")
    plt.xlabel("φ")
    plt.ylabel("θ")

    t_now = tlist[t_idx]

    # ON / OFF label
    if t_now <= T_drive:
        phase = "DRIVE ON"
        color = 'green'
    else:
        phase = "DRIVE OFF"
        color = 'red'

    plt.title(f"t = {t_now:.1f} ns | {phase}", color=color)

    plt.pause(pause_time)

plt.show()
