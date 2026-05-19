'''
This script generates animations of the evolution of the Husimi function
 for a hamiltonian from eq. 14 in https://arxiv.org/pdf/2511.04339
'''

# ffmpeg -framerate 10 -i animation/husimi_frame_%03d.png -c:v libx264 -pix_fmt yuv420p husimi_animation.mp4

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
from scipy.special import jv
from qutip.solver.heom import DrudeLorentzPadeBath
from qutip.solver.heom import HEOMSolver

# ---------------------- Parameters ----------------------
omega_tls = 3.75 # tls frequency
detuning = 0.0 

k_max = 10 # maximum bessel index

Omega_amp = 0.5 # drive amplitude
omega_d = 3.75 # drive frequency

T_drive = 40.0 # drive time
T_total = 200.0 # total time
dt = 0.5 # time step
tlist = np.arange(0, T_total, dt)
time_steps = len(tlist)

# solver parameters:
Nk = 2 # expansion terms
solver_steps = 10000
max_depth = 5  # maximum hierarchy depth to retain

# coupling operator
coupling_op = sigmax()
Temp = 0.5  # K temperature
lam = 0.02  # coupling strength
omega_cut = omega_tls # GHz cut off frequency

SAVE_ANIMATION = True

# ---------------------- Disorder ----------------------
np.random.seed(18)
# omega_tls += np.random.normal(0, 0.05)

# print("Disorder applied.")

# ---------------------- Operators ----------------------
sx = sigmax()
sy = sigmay()
sz = sigmaz()

sm = sigmam()

c_ops = [np.sqrt(lam)*sm] # collapse op

# ---------------------- Pulse envelope ----------------------
def pulse_env(t, args):
    return 1.0 if t <= T_drive else 0.0

#-------------Markovian Hamiltonian--------------
# eq. 14 from the paper
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

H0_mark = 0.5 * omega_tls * jv(0, Omega_amp/omega_d) * sz + 0.5* detuning * sx

H = [H0_mark, [sz, drive_z], [sy, drive_y]]

#-------------Non-Markovian Hamiltonian--------------
# time independent part of the Hamiltonian:
H_sys = 0.5 * omega_tls * sigmaz() + 0.5 * detuning * sigmax() # static Hamiltonian
# time dependent part of the Hamiltonian:
def H_coeff(t, args):
    return 0.5 * Omega_amp * np.cos(omega_d * t) * pulse_env(t, args)
H_non_mark = [H_sys, [sigmax(), H_coeff]]

# total Hamiltonian
H_non_mark = QobjEvo(H_non_mark)

# define bath and solver
bath = DrudeLorentzPadeBath(coupling_op, lam=lam, gamma=omega_cut, T=Temp, Nk=Nk)
options = {"nsteps": time_steps, "progress_bar": ''}
heom_solver = HEOMSolver(H_non_mark, bath, max_depth=max_depth, options=options)

#-------------Initial state--------------
psi0 = H0_mark.groundstate()[1]
rho0 = psi0 * psi0.dag() # initial density matrix

#-------------Solve--------------
# Markovian dynamics using Lindblad master equation
result_mark = mesolve(
    H,
    psi0,
    tlist,
    c_ops,
    [],
    options=Options(store_states=True)
)

# Non-Markovian dynamics using HEOM
result_non_mark = heom_solver.run(rho0, tlist)

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

theta_grid = np.linspace(0, np.pi, 80)
phi_grid = np.linspace(0, 2*np.pi, 80)

# ---------------------- Compute Q(t) ----------------------
print("Computing Markovian Husimi-Q(t)...")

Qt_mark = np.zeros((len(tlist), len(theta_grid), len(phi_grid)))

for t_idx, state in enumerate(tqdm(result_mark.states)):

    rho = state
    if rho.isket:
        rho = ket2dm(rho)

    rho1 = ptrace(rho, 0)
    Qt_mark[t_idx] = husimi_Q(rho1, theta_grid, phi_grid)

print("Computing Non-Markovian Husimi-Q(t)...")

Qt_non_mark = np.zeros((len(tlist), len(theta_grid), len(phi_grid)))

for t_idx, state in enumerate(tqdm(result_non_mark.states)):
    
    rho = state
    if rho.isket:
        rho = ket2dm(rho)

    rho1 = ptrace(rho, 0)
    Qt_non_mark[t_idx] = husimi_Q(rho1, theta_grid, phi_grid)

# ---------------------- Animation ----------------------
print("Generating animation...")

fig, ax = plt.subplots(1, 2, figsize=(12,5))

vmax = max(np.max(Qt_mark), np.max(Qt_non_mark))

pause_time = 0.1

cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=vmax)), ax=ax, label="Q(θ,φ)")

for t_idx in range(time_steps):

    # clear frame
    ax[0].clear()
    ax[1].clear()

    # add labels
    ax[0].set_title("Markovian")
    ax[1].set_title("Non-Markovian")
    ax[0].set_xlabel("φ")
    ax[1].set_xlabel("φ")
    ax[0].set_ylabel("θ")

    # add data
    im0 = ax[0].imshow(
        Qt_mark[t_idx],
        origin='lower',
        extent=[0, 2*np.pi, 0, np.pi],
        cmap='inferno',
        vmin=0,
        vmax=vmax
    )

    im1 = ax[1].imshow(
        Qt_non_mark[t_idx],
        origin='lower',
        extent=[0, 2*np.pi, 0, np.pi],
        cmap='inferno',
        vmin=0,
        vmax=vmax
    )

    # label time of the frame
    t_now = tlist[t_idx]

    if t_now <= T_drive:
        phase = "DRIVE ON"
        color = 'green'
    else:
        phase = "DRIVE OFF"
        color = 'red'

    fig.suptitle(f"Husimi Q(t) plot for Markovian and Non-Markovian time evolution\nt = {t_now:.1f} ns | {phase}", color=color)

    # save if needed
    if SAVE_ANIMATION:
        plt.savefig(f"animation/husimi_frame_{t_idx:04d}.png")
    plt.pause(pause_time)
plt.show()
