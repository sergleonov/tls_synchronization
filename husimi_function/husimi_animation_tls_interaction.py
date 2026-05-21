import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import windows
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver
import os
import argparse

# ffmpeg -framerate 10 -i animation/husimi_frame_%03d.png -c:v libx264 -pix_fmt yuv420p husimi_animation.mp4

# ---------------------- System Parameters ----------------------
J = 0.01 # interaction strength

Omega_amp = 0.2 # drive strength

omega_d = 3.75 # drive frequency

# bath parameters
lam = 0.02 # coupling strength
gamma_bath = 0.05
T = 0.5 # temperature

# solver parameters
Nk = 3
max_depth = 5

# time parameters
T_total = 1000 # ns
T_drive = 60.0   # ns
dt = 0.5 # ns

DISORDER = True # set to True to include disorder in the system parameters
SAVE_FIG = True # set to True to save figures

N_TLS = 2 # number of TLS in the system

# generate TLS frequencies
np.random.seed(17)
omega_tls = np.random.uniform(3.0, 5.0, N_TLS) # GHz
print(f"TLS frequencies: {omega_tls}")

# time list and drive frequencies
tlist = np.arange(0, T_total, dt)
n_time = len(tlist)

# ---------------------- Disorder ----------------------
sigma_disorder = 0.1

if DISORDER:
    for i in range(N_TLS):
        omega_tls[i] += np.random.normal(0.0, sigma_disorder)
    J += np.random.normal(0.0, sigma_disorder)
    print(f"Disordered parameters: J={J}, omega_tls={omega_tls}")


# ---------------------- Operators ----------------------
sx = []
sz = []
sm = []
sp = []

for i in range(N_TLS):
    op_list = [qeye(2) for _ in range(N_TLS)]
    op_list[i] = sigmax()
    sx.append(tensor(op_list))
    
    op_list[i] = sigmaz()
    sz.append(tensor(op_list))
    
    op_list[i] = sigmam()
    sm.append(tensor(op_list))
    
    op_list[i] = sigmap()
    sp.append(tensor(op_list))

collective_sp = sum(sp)
collective_sm = sum(sm)
collective_excitation = collective_sp * collective_sm

# collapse operators
n_th = []
for i in range(N_TLS):
    n_th.append(1 / (np.exp(omega_tls[i] / T) - 1))
c_ops = []
for i in range(N_TLS):
    c_ops.append(np.sqrt(lam * (n_th[i] + 1)) * sm[i])
    c_ops.append(np.sqrt(lam * n_th[i]) * sp[i])

# ---------------------- Square pulse ----------------------
def drive_coeff(t, args):
    if 0.0 <= t <= args['T_drive']:
        return args['Omega'] * np.cos(args['omega_d'] * t)
    else:
        return 0.0
    
# ---------------------- Hamiltonian ----------------------
H0 = sum(0.5 * omega_tls[i] * sz[i] for i in range(N_TLS))
Hint = 0
for i in range(N_TLS):
    for j in range(i+1, N_TLS):
        Hint += J * (sx[i] * sx[j])
H_static = H0 + Hint

H_full = QobjEvo(
    [H_static,
    [sum(sx), drive_coeff]],
    args={
        'omega_d': omega_d,
        'Omega': Omega_amp,
        'T_drive': T_drive
    }
)

# ---------------------- HEOM Bath ----------------------
Q_bath = sum(sx)

bath = DrudeLorentzBath(Q_bath, lam=lam, gamma=gamma_bath, T=T, Nk=Nk)

# ---------------------- Square pulse ----------------------
def drive_coeff(t, args):
    if 0.0 <= t <= args['T_drive']:
        return args['Omega'] * np.cos(args['omega_d'] * t)
    else:
        return 0.0
    
#-------------Initial state--------------
evals, evecs = H_static.eigenstates()
psi0 = evecs[0] # initial state
rho0 = ket2dm(psi0) # initial density matrix

# ---------------------- Solve ----------------------
print("Solving dynamics...")

solver = HEOMSolver(
        H_full,
        [bath],
        max_depth=max_depth,
        options={"nsteps": 5000, "progress_bar": '', "store_states": True},
    )

result_heom = solver.run(
    rho0,
    tlist,
    e_ops=[collective_excitation, collective_sp]
)

result_mark = mesolve(
    H_full,
    psi0,
    tlist,
    c_ops,
    e_ops=[collective_excitation, collective_sp],
    options={"nsteps": 5000, "progress_bar": '', "store_states": True},
)
print("Done solving dynamics.")
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

theta_grid = np.linspace(0, np.pi, 100)
phi_grid = np.linspace(0, 2*np.pi, 100)

# ---------------------- Compute Q(t) ----------------------
print("Computing Husimi-Q(t)...")

Qt_mark = np.zeros((len(tlist), len(theta_grid), len(phi_grid)))
Qt_heom = np.zeros((len(tlist), len(theta_grid), len(phi_grid)))

for t_idx, state in enumerate(tqdm(result_mark.states)):

    rho = state
    if rho.isket:
        rho = ket2dm(rho)

    rho1 = ptrace(rho, 0)
    Qt_mark[t_idx] = husimi_Q(rho1, theta_grid, phi_grid)


for t_idx, state in enumerate(tqdm(result_heom.states)):
    
    rho = state
    if rho.isket:
        rho = ket2dm(rho)

    rho1 = ptrace(rho, 0)
    Qt_heom[t_idx] = husimi_Q(rho1, theta_grid, phi_grid)

print("Done computing Husimi-Q(t).")
# ---------------------- Animation ----------------------
print("Generating animation...")
gridspec = {'width_ratios': [1, 1, 0.1]}
fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)

pause_time = 0.1

for t_idx in range(n_time):

    # clear frame
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    # add labels
    ax[0].set_title("Markovian")
    ax[1].set_title("Non-Markovian")
    ax[0].set_xlabel(r"$\phi$")
    ax[1].set_xlabel(r"$\phi$")
    ax[0].set_ylabel(r"$\theta$")
    # add data
    im0 = ax[0].imshow(
        Qt_mark[t_idx],
        origin='lower',
        extent=[0, 2*np.pi, 0, np.pi],
        cmap='inferno'
    )

    im1 = ax[1].imshow(
        Qt_heom[t_idx],
        origin='lower',
        extent=[0, 2*np.pi, 0, np.pi],
        cmap='inferno'
    )
    # colorbar
    cb1 = fig.colorbar(im1, cax=ax[2])
    cb1.set_label(r"$Q(\theta,\phi)$")
    plt.tight_layout()
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
    if SAVE_FIG:
        os.makedirs("animation",exist_ok=True)
        plt.savefig(f"animation/husimi_frame_{t_idx:04d}.png")
    plt.pause(pause_time)
