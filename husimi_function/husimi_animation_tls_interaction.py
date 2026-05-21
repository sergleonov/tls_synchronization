import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import windows
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver
import os
import matplotlib.animation as animation
import argparse
import sys

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

DISORDER = False # set to True to include disorder in the system parameters
SAVE_FIG = True # set to True to save figures

N_TLS = 2 # number of TLS in the system

# generate TLS frequencies
np.random.seed(17)
omega_tls = np.random.uniform(3.0, 5.0, N_TLS) # GHz
print(f"TLS frequencies: {omega_tls}")

# time list and drive frequencies
tlist = np.arange(0, T_total, dt)
n_time = len(tlist)

# husimi Q resolution
theta_grid = np.linspace(0, np.pi, 100)
phi_grid = np.linspace(0, 2*np.pi, 100)

# disorder 
sigma_disorder = 0.1

if DISORDER:
    for i in range(N_TLS):
        omega_tls[i] += np.random.normal(0.0, sigma_disorder)
    J += np.random.normal(0.0, sigma_disorder*0.1) # smaller disorder for J 
    print(f"Disordered parameters: J={J}, omega_tls={omega_tls}")

# ----------------------- Operators ----------------------
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
    
# ---------------------- Hamiltonian ----------------------
H0 = sum(0.5 * omega_tls[i] * sz[i] for i in range(N_TLS))
Hint = 0
for i in range(N_TLS):
    for j in range(i+1, N_TLS):
        Hint += J * (sx[i] * sx[j])
H_static = H0 + Hint

# ---------------------- HEOM Bath ----------------------
Q_bath = sum(sx)
bath = DrudeLorentzBath(Q_bath, lam=lam, gamma=gamma_bath, T=T, Nk=Nk)

# ---------------------- Initial state -------------------
evals, evecs = H_static.eigenstates()
psi0 = evecs[0] # initial state
rho0 = ket2dm(psi0) # initial density matrix

# ---------------------- Functions ----------------------
def compute_husimi_Q(state):
    rho = state
    if rho.isket:
        rho = ket2dm(rho)

    rho1 = ptrace(rho, 0)
    return husimi_Q(rho1, theta_grid, phi_grid)

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

def drive_coeff(t, args):
    if 0.0 <= t <= args['T_drive']:
        return 0.5 * args['Omega'] * np.cos(args['omega_d'] * t)
    else:
        return 0.0
    
def generate_animation(Qt_mark, Qt_heom, tlist):
    print("Generating animation...")
    gridspec = {'width_ratios': [1, 1, 0.05]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)

    vmax = max(np.max(Qt_mark), np.max(Qt_heom))
    t_idx = 0

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
        cmap='inferno',
        vmin=0, vmax=vmax
    )

    im1 = ax[1].imshow(
        Qt_heom[t_idx],
        origin='lower',
        extent=[0, 2*np.pi, 0, np.pi],
        cmap='inferno',
        vmin=0, vmax=vmax
    )
    # colorbar
    cb1 = fig.colorbar(im1, cax=ax[2])
    cb1.set_label(r"$Q(\theta,\phi)$")
    plt.tight_layout()

    def update(frame):
        t_idx = frame

        # clear frame
        ax[0].clear()
        ax[1].clear()

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
            cmap='inferno',
            vmin=0,
            vmax=vmax
        )

        im1 = ax[1].imshow(
            Qt_heom[t_idx],
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

    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_time, interval=30)
    os.makedirs("husimi_animation",exist_ok=True)
    ani.save(f"husimi_animation/husimi_animation_NTLS{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_Nk{Nk}_depth{max_depth}.mp4", writer='ffmpeg', fps=10)
    print("Animation saved.")

def run_full_sim():

    # build hamiltonian
    H_full = QobjEvo(
        [H_static,
        [sum(sx), drive_coeff]],
        args={
            'omega_d': omega_d,
            'Omega': Omega_amp,
            'T_drive': T_drive
        }
    )

    print("Solving dynamics...")

    # solve heom
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

    # solve markovian
    result_mark = mesolve(
        H_full,
        psi0,
        tlist,
        c_ops,
        e_ops=[collective_excitation, collective_sp],
        options={"nsteps": 5000, "progress_bar": '', "store_states": True},
    )
    print("Done solving dynamics.")

    # compute husimi Q
    Qt_mark = np.zeros((len(tlist), len(theta_grid), len(phi_grid)))
    Qt_heom = np.zeros((len(tlist), len(theta_grid), len(phi_grid)))

    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
        
        for t_idx, Q in enumerate(tqdm(executor.map(compute_husimi_Q,result_mark.states), 
                                        total=len(result_mark.states), 
                                        desc="Markovian Husimi-Q")):
            Qt_mark[t_idx] = Q


        for t_idx, Q in enumerate(tqdm(executor.map(compute_husimi_Q, result_heom.states), 
                                        total=len(result_heom.states),
                                        desc="HEOM Husimi-Q")):
            Qt_heom[t_idx] = Q

    print("Saving data...")
    os.makedirs("husimi_data", exist_ok=True)
    np.savez(f"husimi_data/husimi_data_NTLS{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_Nk{Nk}_depth{max_depth}.npz", Qt_mark=Qt_mark, Qt_heom=Qt_heom, theta_grid=theta_grid, phi_grid=phi_grid, tlist=tlist)
    print("Data saved successfully.")

    # generate animation
    generate_animation(Qt_mark, Qt_heom, tlist)
    sys.exit(0)

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Husimi Q-function animation for interacting TLS")
    parser.add_argument("--npz", type=str, default=None, help="Path to .npz file containing precomputed Husimi Q data")
    args = parser.parse_args()
    if args.npz:
        print(f"Loading precomputed data from {args.npz}...")
        data = np.load(args.npz)
        Qt_mark = data['Qt_mark']
        Qt_heom = data['Qt_heom']
        tlist = data['tlist']
        print("Data loaded successfully.")
        generate_animation(Qt_mark, Qt_heom, tlist)
    else: 
        run_full_sim()