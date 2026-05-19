'''
Animation of Husimi function evolution from https://arxiv.org/pdf/2511.04339
written by Salil Bedkihal
'''
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm

# ---------------------- Parameters ----------------------
omega_tls_1 = 3.0
omega_tls_2 = 4.0
J = 0.01

Omega = 0.5

T_drive = 50
T_total = 100
dt = 0.5

tlist = np.arange(0, T_total, dt)
omega_d = 3.0  # choose one frequency

# ---------------------- Disorder ----------------------
np.random.seed(18)
omega_tls_1 += np.random.normal(0, 0.05)
omega_tls_2 += np.random.normal(0, 0.05)
J += np.random.normal(0, 0.01)

print("Disorder applied.")

# ---------------------- Operators ----------------------
sx1 = tensor(sigmax(), qeye(2))
sx2 = tensor(qeye(2), sigmax())
sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())

sm1 = tensor(sigmam(), qeye(2))
sm2 = tensor(qeye(2), sigmam())

drive_op = sx1 + sx2

# ---------------------- Hamiltonian ----------------------
H0 = 0.5 * omega_tls_1 * sz1 + 0.5 * omega_tls_2 * sz2
H_int = J * sz1 * sz2+J*sx1*sx2
H_static = H0 + H_int

# ---------------------- Dissipation ----------------------
gamma = 0.002
c_ops = [np.sqrt(gamma)*(sm1+sm2)]

# ---------------------- Pulse envelope ----------------------
def pulse_env(t, args):
    return 1.0 if t <= T_drive else 0.0

def drive(t, args):
    return Omega * np.cos(omega_d * t) * pulse_env(t, args)

H = [H_static, [drive_op, drive]]

# ---------------------- Initial state ----------------------
psi0 = H_static.groundstate()[1]

# ---------------------- Solve ----------------------
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

theta_grid = np.linspace(0, np.pi, 100)
phi_grid = np.linspace(0, 2*np.pi, 100)

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
        cmap='viridis',
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

print("Done.")
