import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
from scipy.special import jv

# ---------------------- Parameters ----------------------
omega_tls = 3.75
detuning = 0.0
J = 0.01

k_max = 10

Omega_list = [0.5]

T_drive = 50.0
T_total = 100.0
dt = 0.5

tlist = np.arange(0, T_total, dt)
omega_d_vals = np.linspace(3.75, 4.0, 20)

# ---------------------- Disorder ----------------------
np.random.seed(18)
omega_tls += np.random.normal(0, 0.05)
J += np.random.normal(0, 0.01)

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

# ---------------------- STORAGE ----------------------
Q_snapshot = None

# ---------------------- SWEEP ----------------------
for Omega_amp in Omega_list:
    print("\nOmega =", Omega_amp)

    for idx, omega_d in enumerate(tqdm(omega_d_vals)):

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

        psi0 = H_static.groundstate()[1]

        result = mesolve(
            H,
            psi0,
            tlist,
            c_ops,
            [],
            args={'omega_drive': omega_d,
                'Omega': Omega_amp},
            options=Options(store_states=True)
        )

        rho_final = result.states[-1]
        if rho_final.isket:
            rho_final = ket2dm(rho_final)

        if idx == len(omega_d_vals)//2:
            Q_snapshot = husimi_Q(
                rho_final,
                theta_grid,
                phi_grid
            )

# ---------------------- PLOT ----------------------
plt.figure(figsize=(7,6))

plt.imshow(
    Q_snapshot,
    origin='lower',
    extent=[0, 2*np.pi, 0, np.pi],
    cmap='viridis'
)

plt.colorbar(label="Q(θ,φ)")
plt.xlabel("φ")
plt.ylabel("θ")
plt.title(f"Full evolution Husimi Q (drive ON→OFF), ω_d mid")
plt.tight_layout()
plt.show()

print("Done.")