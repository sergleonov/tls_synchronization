import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import oqupy
import os
import argparse
# ---------------------- System Parameters ----------------------
J = 0.02 # interaction strength

Omega_amp = 0.1 # drive strength

# bath parameters
lam = 0.02 # coupling strength
gamma_bath = 0.05
T = 0.5 # temperature

# time parameters
T_total = 400 # ns
T_drive = 100.0   # ns
dt = 2.0 # ns
tcut = None
epsrel = 1e-3

SAVE_FIG = True # set to True to save figures

N_TLS = 2 # number of TLS in the system

# generate TLS frequencies
np.random.seed(17)
omega_tls = np.random.uniform(3.0, 5.0, N_TLS) # GHz
print(f"TLS frequencies: {omega_tls}")

# time list and drive frequencies
tlist = np.arange(0, T_total+dt, dt)
n_time = len(tlist)
omega_d_vals = np.linspace(3.0, 5.0, 100)

ap = argparse.ArgumentParser(description="TEMPO Simulation for single TLS")
ap.add_argument("--tag", type=str, default="", help="Tag for output files")
ap.add_argument("--npz", type=str, default="", help="Path to .npz file with precomputed results")
args = ap.parse_args()

# operators
sx = oqupy.operators.sigma("x")
sp = oqupy.operators.sigma("+")
sm = oqupy.operators.sigma("-")
sz = oqupy.operators.sigma("z")
rho0 = oqupy.operators.spin_dm("z-")

def drive_coeff(t, args):
    if 0.0 <= t <= args['T_drive']:
        return 0.5 * args['Omega'] * np.cos(args['omega_d'] * t)
    else:
        return 0.0

def compute_tempo(omega_d):
    args={
            'omega_d': omega_d,
            'Omega': Omega_amp,
            'T_drive': T_drive
        }
    
    def ham(t):
        return drive_coeff(t, args) * sx + 0.5 * omega_tls[0] * sz
    
    system = oqupy.TimeDependentSystem(ham)
    correlations = oqupy.PowerLawSD(alpha=lam,
                                    zeta=2,
                                    cutoff=gamma_bath,
                                    cutoff_type="exponential",
                                    temperature=T)
    bath = oqupy.Bath(sx, correlations)
    tempo_params = oqupy.TempoParameters(dt=dt, tcut=tcut, epsrel=epsrel)

    tempo_sys = oqupy.Tempo(system=system,
                            bath=bath,
                            initial_state=rho0,
                            start_time=0.0,
                            parameters=tempo_params)
    dynamics = tempo_sys.compute(end_time=T_total, progress_type="silent")

    global sp, sm
    t, exc_tempo = dynamics.expectations(np.matmul(sp, sm), real=True)
    t, sp_tempo  = dynamics.expectations(sp, real=False)

    return exc_tempo, sp_tempo

def plot_exc_map(res, omega_d_vals, tlist, tag):
    gridspec = {'width_ratios': [1, 0.1]}
    fig, ax = plt.subplots(1, 2, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=np.min(res[0])
    vmax=np.max(res[0])
    # plot markov
    im0 = ax[0].imshow(np.transpose(res[0]),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin,
                vmax=vmax)
    ax[0].set_title(r"$\langle S_+S_-\rangle$ (TEMPO, square pulse)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("Time (ns)")


    # colorbar
    cb1 = fig.colorbar(im0, cax=ax[1])
    cb1.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if SAVE_FIG:
        os.makedirs("tempo_figures",exist_ok=True)
        plt.savefig(f"tempo_figures/tempo_exc_map_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{tag}.png")

def plot_sp_map(res, omega_d_vals, tlist, tag):
    gridspec = {'width_ratios': [1, 0.1]}
    fig, ax = plt.subplots(1, 2, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=np.min(np.abs(res[1]))
    vmax=np.max(np.abs(res[1]))
    # plot markov
    im0 = ax[0].imshow(np.transpose(np.abs(res[1])),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin,
                vmax=vmax)
    ax[0].set_title(r"$\langle S_+\rangle$ (TEMPO, square pulse)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("Time (ns)")


    # colorbar
    cb1 = fig.colorbar(im0, cax=ax[1])
    cb1.set_label(r"$\langle \sigma^{+} \rangle$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if SAVE_FIG:
        os.makedirs("tempo_figures",exist_ok=True)
        plt.savefig(f"tempo_figures/tempo_sp_map_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{tag}.png")

def run_sim():
    exc_list = np.zeros((len(omega_d_vals), n_time))
    sp_list  = np.zeros((len(omega_d_vals), n_time))

    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

        for idx, (exc, sp_res) in enumerate(tqdm(executor.map(compute_tempo,omega_d_vals),
                                         total=len(omega_d_vals),
                                         desc="TEMPO Simulation")):
            exc_list[idx,:], sp_list[idx,:] = exc, sp_res

        res = (exc_list, sp_list)
    
    plot_exc_map(res, omega_d_vals, tlist, args.tag)
    plot_sp_map(res, omega_d_vals, tlist, args.tag)
    plt.show()


if __name__ == "__main__":
    run_sim()

    