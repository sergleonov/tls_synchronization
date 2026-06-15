import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import windows
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver
import os
import argparse
# ---------------------- System Parameters ----------------------
J = 0.02 # interaction strength

Omega_amp = 0.1 # drive strength

# bath parameters
lam = 0.02 # coupling strength
gamma_bath = 0.05
T = 0.5 # temperature

# solver parameters
Nk = 3
max_depth = 5

# time parameters
T_total = 1600 # ns
T_drive = 100.0   # ns
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
omega_d_vals = np.linspace(3.0, 5.0, 300)

ap = argparse.ArgumentParser(description="HEOM vs Markovian comparison for two coupled TLS under square pulse drive")
ap.add_argument("--tag", type=str, default="", help="Tag for output files")
ap.add_argument("--npz", type=str, default="", help="Path to .npz file with precomputed results")
args = ap.parse_args()

# ---------------------- Disorder ----------------------
sigma_disorder = 0.1

if DISORDER:
    for i in range(N_TLS):
        omega_tls[i] += np.random.normal(0.0, sigma_disorder)
    J += np.random.normal(0.0, sigma_disorder*0.1) # smaller disorder for J 
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

# ---------------------- Hamiltonian ----------------------
H0 = sum(0.5 * omega_tls[i] * sz[i] for i in range(N_TLS))
Hint = 0
for i in range(N_TLS):
    for j in range(i+1, N_TLS):
        Hint += J * (sz[i] * sz[j])
H_static = H0 + Hint

# ---------------------- HEOM Bath ----------------------
Q_bath = sum(sx)
bath = DrudeLorentzBath(Q_bath, lam=lam, gamma=gamma_bath, T=T, Nk=Nk)

# ---------------------- Initial state -------------------
evals, evecs = H_static.eigenstates()
psi0 = evecs[0] # initial state
rho0 = ket2dm(psi0) # initial density matrix

# ---------------------- Functions ----------------------
def compute_heom(omega_d):

    H_full = QobjEvo(
        [H_static, [sum(sx), drive_coeff]],
        args={
            'omega_d': omega_d,
            'Omega': Omega_amp,
            'T_drive': T_drive
        }
    )

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

def compute_mark(omega_d):
    H_full = QobjEvo(
        [H_static,
        [sum(sx), drive_coeff]],
        args={
            'omega_d': omega_d,
            'Omega': Omega_amp,
            'T_drive': T_drive
        }
    )

    # collaps ops with temperature dependence
    n_th = []
    for i in range(N_TLS):
        n_th.append(1 / (np.exp(omega_tls[i] / T) - 1))
    c_ops = []
    for i in range(N_TLS):
        c_ops.append(np.sqrt(lam * (n_th[i] + 1)) * sm[i])
        c_ops.append(np.sqrt(lam * n_th[i]) * sp[i])
    
    result = mesolve(
        H_full,
        psi0,
        tlist,
        c_ops,
        e_ops=[collective_excitation, collective_sp],
        options={"nsteps": 5000, "progress_bar": ''},
    )

    return np.real(result.expect[0]), result.expect[1]

def drive_coeff(t, args):
    if 0.0 <= t <= args['T_drive']:
        return 0.5 * args['Omega'] * np.cos(args['omega_d'] * t)
    else:
        return 0.0
    
def plot_fft_map(fft_freqs_mark, fft_data_mark, fft_freqs_heom, fft_data_heom, omega_d_vals, tag):
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=min(np.min(fft_data_heom), np.min(fft_data_mark))
    vmax=max(np.max(fft_data_heom), np.max(fft_data_mark))
    # plot markov
    im0 = ax[0].imshow(fft_data_mark.T,
                        extent=[omega_d_vals[0], omega_d_vals[-1],
                                fft_freqs_mark[0], fft_freqs_mark[-1]],
                        origin='lower', aspect='auto', cmap='Oranges',
                        vmin=vmin,
                        vmax=vmax)
    ax[0].set_title("FFT of phase*env (Markovian, square pulse)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("FFT Frequency (GHz)")

    # plot heom
    im1 = ax[1].imshow(fft_data_heom.T,
                        extent=[omega_d_vals[0], omega_d_vals[-1],
                                fft_freqs_heom[0], fft_freqs_heom[-1]],
                        origin='lower', aspect='auto', cmap='Oranges',
                        vmin=vmin,
                        vmax=vmax)
    ax[1].set_title("FFT of phase*env (HEOM, square pulse)")
    ax[1].set_xlabel("Drive Frequency (GHz)")
    ax[1].set_ylabel("FFT Frequency (GHz)")

    # add bare eigenfrequencies as vertical lines
    for i in range(2):
        ax[i].vlines(x=omega_tls, color='black', ymin=fft_freqs_heom[0], ymax=fft_freqs_heom[-1], linestyle='--', linewidth=0.9)

    # colorbar
    cb3 = fig.colorbar(im1, cax=ax[2])
    cb3.set_label(r"$|\mathrm{FFT}(\phi)|$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save and show
    if SAVE_FIG:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/heom_fft_map_cut_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_Nk{Nk}_depth{max_depth}_{tag}.png")

def plot_exc_map(results_mark, results_heom, omega_d_vals, tlist, tag):
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=min(np.min(results_heom[0]), np.min(results_mark[0]))
    vmax=max(np.max(results_heom[0]), np.max(results_mark[0]))
    # plot markov
    im0 = ax[0].imshow(np.transpose(results_mark[0]),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin,
                vmax=vmax)
    ax[0].set_title(r"$ \langle S_+S_- \rangle $ (Markovian, square pulse)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("Time (ns)")

    # plot heom
    im1 = ax[1].imshow(np.transpose(results_heom[0]),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin,
                vmax=vmax)
    ax[1].set_title(r"$ \langle S_+S_- \rangle $ (HEOM, square pulse)")
    ax[1].set_xlabel("Drive Frequency (GHz)")
    ax[1].set_ylabel("Time (ns)")

    # colorbar
    cb1 = fig.colorbar(im1, cax=ax[2])
    cb1.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if SAVE_FIG:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/heom_exc_map_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_Nk{Nk}_depth{max_depth}_{tag}.png")

def plot_sp_map(results_mark, results_heom, omega_d_vals, tlist, tag):
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=min(np.min(np.abs(results_heom[1])), np.min(np.abs(results_mark[1])))
    vmax=max(np.max(np.abs(results_heom[1])), np.max(np.abs(results_mark[1])))
    # plot markov
    im0 = ax[0].imshow(np.transpose(np.abs(results_mark[1])),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin,
                vmax=vmax)
    ax[0].set_title(r"$ | \langle S_+ \rangle | $ (Markovian, square pulse)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("Time (ns)")

    # plot heom
    im1 = ax[1].imshow(np.transpose(np.abs(results_heom[1])),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin, 
                vmax=vmax)
    ax[1].set_title(r"$ | \langle S_+ \rangle | $ (HEOM, square pulse)")
    ax[1].set_xlabel("Drive Frequency (GHz)")
    ax[1].set_ylabel("Time (ns)")

    # colorbar
    cb2 = fig.colorbar(im1, cax=ax[2])
    cb2.set_label(r"$|\langle \sigma^{+} \rangle|$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if SAVE_FIG:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/heom_sp_map_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_Nk{Nk}_depth{max_depth}_{tag}.png")

def plot_diff_map(results_mark, results_heom, omega_d_vals, tlist, tag):
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)
    # compute diffs
    exc_diff = np.transpose(results_mark[0] - results_heom[0])
    sp_diff = np.transpose(np.abs(results_mark[1]) - np.abs(results_heom[1]))
    # normalize cmaps
    vmin=min(np.min(exc_diff), np.min(sp_diff))
    vmax=max(np.max(exc_diff), np.max(sp_diff))
    # plot difference in total excitation
    im0 = ax[0].imshow(exc_diff,
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='bwr',
                vmin=vmin,
                vmax=vmax)
    ax[0].set_title(r"Difference in $ \langle S_+S_- \rangle $ (Markovian - HEOM)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("Time (ns)")

    # plot difference in sigma plus
    im1 = ax[1].imshow(sp_diff,
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='bwr',
                vmin=vmin,
                vmax=vmax)
    ax[1].set_title(r"Difference in $ | \langle S_+ \rangle | $(Markovian - HEOM)")
    ax[1].set_xlabel("Drive Frequency (GHz)")
    ax[1].set_ylabel("Time (ns)")

    # colorbar
    cb3 = fig.colorbar(im1, cax=ax[2])
    cb3.set_label(r"Difference (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if SAVE_FIG:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/heom_diff_map_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_Nk{Nk}_depth{max_depth}_{tag}.png")

def smooth_envelope(amplitude, fraction=0.02):
    N = len(amplitude)
    win_len = int(max(3, min(N-1, np.round(N * fraction))))
    if win_len % 2 == 0:
        win_len += 1
    kernel = np.ones(win_len) / win_len
    return np.convolve(amplitude, kernel, mode='same')

def compute_fft(omega_d, sp_t): 

    window_fn = windows.hann(n_time)
    window_rms = np.sqrt(np.mean(window_fn**2))
    N_pad = 2**13

    fft_data = []

    for idx, omega_d in enumerate(omega_d_vals):
        Splus_t = sp_t[idx, :]

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
    fmax = 0.1 # GHz
    idx_max = np.searchsorted(fft_freqs, fmax) 

    fft_data = fft_data[:, :idx_max]
    fft_freqs = fft_freqs[:idx_max]

    return fft_freqs, fft_data
# ---------------------- Simulation ----------------------
def run_full_sim():
    print(f"Starting simulations with parameters: \nN_TLS={N_TLS} \nJ={J} \nOmega_amp={Omega_amp} \nlam={lam}" +
        f"\ngamma_bath={gamma_bath} \nT={T} \nNk={Nk} \nmax_depth={max_depth} \nT_total={T_total} \nT_drive={T_drive}")

    exc_mark = np.zeros((len(omega_d_vals), n_time))
    sp_mark = np.zeros((len(omega_d_vals), n_time), dtype=complex)

    exc_heom = np.zeros((len(omega_d_vals), n_time))
    sp_heom = np.zeros((len(omega_d_vals), n_time), dtype=complex)

    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

        for idx, (exc, sp) in enumerate(tqdm(executor.map(compute_heom, omega_d_vals),
                                            total=len(omega_d_vals),
                                            desc="HEOM simulations")):
            exc_heom[idx, :] = exc
            sp_heom[idx, :] = sp
        
        results_heom = (exc_heom, sp_heom)
        
        for idx, (exc, sp) in enumerate(tqdm(executor.map(compute_mark, omega_d_vals),
                                            total=len(omega_d_vals),
                                            desc="Markovian simulations")):
            exc_mark[idx, :] = exc
            sp_mark[idx, :] = sp
        
        results_mark = (exc_mark, sp_mark)

    print("Computing FFT data...")
    # compute FFT data
    sp_mark = results_mark[1]
    sp_heom = results_heom[1]

    fft_freqs_mark, fft_data_mark = compute_fft(omega_d_vals, sp_mark)
    fft_freqs_heom, fft_data_heom = compute_fft(omega_d_vals, sp_heom)

    print("Saving data...")
    os.makedirs("bctds_data", exist_ok=True)
    np.savez(f"bctds_data/heom_bctds_data_N_TLS_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_Nk{Nk}_depth{max_depth}_{args.tag}.npz",
             results_mark=results_mark, results_heom=results_heom, omega_d_vals=omega_d_vals, tlist=tlist,
             fft_freqs_mark=fft_freqs_mark, fft_data_mark=fft_data_mark,
             fft_freqs_heom=fft_freqs_heom, fft_data_heom=fft_data_heom)

    # plot results
    print("Plotting results...")
    plot_exc_map(results_mark, results_heom, omega_d_vals, tlist, args.tag)
    plot_sp_map(results_mark, results_heom, omega_d_vals, tlist, args.tag)
    plot_diff_map(results_mark, results_heom, omega_d_vals, tlist, args.tag)
    plot_fft_map(fft_freqs_mark, fft_data_mark, fft_freqs_heom, fft_data_heom, omega_d_vals, args.tag)
    plt.show()
    print("Done.")

if __name__ == "__main__":
    if args.npz:
        print(f"Loading data from {args.npz}...")
        data = np.load(args.npz, allow_pickle=True)
        results_mark = data['results_mark']
        results_heom = data['results_heom']
        omega_d_vals = data['omega_d_vals']
        tlist = data['tlist']
        fft_freqs_mark = data['fft_freqs_mark']
        fft_data_mark = data['fft_data_mark']
        fft_freqs_heom = data['fft_freqs_heom']
        fft_data_heom = data['fft_data_heom']

        # plot results
        print("Plotting results...")
        plot_exc_map(results_mark, results_heom, omega_d_vals, tlist, args.tag)
        plot_sp_map(results_mark, results_heom, omega_d_vals, tlist, args.tag)
        plot_diff_map(results_mark, results_heom, omega_d_vals, tlist, args.tag)
        plot_fft_map(fft_freqs_mark, fft_data_mark, fft_freqs_heom, fft_data_heom, omega_d_vals, args.tag)
        plt.show()
        print("Done.")
    else:
        run_full_sim()
