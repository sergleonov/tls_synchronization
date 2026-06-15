import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import windows
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import oqupy
from qutip import *
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
T_total = 1600 # ns
T_drive = 100.0   # ns
dt = 0.5 # ns
tcut = 5.0
epsrel = 1e-4

SAVE_FIG = True # set to True to save figures

N_TLS = 2 # number of TLS in the system

# generate TLS frequencies
np.random.seed(17)
omega_tls = np.random.uniform(3.0, 5.0, N_TLS) # GHz
print(f"TLS frequencies: {omega_tls}")

# time list and drive frequencies
tlist = np.arange(0, T_total+dt, dt)
n_time = len(tlist)
omega_d_vals = np.linspace(3.0, 5.0, 300)

ap = argparse.ArgumentParser(description="TEMPO Simulation for single TLS")
ap.add_argument("--tag", type=str, default="", help="Tag for output files")
ap.add_argument("--npz", type=str, default="", help="Path to .npz file with precomputed results")
args = ap.parse_args()

# operators
sx = oqupy.operators.sigma("x")
sz = oqupy.operators.sigma("z")
sp = oqupy.operators.sigma("+")
sm = oqupy.operators.sigma("-")

sx_tempo = []
sz_tempo = []
sp_tempo = []
sm_tempo = []

sx_mark = []
sz_mark = []
sp_mark = []
sm_mark = []

def tensor_tempo(mats: list):
    res = mats[0]
    for i in range(1, len(mats)):
        res = np.kron(res, mats[i])
    return res

# create tempo ops
for i in range(N_TLS):
    op_list = [np.eye(2) for _ in range(N_TLS)]

    op_list[i] = sx
    sx_tempo.append(tensor_tempo(op_list))

    op_list[i] = sz
    sz_tempo.append(tensor_tempo(op_list))

    op_list[i] = sp
    sp_tempo.append(tensor_tempo(op_list))

    op_list[i] = sm
    sm_tempo.append(tensor_tempo(op_list))

# create markov ops
for i in range(N_TLS):
    op_list = [qeye(2) for _ in range(N_TLS)]

    op_list[i] = sigmax()
    sx_mark.append(tensor(op_list))
    
    op_list[i] = sigmaz()
    sz_mark.append(tensor(op_list))
    
    op_list[i] = sigmam()
    sm_mark.append(tensor(op_list))
    
    op_list[i] = sigmap()
    sp_mark.append(tensor(op_list))
    
collective_sp_tempo  = sum(sp_tempo)
collective_sm_tempo = sum(sm_tempo)
collective_exc_tempo = np.matmul(collective_sp_tempo, collective_sm_tempo)

collective_sp_mark = sum(sp_mark)
collective_sm_mark = sum(sm_mark)
collective_exc_mark = collective_sp_mark * collective_sm_mark

# bath
correlations = oqupy.PowerLawSD(alpha=lam,
                                    zeta=0.5,
                                    cutoff=gamma_bath,
                                    cutoff_type="exponential",
                                    temperature=T)
bath = oqupy.Bath(sum(sx_tempo), correlations)

# compute process tensor
tempo_params = oqupy.TempoParameters(dt=dt, tcut=tcut, epsrel=epsrel)

process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=0.0,
                                        end_time=T_total,
                                        parameters=tempo_params)

# hamiltonian
def get_hamiltonian(sx, sz):
    H = sum(0.5 * omega_tls[i] * sz[i] for i in range(N_TLS))
    for i in range(N_TLS):
        for j in range(i+1, N_TLS):
            if isinstance(sx[0], Qobj):
                H += J * sz[i] * sz[j]
            else:
                H += J * np.matmul(sz[i], sz[j])
    return H

H_tempo = get_hamiltonian(sx_tempo, sz_tempo)
H_mark = get_hamiltonian(sx_mark, sz_mark)

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
        return H_tempo + drive_coeff(t, args) * sum(sx_tempo)
    
    # initial state
    psi0 = np.array([[0] for _ in range(2*N_TLS)])
    psi0[-1] = [1] # ground state
    rho0 = np.matmul(psi0, np.transpose(psi0))
    
    system = oqupy.TimeDependentSystem(ham)

    dynamics = oqupy.compute_dynamics(process_tensor=process_tensor, 
                                      system=system,
                                      initial_state=rho0,
                                      start_time=0.0,
                                      progress_type="silent")

    t, exc_tempo = dynamics.expectations(np.matmul(collective_sp_tempo, collective_sm_tempo), real=True)
    t, sp_tempo  = dynamics.expectations(collective_sp_tempo, real=False)

    return exc_tempo, sp_tempo

def compute_mark(omega_d):

    H_full = QobjEvo(
        [H_mark,
        [sum(sx_mark), drive_coeff]],
        args={
            'omega_d': omega_d,
            'Omega': Omega_amp,
            'T_drive': T_drive
        }
    )

    evals, evecs = H_mark.eigenstates()
    psi0 = evecs[0] # initial state

    # collaps ops with temperature dependence
    n_th = []
    for i in range(N_TLS):
        n_th.append(1 / (np.exp(omega_tls[i] / T) - 1))
    c_ops = []
    for i in range(N_TLS):
        c_ops.append(np.sqrt(lam * (n_th[i] + 1)) * sm_mark[i])
        c_ops.append(np.sqrt(lam * n_th[i]) * sp_mark[i])
    
    result = mesolve(
        H_full,
        psi0,
        tlist,
        c_ops,
        e_ops=[collective_exc_mark, collective_sp_mark],
        options={"nsteps": 5000, "progress_bar": ''},
    )

    return np.real(result.expect[0]), result.expect[1]

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

def drive_coeff(t, args):
    if 0.0 <= t <= args['T_drive']:
        return 0.5 * args['Omega'] * np.cos(args['omega_d'] * t)
    else:
        return 0.0
    
def plot_fft_map(fft_freqs_mark, fft_data_mark, fft_freqs_tempo, fft_data_tempo, omega_d_vals, tag):
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=min(np.min(fft_data_tempo), np.min(fft_data_mark))
    vmax=max(np.max(fft_data_tempo), np.max(fft_data_mark))
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

    # plot TEMPO
    im1 = ax[1].imshow(fft_data_tempo.T,
                        extent=[omega_d_vals[0], omega_d_vals[-1],
                                fft_freqs_tempo[0], fft_freqs_tempo[-1]],
                        origin='lower', aspect='auto', cmap='Oranges',
                        vmin=vmin,
                        vmax=vmax)
    ax[1].set_title("FFT of phase*env (TEMPO, square pulse)")
    ax[1].set_xlabel("Drive Frequency (GHz)")
    ax[1].set_ylabel("FFT Frequency (GHz)")

    # add bare eigenfrequencies as vertical lines
    for i in range(2):
        ax[i].vlines(x=omega_tls, color='black', ymin=fft_freqs_tempo[0], ymax=fft_freqs_tempo[-1], linestyle='--', linewidth=0.9)

    # colorbar
    cb3 = fig.colorbar(im1, cax=ax[2])
    cb3.set_label(r"$|\mathrm{FFT}(\phi)|$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save and show
    if SAVE_FIG:
        os.makedirs("tempo_figures",exist_ok=True)
        plt.savefig(f"tempo_figures/tempo_fft_map_cut_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{tag}.png")

def plot_exc_map(results_mark, results_tempo, omega_d_vals, tlist, tag):
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=min(np.min(results_tempo[0]), np.min(results_mark[0]))
    vmax=max(np.max(results_tempo[0]), np.max(results_mark[0]))
    # plot markov
    im0 = ax[0].imshow(np.transpose(results_mark[0]),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin,
                vmax=vmax)
    ax[0].set_title(r"$ \langle S_+S_- \rangle $ (Markovian, square pulse)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("Time (ns)")

    # plot TEMPO
    im1 = ax[1].imshow(np.transpose(results_tempo[0]),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin,
                vmax=vmax)
    ax[1].set_title(r"$ \langle S_+S_- \rangle $ (TEMPO, square pulse)")
    ax[1].set_xlabel("Drive Frequency (GHz)")
    ax[1].set_ylabel("Time (ns)")

    # colorbar
    cb1 = fig.colorbar(im1, cax=ax[2])
    cb1.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if SAVE_FIG:
        os.makedirs("tempo_figures",exist_ok=True)
        plt.savefig(f"tempo_figures/tempo_exc_map_cut_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{tag}.png")

def plot_sp_map(results_mark, results_tempo, omega_d_vals, tlist, tag):
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=min(np.min(np.abs(results_tempo[1])), np.min(np.abs(results_mark[1])))
    vmax=max(np.max(np.abs(results_tempo[1])), np.max(np.abs(results_mark[1])))
    # plot markov
    im0 = ax[0].imshow(np.transpose(np.abs(results_mark[1])),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin,
                vmax=vmax)
    ax[0].set_title(r"$ | \langle S_+ \rangle | $ (Markovian, square pulse)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("Time (ns)")

    # plot TEMPO
    im1 = ax[1].imshow(np.transpose(np.abs(results_tempo[1])),
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='inferno',
                vmin=vmin, 
                vmax=vmax)
    ax[1].set_title(r"$ | \langle S_+ \rangle | $ (TEMPO, square pulse)")
    ax[1].set_xlabel("Drive Frequency (GHz)")
    ax[1].set_ylabel("Time (ns)")

    # colorbar
    cb2 = fig.colorbar(im1, cax=ax[2])
    cb2.set_label(r"$|\langle \sigma^{+} \rangle|$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if SAVE_FIG:
        os.makedirs("tempo_figures",exist_ok=True)
        plt.savefig(f"tempo_figures/tempo_sp_map_cut_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{tag}.png")

def plot_diff_map(results_mark, results_tempo, omega_d_vals, tlist, tag):
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, ax = plt.subplots(1, 3, figsize=(12,6), gridspec_kw=gridspec)
    # compute diffs
    exc_diff = np.transpose(results_mark[0] - results_tempo[0])
    sp_diff = np.transpose(np.abs(results_mark[1]) - np.abs(results_tempo[1]))
    # normalize cmaps
    vmin=min(np.min(exc_diff), np.min(sp_diff))
    vmax=max(np.max(exc_diff), np.max(sp_diff))
    # plot difference in total excitation
    im0 = ax[0].imshow(exc_diff,
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='bwr',
                vmin=vmin,
                vmax=vmax)
    ax[0].set_title(r"Difference in $ \langle S_+S_- \rangle $ (Markovian - TEMPO)")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("Time (ns)")

    # plot difference in sigma plus
    im1 = ax[1].imshow(sp_diff,
                extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                origin='lower', aspect='auto', cmap='bwr',
                vmin=vmin,
                vmax=vmax)
    ax[1].set_title(r"Difference in $ | \langle S_+ \rangle | $ (Markovian - TEMPO)")
    ax[1].set_xlabel("Drive Frequency (GHz)")
    ax[1].set_ylabel("Time (ns)")

    # colorbar
    cb3 = fig.colorbar(im1, cax=ax[2])
    cb3.set_label(r"Difference (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if SAVE_FIG:
        os.makedirs("tempo_figures",exist_ok=True)
        plt.savefig(f"tempo_figures/tempo_diff_map_cut_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{tag}.png")

def run_sim():
    print(f"Starting TEMPO simulation with parameters: \nN_TLS={N_TLS} \nJ={J} \nOmega_amp={Omega_amp} \nlam={lam}" +
        f"\ngamma_bath={gamma_bath} \nT={T} \ntcut={tcut} \ndt={dt} \nepsrel={epsrel} \nT_total={T_total} \nT_drive={T_drive}")

    exc_tempo = np.zeros((len(omega_d_vals), n_time))
    sp_tempo  = np.zeros((len(omega_d_vals), n_time), dtype=complex)

    exc_mark = np.zeros((len(omega_d_vals), n_time))
    sp_mark = np.zeros((len(omega_d_vals), n_time), dtype=complex)

    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

        for idx, (exc_res, sp_res) in enumerate(tqdm(executor.map(compute_tempo, omega_d_vals),
                                                total=len(omega_d_vals),
                                                desc="TEMPO Simulations")):
            exc_tempo[idx,:], sp_tempo[idx,:] = exc_res, sp_res

        results_tempo = (exc_tempo, sp_tempo)

        for idx, (exc, sp) in enumerate(tqdm(executor.map(compute_mark, omega_d_vals),
                                        total=len(omega_d_vals),
                                        desc="Markovian simulations")):
            exc_mark[idx, :], sp_mark[idx, :] = exc, sp
    
        results_mark = (exc_mark, sp_mark)
    
    print("Computing FFT data...")
    fft_freqs_tempo, fft_data_tempo = compute_fft(omega_d_vals, results_tempo[1])
    fft_freqs_mark, fft_data_mark = compute_fft(omega_d_vals, results_mark[1])

    os.makedirs("tempo_data", exist_ok=True)
    np.savez(f"tempo_data/tempo_data_N_TLS_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{args.tag}.npz",
             results_tempo=results_tempo, results_mark=results_mark, omega_d_vals=omega_d_vals, tlist=tlist,
             fft_freqs_tempo=fft_freqs_tempo, fft_data_tempo=fft_data_tempo,
             fft_freqs_mark=fft_freqs_mark, fft_data_mark=fft_data_mark)
    
    print("Plotting results...")
    plot_exc_map(results_mark, results_tempo, omega_d_vals, tlist, args.tag)
    plot_sp_map(results_mark, results_tempo, omega_d_vals, tlist, args.tag)
    plot_diff_map(results_mark, results_tempo, omega_d_vals, tlist, args.tag)
    plot_fft_map(fft_freqs_mark, fft_data_mark, fft_freqs_tempo, fft_data_tempo, omega_d_vals, args.tag)
    plt.show()
    print("Done.")


if __name__ == "__main__":
    if args.npz:
        print(f"Loading data from {args.npz}...")
        data = np.load(args.npz, allow_pickle=True)
        results_mark = data['results_mark']
        results_tempo = data['results_tempo']
        omega_d_vals = data['omega_d_vals']
        tlist = data['tlist']
        fft_freqs_mark = data['fft_freqs_mark']
        fft_data_mark = data['fft_data_mark']
        fft_freqs_tempo = data['fft_freqs_tempo']
        fft_data_tempo = data['fft_data_tempo']

        # plot results
        print("Plotting results...")
        plot_exc_map(results_mark, results_tempo, omega_d_vals, tlist, args.tag)
        plot_sp_map(results_mark, results_tempo, omega_d_vals, tlist, args.tag)
        plot_diff_map(results_mark, results_tempo, omega_d_vals, tlist, args.tag)
        plot_fft_map(fft_freqs_mark, fft_data_mark, fft_freqs_tempo, fft_data_tempo, omega_d_vals, args.tag)
        plt.show()
        print("Done.")
    else:
        run_sim()
    