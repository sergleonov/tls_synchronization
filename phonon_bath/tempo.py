import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import windows
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

sx_list = []
sz_list = []
sp_list = []
sm_list = []

def tensor(mats: list):
    res = mats[0]
    for i in range(1, len(mats)):
        res = np.kron(res, mats[i])
    return res

for i in range(N_TLS):
    op_list = [np.eye(2) for _ in range(N_TLS)]
    op_list[i] = sx
    sx_list.append(tensor(op_list))

    op_list[i] = sz
    sz_list.append(tensor(op_list))

    op_list[i] = sp
    sp_list.append(tensor(op_list))

    op_list[i] = sm
    sm_list.append(tensor(op_list))
    
collective_sp  = sum(sp_list)
collective_sm  = sum(sm_list)
collective_exc = np.matmul(collective_sp, collective_sm)

# initial state
psi0 = np.array([[0] for _ in range(2*N_TLS)])
psi0[-1] = [1] # ground state
rho0 = np.matmul(psi0, np.transpose(psi0))

# bath
correlations = oqupy.PowerLawSD(alpha=lam,
                                    zeta=2,
                                    cutoff=gamma_bath,
                                    cutoff_type="exponential",
                                    temperature=T)
bath = oqupy.Bath(sum(sx_list), correlations)

# compute process tensor
tempo_params = oqupy.TempoParameters(dt=dt, tcut=tcut, epsrel=epsrel)

process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=0.0,
                                        end_time=T_total,
                                        parameters=tempo_params)

# hamiltonian
H0 = sum(0.5 * omega_tls[i] * sz_list[i] for i in range(N_TLS))
Hint = 0
for i in range(N_TLS):
    for j in range(i+1, N_TLS):
        Hint += J * np.matmul(sz_list[i], sz_list[j])
H_static = H0 + Hint

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
        return H_static + drive_coeff(t, args) * sum(sx_list)
    
    system = oqupy.TimeDependentSystem(ham)

    dynamics = oqupy.compute_dynamics(process_tensor=process_tensor, 
                                      system=system,
                                      initial_state=rho0,
                                      start_time=0.0,
                                      progress_type="silent")

    t, exc_tempo = dynamics.expectations(np.matmul(collective_sp, collective_sm), real=True)
    t, sp_tempo  = dynamics.expectations(collective_sp, real=True)

    return exc_tempo, sp_tempo

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

def plot_fft_map(fft_freqs, fft_data, omega_d_vals, tag):
    gridspec = {'width_ratios': [1, 0.1]}
    fig, ax = plt.subplots(1, 2, figsize=(12,6), gridspec_kw=gridspec)
    # normalize cmaps
    vmin=np.min(fft_data)
    vmax=np.max(fft_data)
    # plot markov
    im0 = ax[0].imshow(fft_data.T,
                        extent=[omega_d_vals[0], omega_d_vals[-1],
                                fft_freqs[0], fft_freqs[-1]],
                        origin='lower', aspect='auto', cmap='Oranges',
                        vmin=vmin,
                        vmax=vmax)
    ax[0].set_title("FFT of phase*env")
    ax[0].set_xlabel("Drive Frequency (GHz)")
    ax[0].set_ylabel("FFT Frequency (GHz)")

    # add bare eigenfrequencies as vertical lines
    for i in range(2):
        ax[i].vlines(x=omega_tls, color='black', ymin=fft_freqs[0], ymax=fft_freqs[-1], linestyle='--', linewidth=0.9)

    # colorbar
    cb3 = fig.colorbar(im0, cax=ax[1])
    cb3.set_label(r"$|\mathrm{FFT}(\phi)|$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save and show
    if SAVE_FIG:
        os.makedirs("tempo_figures",exist_ok=True)
        plt.savefig(f"tempo_figures/tempo_fft_map_cut_N_tls_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{tag}.png")

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
    print(f"Starting TEMPO simulation with parameters: \nN_TLS={N_TLS} \nJ={J} \nOmega_amp={Omega_amp} \nlam={lam}" +
        f"\ngamma_bath={gamma_bath} \nT={T} \ntcut={tcut} \ndt={dt} \nepsrel={epsrel} \nT_total={T_total} \nT_drive={T_drive}")

    exc_list = np.zeros((len(omega_d_vals), n_time))
    sp_list  = np.zeros((len(omega_d_vals), n_time))

    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

        for idx, (exc_res, sp_res) in enumerate(tqdm(executor.map(compute_tempo, omega_d_vals),
                                                total=len(omega_d_vals),
                                                desc="TEMPO Simulations")):
            exc_list[idx,:], sp_list[idx,:] = exc_res, sp_res

        res = (exc_list, sp_list)
    
    print("Computing FFT data...")
    fft_freqs, fft_data = compute_fft(omega_d_vals, res[1])

    os.makedirs("tempo_data", exist_ok=True)
    np.savez(f"tempo_data/tempo_data_N_TLS_{N_TLS}_gamma_bath_{gamma_bath}_drive_{Omega_amp}_lam_{lam}_T{T}_dt{dt}_tcut{tcut}_{args.tag}.npz",
             results=res, omega_d_vals=omega_d_vals, tlist=tlist,
             fft_freqs=fft_freqs, fft_data=fft_data)
    
    print("Plotting results...")
    plot_exc_map(res, omega_d_vals, tlist, args.tag)
    plot_sp_map(res, omega_d_vals, tlist, args.tag)
    plot_fft_map(fft_freqs, fft_data, omega_d_vals, args.tag)
    plt.show()
    print("Done.")


if __name__ == "__main__":
    if args.npz:
        print(f"Loading data from {args.npz}...")
        data = np.load(args.npz, allow_pickle=True)
        results = data['results']
        omega_d_vals = data['omega_d_vals']
        tlist = data['tlist']
        fft_freqs = data['fft_freqs']
        fft_data = data['fft_data']

        print("Plotting results...")
        plot_exc_map(results, omega_d_vals, tlist, args.tag)
        plot_sp_map(results, omega_d_vals, tlist, args.tag)
        plot_fft_map(fft_freqs, fft_data, omega_d_vals, args.tag)
        plt.show()
        print("Done.")
    else:
        run_sim()
    