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

omega_tls = [3.75, 3.82]

print(f"Loading data...")

data1 = np.load("presentation_data/tempo_data_N_TLS_2_gamma_bath_0.05_drive_0.1_lam_0.02_T0.5_dt0.5_tcut5.0_strong_ohmic.npz", allow_pickle=True)
results_mark = np.real(data1['results_mark'])
results_tempo = np.real(data1['results_tempoi'])
omega_d_vals = data1['omega_d_vals']
tlist = data1['tlist']
fft_freqs_mark = data1['fft_freqs_mark']
fft_data_mark = data1['fft_data_mark']
fft_freqs_tempo = data1['fft_freqs_tempo']
fft_data_tempo = data1['fft_data_tempo']

data2 = np.load("presentation_data/heom_bctds_data_N_TLS_2_gamma_bath_0.05_drive_0.1_lam_0.02_T0.5_Nk3_depth5_strong_xcoup_zinter.npz", allow_pickle=True)
results_mark2 = np.real(data2['results_mark'])
results_heom = np.real(data2['results_heom'])
omega_d_vals2 = data2['omega_d_vals']
tlist2 = data2['tlist']
fft_freqs_mark2 = data2['fft_freqs_mark']
fft_data_mark2 = data2['fft_data_mark']
fft_freqs_heom = data2['fft_freqs_heom']
fft_data_heom = data2['fft_data_heom']

gridspec = {'width_ratios': [1, 1, 1, 0.1]}
fig, ax = plt.subplots(1, 4, figsize=(16,6), gridspec_kw=gridspec)
# normalize cmaps
vmin=min(min(np.min(fft_data_tempo), np.min(fft_data_mark)), np.min(fft_data_heom))
vmax=max(max(np.max(fft_data_tempo), np.max(fft_data_mark)), np.max(fft_data_heom))
# plot markov
im0 = ax[0].imshow(fft_data_mark.T,
                    extent=[omega_d_vals[0], omega_d_vals[-1],
                            fft_freqs_mark[0], fft_freqs_mark[-1]],
                    origin='lower', aspect='auto', cmap='Oranges',
                    vmin=vmin,
                    vmax=vmax)
ax[0].set_title("FFT of phase*env (Markovian)")
ax[0].set_xlabel("Drive Frequency (GHz)")
ax[0].set_ylabel("FFT Frequency (GHz)")

# plot heom
im1 = ax[1].imshow(fft_data_heom.T,
                    extent=[omega_d_vals[0], omega_d_vals[-1],
                            fft_freqs_heom[0], fft_freqs_heom[-1]],
                    origin='lower', aspect='auto', cmap='Oranges',
                    vmin=vmin,
                    vmax=vmax)
ax[1].set_title("FFT of phase*env (HEOM)")
ax[1].set_xlabel("Drive Frequency (GHz)")
# ax[1].set_ylabel("FFT Frequency (GHz)")

# plot TEMPO
im1 = ax[2].imshow(fft_data_tempo.T,
                    extent=[omega_d_vals[0], omega_d_vals[-1],
                            fft_freqs_tempo[0], fft_freqs_tempo[-1]],
                    origin='lower', aspect='auto', cmap='Oranges',
                    vmin=vmin,
                    vmax=vmax)
ax[2].set_title("FFT of phase*env (TEMPO)")
ax[2].set_xlabel("Drive Frequency (GHz)")
# ax[2].set_ylabel("FFT Frequency (GHz)")

# add bare eigenfrequencies as vertical lines
for i in range(3):
    ax[i].vlines(x=omega_tls, color='black', ymin=fft_freqs_tempo[0], ymax=fft_freqs_tempo[-1], linestyle='--', linewidth=0.9)

# colorbar
cb3 = fig.colorbar(im1, cax=ax[3])
cb3.set_label(r"$|\mathrm{FFT}(\phi)|$ (arb.)", labelpad=14)
plt.tight_layout()

# save and show
plt.savefig(f"presentation_data/tempo_fft_map.png")







fig, ax = plt.subplots(1, 4, figsize=(16,6), gridspec_kw=gridspec)
# normalize cmaps
vmin=min(min(np.min(results_heom[0]), np.min(results_mark[0])), np.min(results_tempo))
vmax=max(max(np.max(results_heom[0]), np.max(results_mark[0])), np.max(results_tempo))
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

# plot TEMPO
im1 = ax[2].imshow(np.transpose(results_tempo[0]),
            extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
            origin='lower', aspect='auto', cmap='inferno',
            vmin=vmin,
            vmax=vmax)
ax[2].set_title(r"$ \langle S_+S_- \rangle $ (TEMPO, square pulse)")
ax[2].set_xlabel("Drive Frequency (GHz)")
ax[2].set_ylabel("Time (ns)")

# colorbar
cb1 = fig.colorbar(im1, cax=ax[3])
cb1.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$ (arb.)", labelpad=14)
plt.tight_layout()

# save
plt.savefig(f"presentation_data/exc_map.png")


