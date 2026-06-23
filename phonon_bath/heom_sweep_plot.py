import sys
import glob
import numpy as np
sys.path.append("../")
from solvers import HEOM
from bctds_plots import *

def main():
    ohms = [0.5, 1, 3]
    files = []
    for o in ohms:
        files = files + glob.glob(f"bctds_data/*power_{o}*.npz")
    print(files)

    exc, sp = [], []

    fft_freqs, fft_data = [], []

    labels = []

    heom = None

    for f in files:
        with np.load(f) as data:
            results_heom = data['results_heom']
            exc.append(np.real(results_heom[0]))  # heom_exc
            sp.append(results_heom[1])   # heom_sp
            
            fft_freqs.append(data['fft_freqs_heom'])
            fft_data.append(data['fft_data_tempo'])

            tls_freqs = data["tls_freqs"]
            J = data['J']
            Omega_amp = data['Omega_amp']
            lam = data['lam']
            gamma_bath = data['gamma_bath']
            T = data['T']
            Nk = data['Nk']
            max_depth = data['max_depth']
            T_total = data['T_total']
            T_drive = data['T_drive']
            dt = data['dt']
            n_tls = data['n_tls']
            n_freqs = data['n_freqs']
            ohmicity = data['ohmicity'] + 0
            sd_type = data['sd_type']

            labels.append(f"HEOM {ohmicity}")

            heom = HEOM(tls_freqs=tls_freqs, 
                        J=J, 
                        Omega_amp=Omega_amp, 
                        lam=lam, 
                        gamma_bath=gamma_bath, 
                        T=T, 
                        Nk=Nk, 
                        max_depth=max_depth, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt, 
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        sd_type=sd_type,
                        ohmicity=ohmicity)


    print("Plotting results...")
    plot_exc_map(exc, heom.omega_d_vals, heom.tlist, labels=labels, filename=f"sweep_exc_map_{heom}_largercut")
    plot_sp_map(sp, heom.omega_d_vals, heom.tlist, labels=labels, filename=f"sweep_sp_map_{heom}_largercut")
    plot_diff_map(exc, sp, heom.omega_d_vals, heom.tlist, labels=labels, filename=f"sweep_diff_map_{heom}_largercut")
    plot_fft_map(fft_freqs, fft_data, heom.omega_d_vals, heom.omega_tls, labels=labels, filename=f"sweep_fft_map_{heom}_largercut")
    plt.show()
    print("Done.")

if __name__ == "__main__":
    status = main()
    sys.exit(status)


