import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import oqupy
from qutip import *
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver
from functools import partial
import os


class Solver:
    def __init__(self, 
                 tls_freqs=None, 
                 J=0.02, 
                 Omega_amp=0.1, 
                 lam=0.02, 
                 gamma_bath=0.05, 
                 T=0.5, 
                 T_total=1600, 
                 T_drive=100.0, 
                 dt=0.5, 
                 n_tls=2,
                 n_freqs=300,
                 is_qutip_solver=None):
        
        self.J = J # interaction strength
        self.Omega_amp = Omega_amp # drive strength

        # bath parameters
        self.lam = lam # coupling strength
        self.gamma_bath = gamma_bath
        self.T = T # temperature

        # time parameters
        self.T_total = T_total # ns
        self.T_drive = T_drive   # ns
        self.dt = dt # ns

        self.n_tls = n_tls # number of TLS in the system
        self.is_qutip_solver = is_qutip_solver

        # time list and drive frequencies
        self.tlist = np.arange(0, self.T_total+self.dt, self.dt)
        self.n_time = len(self.tlist)
        self.n_freqs = n_freqs
        self.omega_d_vals = np.linspace(3.0, 5.0, self.n_freqs)

        # tls frequencies
        if tls_freqs is not None:
            self.omega_tls = tls_freqs
        else:
            self.omega_tls = np.random.uniform(3.0, 5.0, self.n_tls) # GHz

    def __getstate__(self):
        d = {
            "tls_freqs":self.omega_tls, 
            "J":self.J, 
            "Omega_amp":self.Omega_amp, 
            "lam":self.lam, 
            "gamma_bath":self.gamma_bath, 
            "T":self.T, 
            "T_total":self.T_total, 
            "T_drive":self.T_drive, 
            "dt":self.dt, 
            "n_tls":self.n_tls,
            "n_freqs":self.n_freqs
        }
        return d

    def __setstate__(self, d):
        self.__init__(tls_freqs=d["tls_freqs"], 
                    J=d["J"], 
                    Omega_amp=d["Omega_amp"], 
                    lam=d["lam"], 
                    gamma_bath=d["gamma_bath"], 
                    T=d["T"], 
                    T_total=d["T_total"], 
                    T_drive=d["T_drive"], 
                    dt=d["dt"], 
                    n_tls=d["n_tls"],
                    n_freqs=d["n_freqs"])

    def __str__(self):
        return(f"J_{self.J}_Omega_amp_{self.Omega_amp}_" + 
            f"lam_{self.lam}_gamma_bath_{self.gamma_bath}_" + 
            f"T_{self.T}_" + f"T_total_{self.T_total}_dt_{self.dt}_N_TLS_{self.n_tls}") 
    
    def _tensor(self, mats: list):
        res = mats[0]
        if type(res) == Qobj:
            return tensor(mats)
        
        for i in range(1, len(mats)):
            res = np.kron(res, mats[i])
        return res

    def build_operators(self):
        self.sx = []
        self.sz = []
        self.sm = []
        self.sp = []

        for i in range(self.n_tls):
            op_list = [qeye(2) if self.is_qutip_solver else np.eye(2) for _ in range(self.n_tls)]
            op_list[i] = sigmax() if self.is_qutip_solver else oqupy.operators.sigma("x")
            self.sx.append(self._tensor(op_list))
            
            op_list[i] = sigmaz() if self.is_qutip_solver else oqupy.operators.sigma("z")
            self.sz.append(self._tensor(op_list))
            
            op_list[i] = sigmam() if self.is_qutip_solver else oqupy.operators.sigma("-")
            self.sm.append(self._tensor(op_list))
            
            op_list[i] = sigmap() if self.is_qutip_solver else oqupy.operators.sigma("+")
            self.sp.append(self._tensor(op_list))

        self.collective_sp = sum(self.sp)
        self.collective_sm = sum(self.sm)
        if self.is_qutip_solver:
            self.collective_exc = self.collective_sp * self.collective_sm
        else:
            self.collective_exc = np.matmul(self.collective_sp, self.collective_sm)

        if self._name == "Markovian":
            # collapse ops with temperature dependence
            n_th = []
            for i in range(self.n_tls):
                n_th.append(1 / (np.exp(self.omega_tls[i] / self.T) - 1))
            self.c_ops = []
            for i in range(self.n_tls):
                self.c_ops.append(np.sqrt(self.lam * (n_th[i] + 1)) * self.sm[i])
                self.c_ops.append(np.sqrt(self.lam * n_th[i]) * self.sp[i])

    def get_hamiltonian(self):
        self.H = sum(0.5 * self.omega_tls[i] * self.sz[i] for i in range(self.n_tls))
        for i in range(self.n_tls):
            for j in range(i+1, self.n_tls):
                if isinstance(self.sx[0], Qobj):
                    self.H += self.J * self.sz[i] * self.sz[j]
                else:
                    self.H += self.J * np.matmul(self.sz[i], self.sz[j])

    
    def drive_coeff(self, t, args):
        if 0.0 <= t <= self.T_drive:
            return 0.5 * self.Omega_amp * np.cos(args["omega"] * t)
        else:
            return 0.0

class HEOM(Solver):
    def __init__(self, 
                 tls_freqs=None, 
                 J=0.02, 
                 Omega_amp=0.1, 
                 lam=0.02, 
                 gamma_bath=0.05, 
                 T=0.5, 
                 Nk=3, 
                 max_depth=5, 
                 T_total=1600, 
                 T_drive=100.0, 
                 dt=0.5, 
                 n_tls=2,
                 n_freqs=300):
        
        super().__init__(tls_freqs=tls_freqs, 
                        J=J, 
                        Omega_amp=Omega_amp, 
                        lam=lam, 
                        gamma_bath=gamma_bath, 
                        T=T, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt, 
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        is_qutip_solver=True)
        # label
        self._name = "HEOM"

        # number of expansion terms
        self.Nk = Nk

        # maximum memory depth
        self.max_depth = max_depth

        # build operators
        self.build_operators()

        # build static hamiltonian
        self.get_hamiltonian()

        # bath
        self.bath = DrudeLorentzBath(sum(self.sx), lam=self.lam, gamma=self.gamma_bath, T=self.T, Nk=self.Nk)
        
        # initial state
        self.evals, self.evecs = self.H.eigenstates()
        self.psi0 = self.evecs[0] 
        self.rho0 = ket2dm(self.psi0)

    def __getstate__(self):
        d = super().__getstate__()
        d["Nk"] = self.Nk
        d["max_depth"] = self.max_depth
        return d

    def __setstate__(self, d):
        return self.__init__(tls_freqs=d["tls_freqs"], 
                            J=d["J"], 
                            Omega_amp=d["Omega_amp"], 
                            lam=d["lam"], 
                            gamma_bath=d["gamma_bath"], 
                            T=d["T"], 
                            T_total=d["T_total"], 
                            T_drive=d["T_drive"], 
                            dt=d["dt"],  
                            n_tls=d["n_tls"],
                            n_freqs=d["n_freqs"],
                            Nk=d["Nk"],
                            max_depth=d["max_depth"])
    
    def __str__(self):
        return self._name + "_" + super().__str__() + f"_Nk{self.Nk}_max_depth_{self.max_depth}"
    
    def _worker(self, omega_d):
        H_full = QobjEvo(
        [self.H, [sum(self.sx), self.drive_coeff]],
        args = {"omega": omega_d}
        )

        solver = HEOMSolver(
            H_full,
            [self.bath],
            max_depth=self.max_depth,
            options={"nsteps": 5000, "progress_bar": ''},
        )

        result = solver.run(
            self.rho0,
            self.tlist,
            e_ops=[self.collective_exc, self.collective_sp]
        )

        return np.real(result.expect[0]), result.expect[1]
    
    def run(self):
        exc_heom = np.zeros((len(self.omega_d_vals), self.n_time))
        sp_heom = np.zeros((len(self.omega_d_vals), self.n_time), dtype=complex)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

            for idx, (exc, sp) in enumerate(tqdm(executor.map(self._worker, self.omega_d_vals),
                                                total=len(self.omega_d_vals),
                                                desc="HEOM simulations")):
                exc_heom[idx, :] = exc
                sp_heom[idx, :] = sp
            
            return (exc_heom, sp_heom)

class TEMPO(Solver):
    def __init__(self, 
                 tls_freqs=None, 
                 J=0.02, 
                 Omega_amp=0.1, 
                 lam=0.02, 
                 gamma_bath=0.05, 
                 T=0.5, 
                 T_total=1600, 
                 T_drive=100.0, 
                 dt=0.5, 
                 n_tls=2,
                 n_freqs=300,
                 zeta=1,
                 cutoff_type="exponential",
                 tcut=5.0,
                 epsrel=1e-4):
        
        super().__init__(tls_freqs=tls_freqs, 
                        J=J, 
                        Omega_amp=Omega_amp, 
                        lam=lam, 
                        gamma_bath=gamma_bath, 
                        T=T, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt,
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        is_qutip_solver=False)
        # label
        self._name = "TEMPO"

        # power las SD
        self.ohmicity = zeta

        # cutoff type
        self.cutoff_type = cutoff_type

        # time cutoff
        self.tcut = tcut

        # epsilon for computation
        self.epsrel = epsrel

        # build operators
        self.build_operators()

        # build static hamiltonian
        self.get_hamiltonian()

        # define bath
        #TODO: changing bath type. will need to adjust pickling procedure
        correlations = oqupy.PowerLawSD(alpha=lam,
                                    zeta=self.ohmicity,
                                    cutoff=gamma_bath,
                                    cutoff_type=self.cutoff_type,
                                    temperature=T)
        self.bath = oqupy.Bath(sum(self.sx), correlations)

        self.tempo_params = oqupy.TempoParameters(dt=self.dt, tcut=self.tcut, epsrel=self.epsrel)

        # initial state
        self.psi0 = np.array([[0] for _ in range(2*self.n_tls)])
        self.psi0[-1] = [1] # ground state
        self.rho0 = np.matmul(self.psi0, np.transpose(self.psi0))

    def __getstate__(self):
        d = super().__getstate__()
        d["ohmicity"] = self.ohmicity
        d["cutoff_type"] = self.cutoff_type
        d["tcut"] = self.tcut
        d["epsrel"] = self.epsrel
        return d

    def __setstate__(self, d):
        return self.__init__(tls_freqs=d["tls_freqs"], 
                            J=d["J"], 
                            Omega_amp=d["Omega_amp"], 
                            lam=d["lam"], 
                            gamma_bath=d["gamma_bath"], 
                            T=d["T"], 
                            T_total=d["T_total"], 
                            T_drive=d["T_drive"], 
                            dt=d["dt"], 
                            n_tls=d["n_tls"],
                            n_freqs=d["n_freqs"],
                            zeta=d["ohmicity"],
                            cutoff_type=d["cutoff_type"],
                            tcut=d["tcut"],
                            epsrel=d["epsrel"])
    
    def __str__(self):
        return self._name + "_" + super().__str__() + f"_tcut{self.tcut}_zeta_{self.ohmicity}"
    
    def _worker(self, omega_d, process_tensor):
        # total hamiltonian
        args = {"omega": omega_d}
        def ham(t):
            return self.H + self.drive_coeff(t, args) * sum(self.sx)
        
        # build system
        system = oqupy.TimeDependentSystem(ham)

        dynamics = oqupy.compute_dynamics(process_tensor=process_tensor, 
                                        system=system,
                                        initial_state=self.rho0,
                                        start_time=0.0,
                                        progress_type="silent")

        t, exc_tempo = dynamics.expectations(self.collective_exc, real=True)
        t, sp_tempo  = dynamics.expectations(self.collective_sp, real=False)

        return exc_tempo, sp_tempo
    
    def run(self):
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc_tempo = np.zeros((len(self.omega_d_vals), self.n_time))
        sp_tempo  = np.zeros((len(self.omega_d_vals), self.n_time), dtype=complex)

        worker = partial(self._worker,
                         process_tensor=process_tensor)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

            for idx, (exc_res, sp_res) in enumerate(tqdm(executor.map(worker, self.omega_d_vals),
                                                    total=len(self.omega_d_vals),
                                                    desc="TEMPO Simulations")):
                exc_tempo[idx,:], sp_tempo[idx,:] = exc_res, sp_res

            return (exc_tempo, sp_tempo)
        
class Lindblad(Solver):
    def __init__(self, 
                 tls_freqs=None, 
                 J=0.02, 
                 Omega_amp=0.1, 
                 lam=0.02, 
                 gamma_bath=0.05, 
                 T=0.5, 
                 T_total=1600, 
                 T_drive=100.0, 
                 dt=0.5, 
                 n_tls=2,
                 n_freqs=300):
        
        super().__init__(tls_freqs=tls_freqs, 
                        J=J, 
                        Omega_amp=Omega_amp, 
                        lam=lam, 
                        gamma_bath=gamma_bath, 
                        T=T, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt, 
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        is_qutip_solver=True)
        # label
        self._name = "Markovian"

        # build operators
        self.build_operators()

        # build static hamiltonian
        self.get_hamiltonian()

        # initial state
        self.evals, self.evecs = self.H.eigenstates()
        self.psi0 = self.evecs[0] 
        self.rho0 = ket2dm(self.psi0)

    def __getstate__(self):
        return super().__getstate__()

    def __setstate__(self, d):
        return super().__setstate__(d)
    
    def __str__(self):
        return self._name + "_" + super().__str__()
    
    def _worker(self, omega_d):
        H_full = QobjEvo(
        [self.H, [sum(self.sx), self.drive_coeff]],
        args = {"omega": omega_d}
        )

        result = mesolve(
        H_full,
        self.psi0,
        self.tlist,
        self.c_ops,
        e_ops=[self.collective_exc, self.collective_sp],
        options={"nsteps": 5000, "progress_bar": ''},
    )

        return np.real(result.expect[0]), result.expect[1]
    
    def run(self):
        exc_mark = np.zeros((len(self.omega_d_vals), self.n_time))
        sp_mark = np.zeros((len(self.omega_d_vals), self.n_time), dtype=complex)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

            for idx, (exc, sp) in enumerate(tqdm(executor.map(self._worker, self.omega_d_vals),
                                                total=len(self.omega_d_vals),
                                                desc="Markovian simulations")):
                exc_mark[idx, :] = exc
                sp_mark[idx, :] = sp
            
            return (exc_mark, sp_mark)

# ------------- FFT -------------
def smooth_envelope(amplitude, fraction=0.02):
    N = len(amplitude)
    win_len = int(max(3, min(N-1, np.round(N * fraction))))
    if win_len % 2 == 0:
        win_len += 1
    kernel = np.ones(win_len) / win_len
    return np.convolve(amplitude, kernel, mode='same')

def compute_fft(sp_t, omega_d_vals, tlist, dt, n_time, fmax=0.1): 

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
    idx_max = np.searchsorted(fft_freqs, fmax) 

    fft_data = fft_data[:, :idx_max]
    fft_freqs = fft_freqs[:idx_max]

    return fft_freqs, fft_data

# ------------- Plots -------------
def find_max(mats):
    res = np.max(mats[0])
    for i in range(1, len(mats)):
        res = max(res, np.max(mats[i]))
    return res


def find_min(mats):
    res = np.min(mats[0])
    for i in range(1, len(mats)):
        res = min(res, np.min(mats[i]))
    return res

def plot_exc_map(res_exc, omega_d_vals, tlist, labels, save=True, filename="exc_map"):
    n_plots = len(res_exc)
    # check shape
    for i in range(n_plots):
        assert(len(omega_d_vals) == len(res_exc[i]))
        assert(len(tlist) == len(res_exc[i][0]))
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]}
    fig, ax = plt.subplots(1, n_plots + 1, figsize=(6*n_plots,6), gridspec_kw=gridspec)

    # normalize cmaps
    vmin = find_min(res_exc)
    vmax = find_max(res_exc)

    # plot
    images = []
    
    for i in range(n_plots):
        images.append(ax[i].imshow(np.transpose(res_exc[i]),
                     extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                     origin='lower', aspect='auto', cmap='inferno',
                     vmin=vmin,
                     vmax=vmax))
        ax[i].set_title(r"$ \langle S_+S_- \rangle $ " + labels[i])
        ax[i].set_xlabel("Drive Frequency (GHz)")
        ax[i].set_ylabel("Time (ns)")

    # colorbar
    cb1 = fig.colorbar(images[-1], cax=ax[n_plots])
    cb1.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if save:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/{filename}.png")

def plot_sp_map(res_sp, omega_d_vals, tlist, labels, save=True, filename="sp_map"):
    n_plots = len(res_sp)
    # check shape
    for i in range(n_plots):
        assert(len(omega_d_vals) == len(res_sp[i]))
        assert(len(tlist) == len(res_sp[i][0]))
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]}
    fig, ax = plt.subplots(1, n_plots + 1, figsize=(6*n_plots,6), gridspec_kw=gridspec)

    # normalize cmaps
    vmin = find_min(np.abs(res_sp))
    vmax = find_max(np.abs(res_sp))

    # plot
    images = []
    
    for i in range(n_plots):
        images.append(ax[i].imshow(np.transpose(np.abs(res_sp[i])),
                     extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                     origin='lower', aspect='auto', cmap='inferno',
                     vmin=vmin,
                     vmax=vmax))
        ax[i].set_title(r"$ | \langle S_+ \rangle | $ " + labels[i])
        ax[i].set_xlabel("Drive Frequency (GHz)")
        ax[i].set_ylabel("Time (ns)")

    # colorbar
    cb1 = fig.colorbar(images[-1], cax=ax[n_plots])
    cb1.set_label(r"$ | \langle \sigma^{+}\rangle | $ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if save:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/{filename}.png")

def plot_diff_map(res_exc, res_sp, omega_d_vals, tlist, labels, save=True, filename="diff_map"):

    assert(len(res_exc) == len(labels))
    assert(len(res_sp) == len(labels))
    
    # check shape
    for i in range(len(res_exc)):
        assert(len(omega_d_vals) == len(res_exc[i]))
        assert(len(tlist) == len(res_exc[i][0]))

    # compute diffs
    n_plots = 0
    exc_diffs = {}
    sp_diffs = {}
    for i in range(len(res_exc)):
        for j in range(i+1, len(res_exc)):
            exc_diffs[r"$ \langle S_+S_- \rangle $ Difference " + f"({labels[i]} - {labels[j]})"] = np.subtract(res_exc[i], res_exc[j])
            sp_diffs[r"$ | \langle S_+ \rangle | $ Difference " + f"({labels[i]} - {labels[j]})"] = np.subtract(np.abs(res_sp[i]), np.abs(res_sp[j]))
            n_plots += 1
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]}
    fig, ax = plt.subplots(2, n_plots + 1, figsize=(6*n_plots, 10), gridspec_kw=gridspec)

    # plot
    images = []
    for j, diffs in enumerate([exc_diffs, sp_diffs]):
        vmin = find_min(list(diffs.values()))
        vmax = find_max(list(diffs.values()))
        for i, key in enumerate(diffs.keys()):
            images.append(ax[j][i].imshow(np.transpose(diffs[key]),
                        extent=[omega_d_vals[0], omega_d_vals[-1], tlist[0], tlist[-1]],
                        origin='lower', aspect='auto', cmap='bwr',
                        vmin=vmin,
                        vmax=vmax))
            ax[j][i].set_title(key)
            ax[j][i].set_xlabel("Drive Frequency (GHz)")
            ax[j][i].set_ylabel("Time (ns)")

        # colorbar
        cb1 = fig.colorbar(images[-1], cax=ax[j][n_plots])
        if j == 0:
            cb1.set_label(r"$ \langle S_+S_- \rangle $ Difference", labelpad=14)
        else:
            cb1.set_label(r"$ | \langle S_+ \rangle | $ Difference", labelpad=14)
        plt.tight_layout()

    # save
    if save:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/{filename}.png")

def plot_fft_map(fft_freqs, fft_data, omega_d_vals, omega_tls, labels, save=True, filename="fft_map"):
    n_plots = len(fft_freqs)
    # check shape
    for i in range(n_plots):
        assert(len(omega_d_vals) == len(fft_data[i]))
        assert(len(fft_freqs[i]) == len(fft_freqs[0]))
    
    gridspec = {'width_ratios': [1] * n_plots + [0.1]}
    fig, ax = plt.subplots(1, n_plots + 1, figsize=(6*n_plots, 6), gridspec_kw=gridspec)

    # normalize cmaps
    vmin = find_min(fft_data)
    vmax = find_max(fft_data)

    # plot
    images = []
    
    for i in range(n_plots):
        images.append(ax[i].imshow(fft_data[i].T,
                      extent=[omega_d_vals[0], omega_d_vals[-1],
                              fft_freqs[i][0], fft_freqs[i][-1]],
                      origin='lower', aspect='auto', cmap='Oranges',
                      vmin=vmin,
                      vmax=vmax))
        ax[i].set_title(f"FFT Data {labels[i]}")
        ax[i].set_xlabel("Drive Frequency (GHz)")
        ax[i].set_ylabel("FFT Frequency (GHz)")
        # add bare eigenfrequencies as vertical lines
        ax[i].vlines(x=omega_tls, color='black', ymin=fft_freqs[i][0], ymax=fft_freqs[i][-1], linestyle='--', linewidth=0.9)

    # colorbar
    cb1 = fig.colorbar(images[-1], cax=ax[n_plots])
    cb1.set_label(r"$|\mathrm{FFT}(\phi)|$ (arb.)", labelpad=14)
    plt.tight_layout()

    # save
    if save:
        os.makedirs("bctds_figures",exist_ok=True)
        plt.savefig(f"bctds_figures/{filename}.png")

