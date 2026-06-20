import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import oqupy
import qutip as qt
from qutip.solver.heom import HEOMSolver
from qutip.core.environment import DrudeLorentzEnvironment, OhmicEnvironment
from functools import partial


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
        if type(res) == qt.Qobj:
            return qt.tensor(mats)
        
        for i in range(1, len(mats)):
            res = np.kron(res, mats[i])
        return res

    def build_operators(self):
        self.sx = []
        self.sz = []
        self.sm = []
        self.sp = []

        for i in range(self.n_tls):
            op_list = [qt.qeye(2) if self.is_qutip_solver else np.eye(2) for _ in range(self.n_tls)]
            op_list[i] = qt.sigmax() if self.is_qutip_solver else oqupy.operators.sigma("x")
            self.sx.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmaz() if self.is_qutip_solver else oqupy.operators.sigma("z")
            self.sz.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmam() if self.is_qutip_solver else oqupy.operators.sigma("-")
            self.sm.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmap() if self.is_qutip_solver else oqupy.operators.sigma("+")
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
                if isinstance(self.sx[0], qt.Qobj):
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
                 n_freqs=300,
                 sd_type="dl",
                 ohmicity=None):
        
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

       # bath params for reconstructions
        self.sd_type = sd_type
        self.ohmicity = ohmicity

        # initial state
        self.evals, self.evecs = self.H.eigenstates()
        self.psi0 = self.evecs[0] 
        self.rho0 = qt.ket2dm(self.psi0)

    def __getstate__(self):
        d = super().__getstate__()
        d["Nk"] = self.Nk
        d["max_depth"] = self.max_depth
        d["sd_type"] = self.sd_type
        d["ohmicity"] = self.ohmicity
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
                            max_depth=d["max_depth"],
                            sd_type=d["sd_type"],
                            ohmicity=d["ohmicity"])
    
    def __str__(self):
        sd = self.sd_type
        if sd == "power": sd += f"_{self.ohmicity}"
        return self._name + "_" + super().__str__() + f"_Nk{self.Nk}_max_depth_{self.max_depth}_{sd}"
    
    def _worker(self, omega_d):
        H_full = qt.QobjEvo(
        [self.H, [sum(self.sx), self.drive_coeff]],
        args = {"omega": omega_d}
        )

        global _heom_bath
        solver = HEOMSolver(
            H_full,
            (_heom_bath, sum(self.sx)),
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
        # bath
        global _heom_bath
        match self.sd_type:
            case "drude": # Drude-Lorentz
                env = DrudeLorentzEnvironment(T=self.T, lam=self.lam, gamma=self.gamma_bath, Nk=self.Nk)
                _heom_bath = env.approximate("matsubara", Nk=self.Nk)
            case "power": # Power Law
                env = OhmicEnvironment(T=self.T, alpha=self.lam, wc=self.gamma_bath, s=self.ohmicity) # alpha is coupling, wc is cutoff
                _heom_bath, info = env.approximate(method="cf", tlist=self.tlist, target_rmse=None, Nr_max=self.Nk, Ni_max=self.Nk, maxfev=1e8)
            case _: 
                raise ValueError("Invalid spectral density type.")

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
        self.rho0 = qt.ket2dm(self.psi0)

    def __getstate__(self):
        return super().__getstate__()

    def __setstate__(self, d):
        return super().__setstate__(d)
    
    def __str__(self):
        return self._name + "_" + super().__str__()
    
    def _worker(self, omega_d):
        H_full = qt.QobjEvo(
        [self.H, [sum(self.sx), self.drive_coeff]],
        args = {"omega": omega_d}
        )

        result = qt.mesolve(
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
