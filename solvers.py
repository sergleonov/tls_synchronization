import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import oqupy
import qutip as qt
from qutip.solver.heom import HEOMSolver
from qutip.core.environment import DrudeLorentzEnvironment, OhmicEnvironment
from functools import partial
from scipy.signal import hilbert

SOLVERS = ["Markovian", "Tiered", "HEOM", "TEMPO"]
SD_TYPES = ["power", "drude"]
HUSIMI_EVAL_METHODS = ["avg", "ptrace"]

class Solver:
    def __init__(self,
                 tls_freqs=None, 
                 J=0.02, 
                 Omega_amp=0.1, 
                 lam=0.02, 
                 T=0.5, 
                 T_total=1600, 
                 T_drive=100.0, 
                 dt=0.5, 
                 n_tls=2,
                 n_freqs=300,
                 is_qutip_solver=None,
                 name="Abstract"):

        assert name in SOLVERS, "Error: Invalid solver name"
        self._name = name
        
        self.J = J # interaction strength
        self.Omega_amp = Omega_amp # drive strength

        # bath parameters
        self.lam = lam # coupling strength
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
            "T":self.T, 
            "T_total":self.T_total, 
            "T_drive":self.T_drive, 
            "dt":self.dt, 
            "n_tls":self.n_tls,
            "n_freqs":self.n_freqs,
            "name":self._name
        }
        return d

    def __setstate__(self, d):
        self.__init__(tls_freqs=d["tls_freqs"], 
                    J=d["J"], 
                    Omega_amp=d["Omega_amp"], 
                    lam=d["lam"], 
                    T=d["T"], 
                    T_total=d["T_total"], 
                    T_drive=d["T_drive"], 
                    dt=d["dt"], 
                    n_tls=d["n_tls"],
                    n_freqs=d["n_freqs"],
                    name=d["name"])

    def __str__(self):
        return(f"{self._name}_J_{self.J}_Omega_amp_{self.Omega_amp}_" + 
            f"lam_{self.lam}_T_{self.T}_" + 
            f"T_total_{self.T_total}_dt_{self.dt}_N_TLS_{self.n_tls}") 
    
    def _tensor(self, mats: list):
        res = mats[0]
        if type(res) == qt.Qobj:
            return qt.tensor(mats)
        
        for i in range(1, len(mats)):
            res = np.kron(res, mats[i])
        return res
    
    def _build_c_ops(self):
        # collapse ops with temperature dependence
        n_th = []
        for i in range(self.n_tls):
            n_th.append(1 / (np.exp(self.omega_tls[i] / self.T) - 1))
        self.c_ops = []
        for i in range(self.n_tls):
            self.c_ops.append(np.sqrt(self.lam * (n_th[i] + 1)) * sum(self.sm))
            self.c_ops.append(np.sqrt(self.lam * n_th[i]) * sum(self.sp))
        if self._name == "Tiered":
            self.a = self._tensor([qt.qeye(2), qt.qeye(2), qt.destroy(self.Nb)])
            n_th_mode = 1 / (np.exp(self.omega_c / self.T) - 1)
            self.c_ops.append(np.sqrt(self.lam * (n_th_mode + 1)) * (self.a))
            self.c_ops.append(np.sqrt(self.lam * (n_th_mode)) * (self.a.dag()))

    def build_operators(self):
        assert self._name in SOLVERS, "Error: Invalid solver name"

        sx_tls = []
        sy_tls = []
        sz_tls = []
        sm_tls = []
        sp_tls = []

        for i in range(self.n_tls):
            op_list = [qt.qeye(2) if self.is_qutip_solver else np.eye(2) for _ in range(self.n_tls)]
            op_list[i] = qt.sigmax() if self.is_qutip_solver else oqupy.operators.sigma("x")
            sx_tls.append(self._tensor(op_list))

            op_list[i] = qt.sigmay() if self.is_qutip_solver else oqupy.operators.sigma("y")
            sy_tls.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmaz() if self.is_qutip_solver else oqupy.operators.sigma("z")
            sz_tls.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmam() if self.is_qutip_solver else oqupy.operators.sigma("-")
            sm_tls.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmap() if self.is_qutip_solver else oqupy.operators.sigma("+")
            sp_tls.append(self._tensor(op_list))

        match self._name:
            case "Markovian":
                self.sx = sx_tls
                self.sy = sy_tls
                self.sz = sz_tls
                self.sp = sp_tls
                self.sm = sm_tls
                # collapse operators
                self._build_c_ops()

            case "Tiered":
                self.sx = []
                self.sy = []
                self.sz = []
                self.sp = []
                self.sm = []
                I_cav = qt.qeye(self.Nb) 
                for i in range(self.n_tls):
                    op_list = [sx_tls[i], I_cav]
                    self.sx.append(self._tensor(op_list))

                    op_list = [sy_tls[i], I_cav]
                    self.sy.append(self._tensor(op_list))

                    op_list = [sz_tls[i], I_cav]
                    self.sz.append(self._tensor(op_list))

                    op_list = [sp_tls[i], I_cav]
                    self.sp.append(self._tensor(op_list))

                    op_list = [sm_tls[i], I_cav]
                    self.sm.append(self._tensor(op_list))
                # annihilator and collapse operators
                self._build_c_ops()
        
            case _:
                self.sx = sx_tls
                self.sy = sy_tls
                self.sz = sz_tls
                self.sp = sp_tls
                self.sm = sm_tls
        
        # observables
        self.collective_sp = sum(self.sp)
        self.collective_sm = sum(self.sm)
        if self.is_qutip_solver:
            self.collective_exc = self.collective_sp * self.collective_sm
        else:
            self.collective_exc = np.matmul(self.collective_sp, self.collective_sm)

    def build_hamiltonian(self):
        assert self._name in SOLVERS, "Error: Invalid solver name"

        self.H = sum(0.5 * self.omega_tls[i] * self.sz[i] for i in range(self.n_tls))
        for i in range(self.n_tls):
            for j in range(i+1, self.n_tls):
                if self.is_qutip_solver:
                    self.H += self.J * self.sz[i] * self.sz[j]
                else:
                    self.H += self.J * np.matmul(self.sz[i], self.sz[j])

        if self._name == "Tiered":
            self.H += self.omega_c * self.a.dag() * self.a # cavity hamiltonian
            self.H += self.g * sum(self.sx) * (self.a.dag() + self.a) # system-bath hamiltonian
    
    def drive_coeff(self, t, args):
        if 0.0 <= t <= self.T_drive:
            return 0.5 * self.Omega_amp * np.cos(args["omega"] * t)
        else:
            return 0.0
        
    def eval_husimi(self, rho, theta, phi, tls_idx=None, method="avg"):
        assert method in HUSIMI_EVAL_METHODS, "Error: Invalid husimi evaluation method"

        if self.is_qutip_solver:
            if rho.isket:
                rho = qt.ket2dm(rho)
            
            j = 1/2 # spin of TLS
            prefactor = (2 * j + 1) / (4 * np.pi) # husimi prefactor
            match method:
                case "ptrace":
                    if tls_idx is None:
                        raise ValueError("Error: Index for the partial trace is None")
                    rho_partial = qt.ptrace(rho, tls_idx)
                    Q, theta_list, phi_list = qt.spin_q_function(rho_partial, theta, phi)
                    return prefactor * np.transpose(Q)
                case "avg":
                    Qs = []
                    for i in range(self.n_tls):
                        rho_partial = qt.ptrace(rho, i)
                        Q, theta_list, phi_list = qt.spin_q_function(rho_partial, theta, phi)
                        Qs.append(Q)
                    Q_res = np.mean(Qs, axis=0)
                    return prefactor * np.transpose(Q_res)
        else:
            # TODO: implement TEMPO Husimi computation
            raise NotImplementedError("Unsupported computation for TEMPO")
        
    def _pearson_evolution(self, x, y, window_size, overlap=1):
        step = window_size - overlap
        assert step > 0
        n_steps = self.n_time // step + 1
        C_t = np.zeros(self.n_time)

        C = None
        for i in range(n_steps):
            start = step * i + 1
            end = start + window_size
            if end > self.n_time:
                C = np.corrcoef(x[start::], y[start::])[1, 0]
                C_t[start::] = C
                break
            C = np.corrcoef(x[start:end], y[start:end])[1, 0]
            C_t[start:end] = C
        return C_t

# TODO: implement a function to compute pearson correlation in an abstract way to avoid copy-paste code in subclasses
        
    # helper for computing phase lock value
    def _plv(self, analytic_x, analytic_y):
        phase_x, phase_y = np.angle(analytic_x), np.angle(analytic_y)
        phase_diff = phase_x - phase_y
        return np.abs(np.mean(np.exp(1j * phase_diff)))

    def _plv_evolution(self, x, y, window_size, overlap=1):
        step = window_size - overlap
        assert step > 0
        n_steps = self.n_time // step + 1
        plv_t = np.zeros(self.n_time)

        # analytic signal
        x_a, y_a = hilbert(np.real(x)), hilbert(np.real(y))

        for i in range(n_steps):
            start = step * i + 1
            end = start + window_size
            if end > self.n_time:
                plv_t[start::] = self._plv(x_a[start::], y_a[start::])
                break
            plv_t[start:end] = self._plv(x_a[start:end], y_a[start:end])
        return plv_t
    
    def _cor_sim_helper(self, states, corr_evo, window_size, overlap):
        plvs = {}
        exp_xs = [] # expectations in x

        for i in range(self.n_tls):
            if self.is_qutip_solver:
                exp_x = qt.expect(self.sx[i], states)
            else:
                t, exp_x = states.expectations(self.sx[i], real=True)

            exp_xs.append(exp_x)
        
        for i in range(0, len(exp_xs)):
            for j in range(i+1, len(exp_xs)):
                plvs[f"TLS {self.omega_tls[i]}, {self.omega_tls[j]}"] = corr_evo(exp_xs[i], exp_xs[j], window_size=window_size, overlap=overlap)

        return plvs, self.tlist
    
# --------------------- HEOM --------------------

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
                 sd_type="drude",
                 ohmicity=None):
        
        assert sd_type in SD_TYPES, "Error: Invalid spectral density"

        self.gamma_bath = gamma_bath
        
        super().__init__(tls_freqs=tls_freqs, 
                        J=J, 
                        Omega_amp=Omega_amp, 
                        lam=lam, 
                        T=T, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt, 
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        is_qutip_solver=True,
                        name="HEOM")

        # number of expansion terms
        self.Nk = Nk

        # maximum memory depth
        self.max_depth = max_depth

        # build operators
        self.build_operators()

        # build static hamiltonian
        self.build_hamiltonian()

       # bath params for reconstructions
        self.sd_type = sd_type
        self.ohmicity = ohmicity

        # initial state
        self.evals, self.evecs = self.H.eigenstates()
        self.psi0 = self.evecs[0] 
        self.rho0 = qt.ket2dm(self.psi0)

    def __getstate__(self):
        d = super().__getstate__()
        d["gamma_bath"]=self.gamma_bath, 
        d["Nk"] = self.Nk
        d["max_depth"] = self.max_depth
        d["sd_type"] = self.sd_type
        d["ohmicity"] = self.ohmicity
        assert self.sd_type in SD_TYPES, "Error: Invalid spectral density"
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
        assert self.sd_type in SD_TYPES, "Error: Invalid spectral density"
        sd = self.sd_type
        if sd == "power": sd += f"_{self.ohmicity}"
        return super().__str__() + f"_gamma_bath_{self.gamma_bath}_Nk{self.Nk}_max_depth_{self.max_depth}_{sd}"
    
    def _worker(self, omega_d, store_states=False):
        H_full = qt.QobjEvo(
        [self.H, [sum(self.sx), self.drive_coeff]],
        args = {"omega": omega_d}
        )

        global _heom_bath
        solver = HEOMSolver(
            H_full,
            (_heom_bath, sum(self.sx)),
            max_depth=self.max_depth,
            options={"nsteps": 5000, "progress_bar": '', "store_states": store_states},
        )

        result = solver.run(
            self.rho0,
            self.tlist,
            e_ops=[self.collective_exc, self.collective_sp]
        )

        if store_states:
            return np.real(result.expect[0]), result.expect[1], result.states
        return np.real(result.expect[0]), result.expect[1]
    
    def _build_bath(self):
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
    
    def run(self):
        assert self.sd_type in SD_TYPES, "Error: Invalid spectral density"
        # bath
        self._build_bath()

        exc_heom = np.zeros((len(self.omega_d_vals), self.n_time))
        sp_heom = np.zeros((len(self.omega_d_vals), self.n_time), dtype=complex)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

            for idx, (exc, sp) in enumerate(tqdm(executor.map(self._worker, self.omega_d_vals),
                                                total=len(self.omega_d_vals),
                                                desc="HEOM simulations")):
                exc_heom[idx, :] = exc
                sp_heom[idx, :] = sp
            
            return (exc_heom, sp_heom)
    
    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
        self._build_bath()

        exc, sp, states = self._worker(omega_d, store_states=True)
        Qt = np.zeros((self.n_time, len(theta), len(phi)))

        eval_husimi_partial = partial(self.eval_husimi,
                                      theta=theta,
                                      phi=phi,
                                      method=method,
                                      tls_idx=tls_idx)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
        
            for t_idx, Q in enumerate(tqdm(executor.map(eval_husimi_partial, states), 
                                            total=len(states), 
                                            desc="HEOM Husimi-Q Computation")):
                Qt[t_idx] = Q
        
        return Qt

    def phase_sim(self, omega_d):
        self._build_bath()

        exc, sp, states = self._worker(omega_d, store_states=True)
        
        phases = []

        for i in range(self.n_tls):
            e_ops = [self.sx[i], self.sy[i]]
            exp_x, exp_y = qt.expect(e_ops, states)

            phases.append(np.arctan2(exp_y, exp_x))
        
        return phases, self.tlist
        
    def pearson_sim(self, omega_d, window_size, overlap):
        self._build_bath()

        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, self._pearson_evolution, window_size, overlap)
    
    def plv_sim(self, omega_d, window_size, overlap):
        self._build_bath()

        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, self._plv_evolution, window_size, overlap)


# --------------------- TEMPO --------------------

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
        
        self.gamma_bath = gamma_bath
        
        super().__init__(tls_freqs=tls_freqs, 
                        J=J, 
                        Omega_amp=Omega_amp, 
                        lam=lam, 
                        T=T, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt,
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        is_qutip_solver=False,
                        name="TEMPO")

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
        self.build_hamiltonian()

        # define bath
        #TODO: changing bath type. will need to adjust pickling procedure
        correlations = oqupy.PowerLawSD(alpha=self.lam,
                                    zeta=self.ohmicity,
                                    cutoff=self.gamma_bath,
                                    cutoff_type=self.cutoff_type,
                                    temperature=self.T)
        self.bath = oqupy.Bath(sum(self.sx), correlations)

        self.tempo_params = oqupy.TempoParameters(dt=self.dt, tcut=self.tcut, epsrel=self.epsrel)

        # initial state
        self.psi0 = np.array([[0] for _ in range(2*self.n_tls)])
        self.psi0[-1] = [1] # ground state
        self.rho0 = np.matmul(self.psi0, np.transpose(self.psi0))

    def __getstate__(self):
        d = super().__getstate__()
        d["gamma_bath"] = self.gamma_bath
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
        return super().__str__() + f"_gamma_bath_{self.gamma_bath}_tcut{self.tcut}_zeta_{self.ohmicity}"
    
    def _worker(self, omega_d, process_tensor, store_states=False):
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

        if store_states:
            return exc_tempo, sp_tempo, dynamics
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
    
    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc, sp, dynamics = self._worker(omega_d, process_tensor, store_states=True)
        Qt = np.zeros((self.n_time, len(theta), len(phi)))

        eval_husimi_partial = partial(self.eval_husimi,
                                      theta=theta,
                                      phi=phi,
                                      method=method,
                                      tls_idx=tls_idx)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
        
            for t_idx, Q in enumerate(tqdm(executor.map(eval_husimi_partial, dynamics.states), 
                                            total=len(dynamics.states), 
                                            desc="TEMPO Husimi-Q Computation")):
                Qt[t_idx] = Q
        
        return Qt
    
    def phase_sim(self, omega_d):
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc, sp, dynamics = self._worker(omega_d, process_tensor, store_states=True)
        
        phases = []

        for i in range(self.n_tls):
            t, exp_x = dynamics.expectations(self.sx[i], real=True)
            t, exp_y = dynamics.expectations(self.sy[i], real=True)

            phases.append(np.arctan2(exp_y, exp_x))
        
        return phases, self.tlist
    
    def pearson_sim(self, omega_d, window_size, overlap):
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc, sp, dynamics = self._worker(omega_d, process_tensor, store_states=True)

        return self._cor_sim_helper(dynamics, self._pearson_evolution, window_size, overlap)
    
    def plv_sim(self, omega_d, window_size, overlap):
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc, sp, dynamics = self._worker(omega_d, process_tensor, store_states=True)

        return self._cor_sim_helper(dynamics, self._plv_evolution, window_size, overlap)
    
# --------------------- Markovian --------------------

class Lindblad(Solver):
    def __init__(self, 
                 tls_freqs=None, 
                 J=0.02, 
                 Omega_amp=0.1, 
                 lam=0.02, 
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
                        T=T, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt, 
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        is_qutip_solver=True,
                        name="Markovian")

        # build operators
        self.build_operators()

        # build static hamiltonian
        self.build_hamiltonian()

        # initial state
        self.evals, self.evecs = self.H.eigenstates()
        self.psi0 = self.evecs[0] 
        self.rho0 = qt.ket2dm(self.psi0)

    def __getstate__(self):
        return super().__getstate__()

    def __setstate__(self, d):
        return self.__init__(tls_freqs=d["tls_freqs"], 
                            J=d["J"], 
                            Omega_amp=d["Omega_amp"], 
                            lam=d["lam"], 
                            T=d["T"], 
                            T_total=d["T_total"], 
                            T_drive=d["T_drive"], 
                            dt=d["dt"], 
                            n_tls=d["n_tls"],
                            n_freqs=d["n_freqs"])
    
    def __str__(self):
        return super().__str__()
    
    def _worker(self, omega_d, store_states=False):
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
        options={"nsteps": 5000, "progress_bar": '', "store_states": store_states},
        )
        
        if store_states:
            return np.real(result.expect[0]), result.expect[1], result.states
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
        
    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
        exc, sp, states = self._worker(omega_d, store_states=True)
        Qt = np.zeros((self.n_time, len(theta), len(phi)))

        eval_husimi_partial = partial(self.eval_husimi,
                                      theta=theta,
                                      phi=phi,
                                      method=method,
                                      tls_idx=tls_idx)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
        
            for t_idx, Q in enumerate(tqdm(executor.map(eval_husimi_partial, states), 
                                            total=len(states), 
                                            desc="Lindblad Husimi-Q Computation")):
                Qt[t_idx] = Q
        
        return Qt
    
    def phase_sim(self, omega_d):
        exc, sp, states = self._worker(omega_d, store_states=True)
        
        phases = []

        for i in range(self.n_tls):
            e_ops = [self.sx[i], self.sy[i]]
            exp_x, exp_y = qt.expect(e_ops, states)

            phases.append(np.arctan2(exp_y, exp_x))
        
        return phases, self.tlist
    
    def correlation_sim(self, omega_d, window_size, overlap):
        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, self._pearson_evolution, window_size, overlap)
    
    def plv_sim(self, omega_d, window_size, overlap):
        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, self._plv_evolution, window_size, overlap)

# --------------------- Tiered --------------------
    
class TieredSolver(Solver):
    def __init__(self, 
                 tls_freqs=None, 
                 J=0.02, 
                 Omega_amp=0.1, 
                 lam=0.002, 
                 g=0.02,
                 T=0.5, 
                 T_total=1600, 
                 T_drive=100.0, 
                 dt=0.5, 
                 n_tls=2,
                 n_freqs=300,
                 omega_c=3.75,
                 Nb=10):
        
        self.g = g
        self.omega_c = omega_c
        self.Nb = Nb
        
        super().__init__(tls_freqs=tls_freqs, 
                        J=J, 
                        Omega_amp=Omega_amp, 
                        lam=lam, 
                        T=T, 
                        T_total=T_total, 
                        T_drive=T_drive, 
                        dt=dt, 
                        n_tls=n_tls,
                        n_freqs=n_freqs,
                        is_qutip_solver=True,
                        name="Tiered")
        
         # build operators
        self.build_operators()

        # build static hamiltonian
        self.build_hamiltonian()

        # initial state
        self.evals, self.evecs = self.H.eigenstates()
        self.psi0 = self.evecs[0] 
        self.rho0 = qt.ket2dm(self.psi0)
        
    def __getstate__(self):
        d = super().__getstate__()
        d["omega_c"] = self.omega_c
        d["Nb"] = self.Nb
        d["g"] = self.g
        return d

    def __setstate__(self, d):
        return self.__init__(tls_freqs=d["tls_freqs"], 
                            J=d["J"], 
                            Omega_amp=d["Omega_amp"], 
                            lam=d["lam"], 
                            g=d["g"],
                            T=d["T"], 
                            T_total=d["T_total"], 
                            T_drive=d["T_drive"], 
                            dt=d["dt"], 
                            n_tls=d["n_tls"],
                            n_freqs=d["n_freqs"],
                            omega_c=d["omega_c"],
                            Nb=d["Nb"])
    
    def __str__(self):
        return super().__str__() + f"_mode_{self.omega_c}_Nb{self.Nb}_g_{self.g}"
    
    def _worker(self, omega_d, store_states=False):
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
        options={"nsteps": 5000, "progress_bar": '', "store_states": store_states},
        )

        if store_states:
            return np.real(result.expect[0]), result.expect[1], result.states
        return np.real(result.expect[0]), result.expect[1]
    
    def run(self):
        exc_mark = np.zeros((len(self.omega_d_vals), self.n_time))
        sp_mark = np.zeros((len(self.omega_d_vals), self.n_time), dtype=complex)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:

            for idx, (exc, sp) in enumerate(tqdm(executor.map(self._worker, self.omega_d_vals),
                                                total=len(self.omega_d_vals),
                                                desc="Tiered System simulations")):
                exc_mark[idx, :] = exc
                sp_mark[idx, :] = sp
            
            return (exc_mark, sp_mark)
    
    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
        exc, sp, states = self._worker(omega_d, store_states=True)
        Qt = np.zeros((self.n_time, len(theta), len(phi)))

        eval_husimi_partial = partial(self.eval_husimi,
                                      theta=theta,
                                      phi=phi,
                                      method=method,
                                      tls_idx=tls_idx)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
        
            for t_idx, Q in enumerate(tqdm(executor.map(eval_husimi_partial, states), 
                                            total=len(states), 
                                            desc="Tiered Husimi-Q Computation")):
                Qt[t_idx] = Q
        
        return Qt
        
    def phase_sim(self, omega_d):
        exc, sp, states = self._worker(omega_d, store_states=True)
        
        phases = []

        for i in range(self.n_tls):
            e_ops = [self.sx[i], self.sy[i]]
            exp_x, exp_y = qt.expect(e_ops, states)

            phases.append(np.arctan2(exp_y, exp_x))
        
        return phases, self.tlist
    
    def correlation_sim(self, omega_d, window_size, overlap):
        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, self._pearson_evolution, window_size, overlap)
    
    def plv_sim(self, omega_d, window_size, overlap):
        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, self._plv_evolution, window_size, overlap)
