from tls_sync.solver import Solver
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import oqupy
from functools import partial

class TEMPO(Solver):
    """TEMPO solver wrapper for non-Markovian process tensor simulations."""

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
        """Initialize a TEMPO solver instance.

        Parameters
        ----------
        tls_freqs : array_like or None
            TLS eigenfrequencies.
        J : float
            TLS coupling strength.
        Omega_amp : float
            Drive amplitude.
        lam : float
            System-bath coupling strength.
        gamma_bath : float
            Bath memory cutoff.
        T : float
            Bath temperature.
        T_total : float
            Total simulation duration.
        T_drive : float
            Drive duration.
        dt : float
            Time step size.
        n_tls : int
            Number of TLS components.
        n_freqs : int
            Number of drive frequencies.
        zeta : float
            Ohmicity exponent for the power-law spectral density.
        cutoff_type : str
            Bath cutoff function type.
        tcut : float
            Process tensor memory cutoff time.
        epsrel : float
            Relative accuracy required for TEMPO computations.
        """
        
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
        """Return the picklable state of the TEMPO solver."""
        d = super().__getstate__()
        d["gamma_bath"] = self.gamma_bath
        d["ohmicity"] = self.ohmicity
        d["cutoff_type"] = self.cutoff_type
        d["tcut"] = self.tcut
        d["epsrel"] = self.epsrel
        return d

    def __setstate__(self, d):
        """Reconstruct the TEMPO solver from a saved state."""
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
    
    def get_name(self):
        """Return a descriptive solver name including power spectral density."""
        return self._name + f" (Power SD {self.ohmicity})"
    
    def __str__(self):
        return super().__str__() + f"_gamma_bath_{self.gamma_bath}_tcut{self.tcut}_zeta_{self.ohmicity}"
    
    def _worker(self, omega_d, process_tensor, store_states=False):
        """Run a single TEMPO simulation for a specific drive frequency."""
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
    
    def run(self, store_states=False):
        """Execute TEMPO simulations across all configured drive frequencies."""
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc_tempo = np.zeros((len(self.omega_d_vals), self.n_time))
        sp_tempo  = np.zeros((len(self.omega_d_vals), self.n_time), dtype=complex)
        states_tempo = np.zeros((len(self.omega_d_vals), self.n_time), dtype=object) if store_states else None

        worker = partial(self._worker,
                         process_tensor=process_tensor,
                         store_states=store_states)

        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as executor:
            if store_states:
                for idx, (exc_res, sp_res, states) in enumerate(tqdm(executor.map(worker, self.omega_d_vals),
                                                        total=len(self.omega_d_vals),
                                                        desc="TEMPO Simulations")):
                    exc_tempo[idx,:], sp_tempo[idx,:] = exc_res, sp_res
                    states_tempo[idx, :] = states.states if hasattr(states, "states") else states
                return (exc_tempo, sp_tempo, states_tempo)
            else:
                for idx, (exc_res, sp_res) in enumerate(tqdm(executor.map(worker, self.omega_d_vals),
                                                        total=len(self.omega_d_vals),
                                                        desc="TEMPO Simulations")):
                    exc_tempo[idx,:], sp_tempo[idx,:] = exc_res, sp_res

                return (exc_tempo, sp_tempo)
    
    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
        """Compute Husimi-Q functions for a TEMPO run at a given drive frequency."""
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
        """Compute TLS phase trajectories for a TEMPO run."""
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc, sp, dynamics = self._worker(omega_d, process_tensor, store_states=True)
        
        return self._phase_sim_helper(dynamics)
    
    def pearson_sim(self, omega_d, window_size, overlap):
        """Compute rolling Pearson correlations from TEMPO temporal states."""
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc, sp, dynamics = self._worker(omega_d, process_tensor, store_states=True)

        return self._cor_sim_helper(dynamics, "pearson", window_size, overlap)
    
    def plv_sim(self, omega_d, window_size, overlap):
        """Compute rolling phase locking values from TEMPO temporal states."""
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        exc, sp, dynamics = self._worker(omega_d, process_tensor, store_states=True)

        return self._cor_sim_helper(dynamics, "plv", window_size, overlap)
    
    def phase_corr_sim(self, omega_d, corr_names, window_size=None, overlap=None):
        """Compute phase trajectories and requested correlations for TEMPO."""
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)
        
        exc, sp, dynamics = self._worker(omega_d, process_tensor, store_states=True)
        phases, t = self._phase_sim_helper(dynamics)
        if isinstance(corr_names, list):
            corrs = []
            for corr_name in corr_names:
                corr, t = self._cor_sim_helper(dynamics, corr_name, window_size, overlap)
                corrs.append(corr)
            return phases, corrs, t

        corr, t = self._cor_sim_helper(dynamics, corr_names, window_size, overlap)
        return phases, corr, t