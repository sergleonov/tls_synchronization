from tls_sync.solver import Solver
import numpy as np
from .parallel import run_parallel, parallel_eval_husimi
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
        self.psi0 = np.zeros((2 ** self.n_tls, 1))
        self.psi0[-1] = 1  # ground state
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

        worker = partial(self._worker,
                         process_tensor=process_tensor,
                         store_states=store_states)

        return run_parallel(
            omega_d_vals=self.omega_d_vals,
            worker=worker,
            n_time=self.n_time,
            store_states=store_states,
            desc="TEMPO Simulations",
        )
    
    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
        """Compute Husimi-Q functions for a TEMPO run at a given drive frequency."""
        dynamics = self._get_states(omega_d)
        return parallel_eval_husimi(
            dynamics.states,
            self.eval_husimi,
            theta,
            phi,
            method,
            tls_idx,
            desc="TEMPO Husimi-Q Computation"
        )
    
    def _get_states(self, omega_d):
        process_tensor = oqupy.pt_tempo_compute(bath=self.bath,
                                            start_time=0.0,
                                            end_time=self.T_total,
                                            parameters=self.tempo_params)

        _, _, dynamics = self._worker(omega_d, process_tensor, store_states=True)
        return dynamics