from tls_sync import Solver, SD_TYPES
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import qutip as qt
from qutip.solver.heom import HEOMSolver
from qutip.core.environment import DrudeLorentzEnvironment, OhmicEnvironment
from functools import partial

class HEOM(Solver):
    """HEOM solver wrapper for hierarchical equations of motion simulations."""

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
        """Initialize a HEOM solver instance.

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
            Bath cutoff or damping rate.
        T : float
            Bath temperature.
        Nk : int
            Number of Matsubara or fitting terms.
        max_depth : int
            Hierarchy truncation depth.
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
        sd_type : {'drude', 'power'}
            Spectral density type.
        ohmicity : float or None
            Power-law exponent for ``power`` spectral density.
        """
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
        """Return the picklable state of the HEOM solver."""
        d = super().__getstate__()
        d["gamma_bath"]=self.gamma_bath, 
        d["Nk"] = self.Nk
        d["max_depth"] = self.max_depth
        d["sd_type"] = self.sd_type
        d["ohmicity"] = self.ohmicity
        assert self.sd_type in SD_TYPES, "Error: Invalid spectral density"
        return d

    def __setstate__(self, d):
        """Reconstruct the HEOM instance from a saved state."""
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
    
    def get_name(self):
        """Return a descriptive solver name including spectral density type."""
        match self.sd_type:
            case "power":
                return self._name + f" (Power SD {self.ohmicity})"
            case "drude":
                return self._name + f" (Drude-Lorentz)"
            case _:
                return self._name

    def __str__(self):
        assert self.sd_type in SD_TYPES, "Error: Invalid spectral density"
        sd = self.sd_type
        if sd == "power": sd += f"_{self.ohmicity}"
        return super().__str__() + f"_gamma_bath_{self.gamma_bath}_Nk{self.Nk}_max_depth_{self.max_depth}_{sd}"
    
    def _worker(self, omega_d, store_states=False):
        """Run a single HEOM simulation for a specific drive frequency."""
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
        """Build the HEOM bath object based on the chosen spectral density."""
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
        """Execute HEOM simulations across all configured drive frequencies."""
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
        """Compute Husimi-Q functions for an HEOM run at a given drive frequency."""
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
        """Compute TLS phase trajectories for an HEOM run."""
        self._build_bath()

        exc, sp, states = self._worker(omega_d, store_states=True)
        
        return self._phase_sim_helper(states)
        
    def pearson_sim(self, omega_d, window_size, overlap):
        """Compute rolling Pearson correlations from HEOM temporal states."""
        self._build_bath()

        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, "pearson", window_size, overlap)
    
    def plv_sim(self, omega_d, window_size, overlap):
        """Compute rolling phase locking values from HEOM temporal states."""
        self._build_bath()

        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, "plv", window_size, overlap)

    def phase_corr_sim(self, omega_d, corr_names, window_size=None, overlap=None):
        """Compute phase trajectories and requested correlations for HEOM."""
        self._build_bath()

        exc, sp, states = self._worker(omega_d, store_states=True)
        phases, t = self._phase_sim_helper(states)
        if isinstance(corr_names, list):
            corrs = []
            for corr_name in corr_names:
                corr, t = self._cor_sim_helper(states, corr_name, window_size, overlap)
                corrs.append(corr)
            return phases, corrs, t

        corr, t = self._cor_sim_helper(states, corr_names, window_size, overlap)
        return phases, corr, t