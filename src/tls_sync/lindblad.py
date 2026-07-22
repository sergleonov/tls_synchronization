from src.tls_sync.solver import Solver
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import qutip as qt
from functools import partial

class Lindblad(Solver):
    """Markovian Lindblad solver for open TLS dynamics."""

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
        """Initialize a Markovian Lindblad solver instance.

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
        """
        
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
        """Return the picklable state of the Lindblad solver."""
        return super().__getstate__()

    def __setstate__(self, d):
        """Reconstruct the Lindblad solver from a saved state."""
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
    
    def get_name(self):
        """Return the solver name."""
        return self._name

    def __str__(self):
        """Return a string summary of the solver configuration."""
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
        """Execute Markovian Lindblad simulations across drive frequencies."""
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
        """Compute Husimi-Q functions for a Lindblad simulation."""
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
        """Compute TLS phase trajectories for a Lindblad run."""
        exc, sp, states = self._worker(omega_d, store_states=True)
        
        return self._phase_sim_helper(states)
    
    def pearson_sim(self, omega_d, window_size, overlap):
        """Compute rolling Pearson correlations from Lindblad states."""
        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, "pearson", window_size, overlap)
    
    def plv_sim(self, omega_d, window_size, overlap):
        """Compute rolling phase locking values from Lindblad states."""
        exc, sp, states = self._worker(omega_d, store_states=True)

        return self._cor_sim_helper(states, "plv", window_size, overlap)
    
    def phase_corr_sim(self, omega_d, corr_names, window_size=None, overlap=None):
        """Compute phase correlations and optional additional correlation metrics."""
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