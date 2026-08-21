from tls_sync.solver import Solver
import numpy as np
from .parallel import run_parallel, parallel_eval_husimi
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
                 n_tls=2):
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
                            n_tls=d["n_tls"])
    
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
    
    def run(self, omega_d_vals, store_states=False):
        """Execute Markovian Lindblad simulations across drive frequencies."""
        worker = partial(self._worker, store_states=store_states)

        return run_parallel(
            omega_d_vals=omega_d_vals,
            worker=worker,
            n_time=self.n_time,
            store_states=store_states,
            desc="Markovian simulations"
        )

    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
        """Compute Husimi-Q functions for a Lindblad simulation."""
        states = self._get_states(omega_d)
        return parallel_eval_husimi(
            states,
            self.eval_husimi,
            theta,
            phi,
            method,
            tls_idx,
            desc="Lindblad Husimi-Q Computation"
        )
    
    def _get_states(self, omega_d):
        _, _, states = self._worker(omega_d, store_states=True)
        return states
