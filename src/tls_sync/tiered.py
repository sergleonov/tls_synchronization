from tls_sync.solver import Solver
import numpy as np
import qutip as qt

class TieredSolver(Solver):
    """Solver for Tiered system with strong coupling to a single bath mode and weak coupling to thermal bath."""

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
                 omega_c=3.75,
                 Nb=10):
        """Initialize a Tiered solver instance.

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
        g : float
            Coupling between TLS and cavity mode.
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
        omega_c : float
            Cavity mode frequency.
        Nb : int
            Number of cavity Fock states.
        """
        
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

    def _embed_operators(self, sx, sy, sz, sp, sm):
        """Embed the TLS operators into the TLS-cavity space and build the
        cavity annihilation operator on the full space."""
        I_cav = qt.qeye(self.Nb)
        def with_cavity(ops):
            return [self._tensor([op, I_cav]) for op in ops]
        self.a = self._tensor([qt.qeye(2)] * self.n_tls + [qt.destroy(self.Nb)])
        return (with_cavity(sx), with_cavity(sy), with_cavity(sz),
                with_cavity(sp), with_cavity(sm))

    def _model_hamiltonian(self):
        """Cavity Hamiltonian plus collective TLS-cavity coupling."""
        return (self.omega_c * self.a.dag() * self.a
                + self.g * sum(self.sx) * (self.a.dag() + self.a))

    def _build_dissipators(self):
        """Standard TLS collapse operators plus cavity decay/excitation."""
        super()._build_dissipators()
        n_th_mode = 1 / (np.exp(self.omega_c / self.T) - 1)
        self.c_ops.append(np.sqrt(self.lam * (n_th_mode + 1)) * self.a)
        self.c_ops.append(np.sqrt(self.lam * n_th_mode) * self.a.dag())

    def __getstate__(self):
        """Return the picklable state of the Tiered solver."""
        d = super().__getstate__()
        d["omega_c"] = self.omega_c
        d["Nb"] = self.Nb
        d["g"] = self.g
        return d

    def __setstate__(self, d):
        """Reconstruct the Tiered solver from a saved state."""
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
                            omega_c=d["omega_c"],
                            Nb=d["Nb"])
    
    def get_name(self):
        """Return the solver name."""
        return self._name
    
    def __str__(self):
        return super().__str__() + f"_mode_{self.omega_c}_Nb{self.Nb}_g_{self.g}"
    
    def _worker(self, omega_d, store_states=False):
        """Run a single Tiered model simulation for a specific drive frequency."""
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