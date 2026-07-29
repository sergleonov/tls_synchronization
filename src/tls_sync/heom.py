from tls_sync import Solver, SD_TYPES
import numpy as np
from .parallel import run_parallel, parallel_eval_husimi
import qutip as qt
from qutip.solver.heom import HEOMSolver
from qutip.core.environment import DrudeLorentzEnvironment, OhmicEnvironment, ExponentialBosonicEnvironment
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
        if sd_type not in SD_TYPES: 
            raise ValueError("Error: Invalid spectral density.")

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
        if self.sd_type not in SD_TYPES:
            raise ValueError("Error: Invalid spectral density.")
        d["gamma_bath"] = self.gamma_bath
        d["Nk"] = self.Nk
        d["max_depth"] = self.max_depth
        d["sd_type"] = self.sd_type
        d["ohmicity"] = self.ohmicity
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
        if self.sd_type not in SD_TYPES:
            raise ValueError("Error: Invalid spectral density.")
        sd = self.sd_type
        if sd == "power": sd += f"_{self.ohmicity}"
        return super().__str__() + f"_gamma_bath_{self.gamma_bath}_Nk{self.Nk}_max_depth_{self.max_depth}_{sd}"
    
    def _worker(self, omega_d, bath_coeffs, store_states=False):
        """Run a single HEOM simulation for a specific drive frequency."""
        H_full = qt.QobjEvo(
        [self.H, [sum(self.sx), self.drive_coeff]],
        args = {"omega": omega_d}
        )

        bath = self._coeffs_to_bath(bath_coeffs)
        solver = HEOMSolver(
            H_full,
            (bath, sum(self.sx)),
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
        """Build and return the HEOM bath object for the chosen spectral density."""
        match self.sd_type:
            case "drude": # Drude-Lorentz
                env = DrudeLorentzEnvironment(T=self.T, lam=self.lam, gamma=self.gamma_bath, Nk=self.Nk)
                return env.approximate("matsubara", Nk=self.Nk)
            case "power": # Power Law
                env = OhmicEnvironment(T=self.T, alpha=self.lam, wc=self.gamma_bath, s=self.ohmicity) # alpha is coupling, wc is cutoff
                bath, _info = env.approximate(method="cf", tlist=self.tlist, target_rmse=None, Nr_max=self.Nk, Ni_max=self.Nk, maxfev=int(1e8))
                return bath
            case _:
                raise ValueError("Invalid spectral density type.")

    @staticmethod
    def _bath_to_coeffs(bath):
        """Extract picklable exponential expansion coefficients from a bosonic bath."""
        ck_real, vk_real, ck_imag, vk_imag = [], [], [], []
        for exp in bath.exponents:
            etype = getattr(exp.type, "name", str(exp.type))
            if etype == "R":
                ck_real.append(complex(exp.ck)); vk_real.append(complex(exp.vk))
            elif etype == "I":
                ck_imag.append(complex(exp.ck)); vk_imag.append(complex(exp.vk))
            elif etype == "RI":
                # combined term: real part uses ck, imaginary part uses ck2,
                # both sharing the same decay rate vk
                ck_real.append(complex(exp.ck)); vk_real.append(complex(exp.vk))
                ck_imag.append(complex(exp.ck2)); vk_imag.append(complex(exp.vk))
            else:
                raise ValueError(
                    f"Unexpected bath exponent type {etype!r}."
                )
        T = getattr(bath, "T", None)
        return (
            np.array(ck_real, dtype=complex),
            np.array(vk_real, dtype=complex),
            np.array(ck_imag, dtype=complex),
            np.array(vk_imag, dtype=complex),
            T,
        )

    @staticmethod
    def _coeffs_to_bath(coeffs):
        """Rebuild a bath from environment exponential expansion coefficients."""
        ck_real, vk_real, ck_imag, vk_imag, T = coeffs
        return ExponentialBosonicEnvironment(
            ck_real=list(ck_real),
            vk_real=list(vk_real),
            ck_imag=list(ck_imag),
            vk_imag=list(vk_imag),
            T=T,
        )
    
    def run(self, store_states=False):
        """Execute HEOM simulations across all configured drive frequencies."""
        if self.sd_type not in SD_TYPES:
            raise ValueError("Error: Invalid spectral density.")

        bath = self._build_bath()
        bath_coeffs = self._bath_to_coeffs(bath)

        worker = partial(self._worker, bath_coeffs=bath_coeffs, store_states=store_states)

        return run_parallel(
            omega_d_vals=self.omega_d_vals,
            worker=worker,
            n_time=self.n_time,
            store_states=store_states,
            desc="HEOM simulations",
        )
    
    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
        """Compute Husimi-Q functions for an HEOM run at a given drive frequency."""
        states = self._get_states(omega_d)
        return parallel_eval_husimi(
            states,
            self.eval_husimi,
            theta,
            phi,
            method,
            tls_idx,
            desc="HEOM Husimi-Q Computation"
        )

    def _get_states(self, omega_d):
        bath = self._build_bath()
        bath_coeffs = self._bath_to_coeffs(bath)
        _, _, states = self._worker(omega_d, bath_coeffs, store_states=True)
        return states