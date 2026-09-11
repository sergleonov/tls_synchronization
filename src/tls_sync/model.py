
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from tls_sync.helpers import Operators, Drive
from tls_sync.backend import Backend

SD_TYPES = ("drude", "ohmic")

#TODO: write tests for model class

@dataclass
class Bath:
    """Spectral density of a bosonic environment (method-independent physics).

    The same object is consumed differently by each solver, so it stores only
    the physical spectral density, never a method's representation of it (no
    Matsubara terms, no hierarchy depth). The system operator the bath couples
    to is backend-native and lives on the model (``bath_coupling_op``), not here,
    so ``Bath`` stays a plain picklable value.

    Parameters
    ----------
    sd_type : {'drude', 'ohmic'}
        Spectral-density family. 'drude' is a Drude-Lorentz (Lorentzian) bath;
        'ohmic' is a power-law bath with a cutoff.
    coupling : float
        System-bath coupling strength (``lam`` for Drude, ``alpha`` for Ohmic).
    cutoff : float
        Bath cutoff frequency (``gamma`` damping for Drude, ``wc`` for Ohmic).
    temperature : float
        Bath temperature.
    ohmicity : float or None
        Power-law exponent (``s``/``zeta``). Required for 'ohmic', ignored for
        'drude'.
    cutoff_type : str
        Shape of the power-law cutoff (used by Ohmic/power-law representations,
        e.g. oqupy's ``PowerLawSD``). Ignored for 'drude'.
    """

    sd_type: str
    coupling: float
    cutoff: float
    temperature: float
    ohmicity: float | None = None
    cutoff_type: str = "exponential"

    def __post_init__(self) -> None:
        if self.sd_type not in SD_TYPES:
            raise ValueError(f"Unknown sd_type {self.sd_type!r}; expected one of {SD_TYPES}.")
        if self.sd_type == "ohmic" and self.ohmicity is None:
            raise ValueError("An 'ohmic' bath requires `ohmicity` (the power-law exponent).")

class Model(ABC):
    """A physical system: operators, Hamiltonian, dissipators, drive, state."""

    n_tls: int

    @property
    @abstractmethod
    def subsystem_dims(self) -> list[int]:
        """Hilbert-space factorization, e.g. [2, 2] or [2, 2, Nb] with a cavity."""

    @abstractmethod
    def build_operators(self, backend: Backend) -> Operators:
        """Construct all operators in `backend`'s representation."""

    @abstractmethod
    def build_hamiltonian(self, backend: Backend, ops: Operators) -> Any:
        """Static Hamiltonian (time-dependent drive is supplied separately)."""

    @abstractmethod
    def build_dissipators(self, backend: Backend, ops: Operators) -> list[Any]:
        """Collapse operators for a Lindblad-form solver (may be empty)."""

    @abstractmethod
    def initial_state(self, backend: Backend, ops: Operators) -> Any:
        """Initial state (ket or density matrix) in `backend`'s representation."""

    @abstractmethod
    def drive(self, ops: Operators) -> Drive:
        """Drive term: coupling operator + coefficient(t, omega_d)."""

class TLSChainModel(Model):
    """A chain of ``n_tls`` two-level systems with sz-sz coupling and a drive.

    Hamiltonian (angular frequencies):

        H = sum_i 0.5 * omega_i * sz_i  +  J * sum_{i<j} sz_i sz_j

    The collective drive couples to ``sum_i sx_i``. Open-system behaviour is
    described by an optional :class:`Bath`; solvers turn that into their own
    representation.
    """

    def __init__(self,
                 omega_tls: Sequence[float] | None = None,
                 J: float = 0.02,
                 Omega_amp: float = 0.1,
                 T_drive: float = 100.0,
                 n_tls: int = 2,
                 bath: Bath | None = None) -> None:
        """Initialize the TLS-chain model.

        Parameters
        ----------
        omega_tls : array_like or None
            TLS frequencies. Defaults to ones if None (set real values in use).
        J : float
            TLS-TLS sz-sz coupling.
        Omega_amp : float
            Collective drive amplitude.
        T_drive : float
            Duration the drive stays on.
        n_tls : int
            Number of two-level systems.
        bath : Bath or None
            Environment spectral density; None for a closed system.
        """
        self.n_tls = n_tls
        self.omega_tls = (np.ones(n_tls) if omega_tls is None
                          else np.asarray(omega_tls, dtype=float))
        if len(self.omega_tls) != n_tls:
            raise ValueError("omega_tls length must equal n_tls.")
        self.J = J
        self.Omega_amp = Omega_amp
        self.T_drive = T_drive
        self.bath = bath

    @property
    def subsystem_dims(self) -> list[int]:
        return [2] * self.n_tls

    def build_operators(self, backend: Backend) -> Operators:
        I2 = backend.identity(2)

        def embed(op: Any, idx: int) -> Any:
            mats = [I2] * self.n_tls
            mats[idx] = op
            return backend.tensor(mats)

        sx = [embed(backend.sigma("x"), i) for i in range(self.n_tls)]
        sy = [embed(backend.sigma("y"), i) for i in range(self.n_tls)]
        sz = [embed(backend.sigma("z"), i) for i in range(self.n_tls)]
        sp = [embed(backend.sigma("+"), i) for i in range(self.n_tls)]
        sm = [embed(backend.sigma("-"), i) for i in range(self.n_tls)]

        collective_sp = sum(sp)
        collective_sm = sum(sm)
        collective_exc = backend.mul(collective_sp, collective_sm)

        return Operators(sx=sx, sy=sy, sz=sz, sp=sp, sm=sm,
                         collective_sp=collective_sp,
                         collective_sm=collective_sm,
                         collective_exc=collective_exc)

    def build_hamiltonian(self, backend: Backend, ops: Operators) -> Any:
        H = 0.5 * self.omega_tls[0] * ops.sz[0]
        for i in range(1, self.n_tls):
            H = H + 0.5 * self.omega_tls[i] * ops.sz[i]
        for i in range(self.n_tls):
            for j in range(i + 1, self.n_tls):
                H = H + self.J * backend.mul(ops.sz[i], ops.sz[j])
        return H

    def build_dissipators(self, backend: Backend, ops: Operators) -> list[Any]:
        """Phenomenological Lindblad terms the model defines directly (default: none).
 
        A TLS chain's dissipation is the structured environment in ``self.bath``,
        which each solver renders into its own representation (Markovian collapse
        operators, HEOM coefficients, TEMPO correlations) -- so that rendering is
        solver-specific and does not live here. This returns an empty list; a
        model with ad-hoc rates (e.g. a cavity's kappa) would return those here.
        """
        return []

    def initial_state(self, backend: Backend, ops: Operators) -> Any:
        # product ground state; for +0.5*omega*sz the lower level is (0, 1),
        # whose projector is 0.5*(I - sz)
        #TODO: verify this
        I2 = backend.identity(2)
        ground_proj = 0.5 * (I2 - backend.sigma("z"))
        return backend.tensor([ground_proj] * self.n_tls)

    def drive(self, ops: Operators) -> Drive:
        drive_op = sum(ops.sx)
        amp, t_drive = self.Omega_amp, self.T_drive

        def coefficient(t: float, omega_d: float) -> float:
            return 0.5 * amp * np.cos(omega_d * t) if 0.0 <= t <= t_drive else 0.0

        return Drive(operator=drive_op, coefficient=coefficient)

    def bath_coupling_op(self, ops: Operators) -> Any:
        """System operator the bath couples to (collective sx here)."""
        return sum(ops.sx)
    


class TLSCavityModel(Model):
    """`n_tls` TLS + a quantized cavity Fock mode (the Tiered case).

    subsystem_dims == [2]*n_tls + [Nb]; `ops.aux["a"]` is the cavity operator;
    build_hamiltonian adds omega_c a†a + g (Sx)(a + a†).
    """

    def __init__(self, omega_tls: Sequence[float], J: float, T: float,
                 Omega_amp: float, T_drive: float, omega_c: float, 
                 g: float, Nb: int, n_tls: int = 2,) -> None:
        ...