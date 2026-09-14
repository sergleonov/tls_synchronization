import numpy as np
import warnings

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from tls_sync.helpers import Operators, Drive
from tls_sync.backend import Backend

SD_TYPES = ("drude", "power")

#TODO: write tests for model class

#TODO: add Jxx coupling terms to the hamiltonians

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
    sd_type : {'drude', 'power'}
        Spectral-density family. 'drude' is a Drude-Lorentz (Lorentzian) bath;
        'power' is a power-law bath with a cutoff.
    coupling : float
        System-bath coupling strength (``lam`` for Drude, ``alpha`` for Ohmic).
    cutoff : float
        Bath cutoff frequency (``gamma`` damping for Drude, ``wc`` for Ohmic).
    temperature : float
        Bath temperature.
    ohmicity : float or None
        Power-law exponent (``s``/``zeta``). Required for 'power', ignored for
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
        if self.sd_type == "power" and self.ohmicity is None:
            raise ValueError("A 'power' bath requires `ohmicity` (the power-law exponent).")

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

# --- shared dissipation helpers ------------------------------------------- #

def _bose(omega: float, temperature: float) -> float:
    """Bose-Einstein occupation ``1/(exp(omega/T) - 1)``; 0 for ``T <= 0``."""
    if temperature <= 0.0:
        return 0.0
    return 1.0 / (np.exp(omega / temperature) - 1.0)


def _tls_thermal_collapse_ops(lowers, raisers, omegas, coupling, temperature):
    """Weak-coupling thermal Lindblad collapse operators for a set of modes::

        sqrt(coupling * (n_i + 1)) * lower_i     (emission / decay)
        sqrt(coupling *  n_i     ) * raiser_i    (absorption / thermal gain)

    with ``n_i = _bose(omega_i, temperature)``. The absorption operator is
    omitted where ``n_i == 0`` (``T <= 0``), at which it would vanish.
    """
    c_ops = []
    for lower, raiser, omega in zip(lowers, raisers, omegas):
        n = _bose(omega, temperature)
        c_ops.append(float(np.sqrt(coupling * (n + 1.0))) * lower)
        if n > 0.0:
            c_ops.append(float(np.sqrt(coupling * n)) * raiser)
    return c_ops


class TLSChainModel(Model):
    """A chain of ``n_tls`` two-level systems with sz-sz coupling and a drive.

    Hamiltonian (angular frequencies):

        H = sum_i 0.5 * omega_i * sz_i  +  J * sum_{i<j} sz_i sz_j

    The collective drive couples to ``sum_i sx_i``. Open-system behaviour is
    described by an optional :class:`Bath`; solvers turn that into their own
    representation.
    """

    def __init__(self,
                 omega_tls: Sequence[float],
                 J: float = 0.02,
                 Omega_amp: float = 0.1,
                 T_drive: float = 100.0,
                 n_tls: int = 2,
                 bath: Bath | None = None) -> None:
        """Initialize the TLS-chain model.

        Parameters
        ----------
        omega_tls : array_like
            TLS frequencies (required); its length must equal ``n_tls``.
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
        self.omega_tls = np.asarray(omega_tls, dtype=float)
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
        """Weak-coupling thermal Lindblad collapse operators rendered from
        ``self.bath`` (the model's *Lindblad* rendering of its environment -- the
        collapse operators any Lindblad-form solver integrates). Reads only the
        bath's coupling and temperature (weak-coupling limit), so it is
        independent of the spectral-density family. Non-Lindblad solvers (HEOM,
        TEMPO) do not use this; they render ``self.bath`` from its spectral
        density instead.

        Warns (rather than silently returning nothing) when there is no bath: the
        chain then has no dissipation at all and the run is closed (unitary).
        """
        if self.bath is None:
            warnings.warn(
                f"{type(self).__name__} has no bath: no dissipation is produced, "
                "so the run is closed (unitary). Attach a Bath for an open system.",
                stacklevel=2,
            )
            return []
        return _tls_thermal_collapse_ops(ops.sm, ops.sp, self.omega_tls,
                                     self.bath.coupling, self.bath.temperature)

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
    """``n_tls`` TLS collectively coupled to one *quantized* cavity Fock mode
    (the full-quantum, Tiered case).

    The cavity is part of the Hilbert space, so the whole system lives on
    ``subsystem_dims == [2]*n_tls + [Nb]``. Hamiltonian (angular frequencies)::

        H = sum_i 0.5 omega_i sz_i          # bare TLS
          + J sum_{i<j} sz_i sz_j           # TLS-TLS coupling (single J)
          + omega_c a^dag a                 # cavity
          + g (sum_i sx_i) (a + a^dag)      # TLS-cavity coupling (single g)

    ``build_operators`` embeds the TLS operators into the TLS-cavity space and
    exposes the cavity annihilation as ``ops.aux["a"]``; ``build_dissipators``
    returns the cavity's thermal photon loss (the TLS thermal :class:`Bath`, if
    any, is rendered by the solver, as for :class:`TLSChainModel`).

    For the *semiclassical* (mean-field) treatment of the cavity use
    :class:`SemiclassicalCavityModel`, whose Hilbert space is the TLS chain alone.

    There is no dedicated "tiered" solver: run this model with
    :class:`~tls_sync.markovian.MarkovianSolver`. Once the physics lives on the
    model (operators, Hamiltonian, thermal + cavity dissipators, drive, initial
    state), the solver is a plain ``qutip.mesolve`` integrator that treats the
    cavity Fock mode as just another subsystem -- so ``MarkovianSolver(model)``
    *is* the tiered solver. Pass cavity observables (e.g. ``a.dag()*a``) as
    ``e_ops`` when you want them; the defaults are the TLS collective operators.

    Simplifying assumptions (provisional): a single sz-sz TLS-TLS coupling ``J``
    and a single, uniform TLS-cavity coupling ``g``.

    Parameters
    ----------
    omega_tls : array_like
        TLS frequencies (required); its length must equal ``n_tls``.
    J : float
        TLS-TLS sz-sz coupling.
    Omega_amp : float
        Collective TLS drive amplitude.
    T_drive : float
        Duration the drive stays on.
    omega_c : float
        Cavity mode frequency.
    g : float
        Uniform TLS-cavity coupling.
    Nb : int
        Cavity Fock-space truncation.
    kappa : float
        Cavity decay rate; the thermal loss/gain ops are returned by
        ``build_dissipators`` when positive.
    n_tls : int
        Number of two-level systems.
    bath : Bath or None
        TLS environment spectral density, rendered by the solver.
    """

    def __init__(self,
                 omega_tls: Sequence[float],
                 J: float = 0.02,
                 Omega_amp: float = 0.1,
                 T_drive: float = 100.0,
                 omega_c: float = 4.0,
                 g: float = 0.02,
                 Nb: int = 10,
                 kappa: float = 0.0,
                 n_tls: int = 2,
                 bath: Bath | None = None) -> None:
        self.n_tls = n_tls
        self.omega_tls = np.asarray(omega_tls, dtype=float)
        if len(self.omega_tls) != n_tls:
            raise ValueError("omega_tls length must equal n_tls.")
        self.J = J
        self.Omega_amp = Omega_amp
        self.T_drive = T_drive
        self.omega_c = omega_c
        self.g = g
        self.Nb = Nb
        self.kappa = kappa
        self.bath = bath

    @property
    def subsystem_dims(self) -> list[int]:
        return [2] * self.n_tls + [self.Nb]

    def build_operators(self, backend: Backend) -> Operators:
        """TLS operators embedded in the TLS-cavity space, with the cavity
        annihilation operator in ``aux["a"]``."""
        I2 = backend.identity(2)
        I_cav = backend.identity(self.Nb)

        def embed(op: Any, idx: int) -> Any:
            mats = [I2] * self.n_tls + [I_cav]
            mats[idx] = op
            return backend.tensor(mats)

        sx = [embed(backend.sigma("x"), i) for i in range(self.n_tls)]
        sy = [embed(backend.sigma("y"), i) for i in range(self.n_tls)]
        sz = [embed(backend.sigma("z"), i) for i in range(self.n_tls)]
        sp = [embed(backend.sigma("+"), i) for i in range(self.n_tls)]
        sm = [embed(backend.sigma("-"), i) for i in range(self.n_tls)]

        a = backend.tensor([I2] * self.n_tls + [backend.destroy(self.Nb)])

        collective_sp = sum(sp)
        collective_sm = sum(sm)
        collective_exc = backend.mul(collective_sp, collective_sm)

        return Operators(sx=sx, sy=sy, sz=sz, sp=sp, sm=sm,
                         collective_sp=collective_sp,
                         collective_sm=collective_sm,
                         collective_exc=collective_exc,
                         aux={"a": a})

    def build_hamiltonian(self, backend: Backend, ops: Operators) -> Any:
        """Embedded TLS part + cavity + TLS-cavity coupling."""
        a = ops.aux["a"]
        H = 0.5 * self.omega_tls[0] * ops.sz[0]
        for i in range(1, self.n_tls):
            H = H + 0.5 * self.omega_tls[i] * ops.sz[i]
        for i in range(self.n_tls):
            for j in range(i + 1, self.n_tls):
                H = H + self.J * backend.mul(ops.sz[i], ops.sz[j])
        H = H + self.omega_c * backend.mul(backend.dag(a), a)       # omega_c a^dag a
        Sx = sum(ops.sx)
        H = H + self.g * backend.mul(Sx, backend.dag(a) + a)        # g Sx (a + a^dag)
        return H

    def build_dissipators(self, backend: Backend, ops: Operators) -> list[Any]:
        """All Lindblad collapse operators for the tiered system: TLS thermal
        (from the bath) plus the cavity's thermal photon loss.

        - TLS: weak-coupling thermal collapse ops rendered from ``self.bath`` (as
          in :class:`TLSChainModel`), on the embedded TLS operators.
        - Cavity: ``sqrt(kappa (n_c+1)) a`` + ``sqrt(kappa n_c) a^dag`` with
          ``n_c`` the Bose occupation at ``omega_c`` and the shared environment
          temperature (pure loss when there is no bath or ``T <= 0``).

        Warns (rather than silently dropping collapse operators) when there is no
        bath: the TLS then get no thermal dissipation, and a lossy cavity is
        treated as zero-temperature (its thermal-gain operator is omitted).
        """
        c_ops: list[Any] = []
        if self.bath is None:
            warnings.warn(
                f"{type(self).__name__} has no bath: the TLS get no thermal "
                "dissipation, and any cavity loss is treated as zero-temperature "
                "(thermal-gain operator omitted). Attach a Bath for an open system.",
                stacklevel=2,
            )
        else:
            c_ops += _tls_thermal_collapse_ops(ops.sm, ops.sp, self.omega_tls,
                                           self.bath.coupling, self.bath.temperature)
        if self.kappa > 0.0:
            a = ops.aux["a"]
            T = self.bath.temperature if self.bath is not None else 0.0
            n_c = _bose(self.omega_c, T)
            c_ops.append(float(np.sqrt(self.kappa * (n_c + 1.0))) * a)
            if n_c > 0.0:
                c_ops.append(float(np.sqrt(self.kappa * n_c)) * backend.dag(a))
        return c_ops

    def initial_state(self, backend: Backend, ops: Operators) -> Any:
        """Product TLS ground state ``|1...1>`` tensored with the cavity vacuum."""
        #TODO: verify this logic
        I2 = backend.identity(2)
        ground = 0.5 * (I2 - backend.sigma("z"))          # |1><1| per TLS
        return backend.tensor([ground] * self.n_tls + [self._cavity_vacuum(backend)])

    def drive(self, ops: Operators) -> Drive:
        """Collective TLS drive coupling to ``sum_i sx_i``."""
        drive_op = sum(ops.sx)
        amp, t_drive = self.Omega_amp, self.T_drive

        def coefficient(t: float, omega_d: float) -> float:
            return 0.5 * amp * np.cos(omega_d * t) if 0.0 <= t <= t_drive else 0.0

        return Drive(operator=drive_op, coefficient=coefficient)

    def bath_coupling_op(self, ops: Operators) -> Any:
        """System operator the TLS bath couples to (collective sx)."""
        return sum(ops.sx)

    def _cavity_vacuum(self, backend: Backend) -> Any:
        """Fock vacuum projector ``|0><0|`` on the ``Nb``-level cavity.

        Built from ``a = destroy(Nb)`` via the number-operator projector
        ``prod_{k>=1} (k I - N) / k`` (which is 1 on the ``N = 0`` eigenstate and
        0 elsewhere), since the Backend ABC exposes no Fock/basis primitive.
        """
        #TODO: verify this logic
        a = backend.destroy(self.Nb)
        N = backend.mul(backend.dag(a), a)
        I_cav = backend.identity(self.Nb)
        proj = I_cav
        for k in range(1, self.Nb):
            proj = backend.mul(proj, (k * I_cav - N) * (1.0 / k))
        return proj


class SemiclassicalCavityModel(TLSChainModel):
    """``n_tls`` TLS (in the Hilbert space) coupled to a *classical* cavity mode.

    The TLS are treated exactly as in :class:`TLSChainModel`: all the
    Hilbert-space machinery -- operators, bare TLS Hamiltonian, ground state,
    collective drive, bath coupling, and ``subsystem_dims == [2]*n_tls`` -- is
    inherited unchanged. The cavity is *not* a Fock mode here; it is a classical
    mean-field amplitude ``alpha`` that the semiclassical solver integrates
    alongside the TLS density matrix. This model therefore only *adds* the cavity
    as scalar parameters, and the solver assembles the coupling
    ``g (sum_i sx_i)(alpha + alpha*)`` and the cavity equation of motion (built
    from ``omega_c``, ``kappa``, ``eta``) from them.

    Note the split in how a parameter is used: ``kappa`` here is the classical
    cavity damping in ``dalpha/dt``, whereas in :class:`TLSCavityModel` the same
    ``kappa`` becomes a Lindblad collapse operator. For the full-quantum
    (Fock-mode) treatment use :class:`TLSCavityModel`.

    Parameters (additional to :class:`TLSChainModel`)
    -------------------------------------------------
    omega_c : float
        Cavity mode frequency.
    g : float
        Uniform TLS-cavity coupling.
    kappa : float
        Cavity (classical) energy decay rate.
    eta : float
        Direct cavity-drive amplitude.
    gamma : float
        Phenomenological TLS relaxation rate (T=0 spontaneous emission);
        ``sqrt(gamma) sm_i`` per TLS. Passed here, not on the :class:`Bath`,
        because a fixed rate has no spectral density for HEOM/TEMPO to render.
    gamma_phi : float
        Phenomenological TLS pure-dephasing rate; ``sqrt(gamma_phi/2) sz_i``.

    Note: compared to Salil's code, this model assumes uniform TLS-cavity coupling ``g``, 
    unifrom TLS-TLS interaction ``J``, unifrom decay rates ``gamma`` and ``gamma_phi`` for TLS
    """

    def __init__(self,
                 omega_tls: Sequence[float],
                 J: float = 0.02,
                 Omega_amp: float = 0.1,
                 T_drive: float = 100.0,
                 omega_c: float = 4.0,
                 g: float = 0.02,
                 kappa: float = 0.001,
                 eta: float = 0.0,
                 gamma: float = 0.0,
                 gamma_phi: float = 0.0,
                 n_tls: int = 2,
                 bath: Bath | None = None) -> None:
        super().__init__(omega_tls=omega_tls, J=J, Omega_amp=Omega_amp,
                         T_drive=T_drive, n_tls=n_tls, bath=bath)
        self.omega_c = omega_c
        self.g = g
        self.kappa = kappa
        self.eta = eta
        self.gamma = gamma
        self.gamma_phi = gamma_phi

    def build_dissipators(self, backend: Backend, ops: Operators) -> list[Any]:
        """TLS Lindblad channels: an optional thermal bath (rendered only if one
        is set) plus the prototype's phenomenological fixed-rate channels, one
        pair per TLS::

            sqrt(gamma)        sm_i     (relaxation, T = 0)
            sqrt(gamma_phi/2)  sz_i     (pure dephasing)

        Unlike :class:`TLSChainModel` / :class:`TLSCavityModel`, a missing bath is
        *not* warned about here: the semiclassical model's dissipation is the
        phenomenological ``gamma``/``gamma_phi``, so running without a bath is the
        normal case. The classical cavity damping ``kappa`` enters the mean-field
        ``alpha`` equation of motion in the solver, not as a collapse operator.
        """
        #NOTE: do we need to add temperature dependence to the TLS relaxation?
        c_ops: list[Any] = []
        for i in range(self.n_tls):
            if self.gamma > 0.0:
                c_ops.append(float(np.sqrt(self.gamma)) * ops.sm[i])
            if self.gamma_phi > 0.0:
                c_ops.append(float(np.sqrt(self.gamma_phi / 2.0)) * ops.sz[i])
        return c_ops