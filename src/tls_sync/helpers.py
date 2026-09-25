import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from tls_sync.backend import Backend

# ============ Operators ============
# data class containing all necessary ops for solvers

@dataclass
class Operators:
    """Backend-native operators for one model instance."""
    sx: list[Any]
    sy: list[Any]
    sz: list[Any]
    sp: list[Any]
    sm: list[Any]
    collective_sp: Any
    collective_sm: Any
    collective_exc: Any
    # extra model operators, e.g. {"a": <cavity annihilation>} for TLS+cavity
    aux: dict[str, Any] = field(default_factory=dict)

@dataclass
class Drive:
    """Time-dependent drive: which operator it couples to and its coefficient.

    `coefficient(t, omega_d)` encodes the envelope (incl. any on/off time) and
    the carrier. Keeping this in the Model makes the drive *physics*; each Solver
    consumes it in its own way (QobjEvo, RK4 substep, ...).
    """
    operator: Any
    coefficient: Callable[[float, float], float]

@dataclass
class Dynamics:
    """Outputs of a solver run over a drive-frequency sweep.

    Attributes
    ----------
    backend : Backend
        The representation the states/operators are in (for downstream ops).
    times : np.ndarray
        Output time grid, shape ``(n_time,)``.
    omegas : np.ndarray
        Drive frequencies swept, shape ``(n_omega,)`` (the leading axis).
    expectations : np.ndarray | None
        A single ``(n_ops, n_omega, n_time)`` array; ``expectations[j]`` is the
        ``(n_omega, n_time)`` heatmap grid for ``e_ops[j]``. None when the run
        produced no expectations (e.g. an empty ``e_ops``).
    states : Sequence[Sequence[Any]] | None
        Full trajectory indexed ``[i_omega][i_time]``, or None if not stored.
    extra : dict[str, Any] | None
        Solver-specific data that is not an operator expectation, e.g. a
        semiclassical solver's classical cavity field ``{"alpha": ...}``.
    """

    backend: Backend
    times: np.ndarray
    omegas: np.ndarray
    expectations: np.ndarray | None
    states: Sequence[Sequence[Any]] | None
    extra: dict[str, Any] | None
