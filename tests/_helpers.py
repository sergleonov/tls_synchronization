"""Small, dependency-light helpers shared across the test suite.

Kept in a plain importable module (not conftest) so that objects defined here
have a stable ``__module__`` and can be unpickled by ``spawn`` worker processes.
"""
import numpy as np


def to_dense(op):
    """Return a dense NumPy array for a QuTiP Qobj or an array-like."""
    if hasattr(op, "full"):          # qutip Qobj
        return np.asarray(op.full())
    return np.asarray(op)


def state_trace(state):
    """Trace of a state whether it is a QuTiP Qobj or a raw density matrix."""
    if hasattr(state, "tr"):         # qutip Qobj
        return complex(state.tr())
    return complex(np.trace(np.asarray(state)))


def is_hermitian(op, atol=1e-8):
    """True if an operator (Qobj or array) equals its conjugate transpose."""
    m = to_dense(op)
    return np.allclose(m, m.conj().T, atol=atol)


def is_density_matrix(state, atol=1e-6):
    """Trace ~ 1, Hermitian, and positive semi-definite (within tolerance)."""
    m = to_dense(state)
    if not np.isclose(np.trace(m).real, 1.0, atol=atol):
        return False
    if not np.allclose(m, m.conj().T, atol=atol):
        return False
    evals = np.linalg.eigvalsh((m + m.conj().T) / 2)
    return evals.min() > -atol


def assert_params_equal(d1, d2):
    """Assert two parameter dicts (possibly containing arrays) are equal."""
    assert set(d1) == set(d2), f"key mismatch: {set(d1) ^ set(d2)}"
    for k in d1:
        a, b = d1[k], d2[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert np.array_equal(np.asarray(a), np.asarray(b)), f"array param {k!r} differs"
        else:
            assert a == b, f"param {k!r} differs: {a!r} != {b!r}"
