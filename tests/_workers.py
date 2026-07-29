"""Picklable, qutip-free stand-ins for exercising the parallel machinery.

Defined at module scope so ``spawn`` worker processes can import and unpickle
them by qualified name.
"""
import numpy as np

N_TIME = 4


def dummy_worker(omega_d, store_states=False):
    """Return deterministic arrays; the excitation encodes the drive frequency."""
    exc = np.full(N_TIME, float(omega_d))
    sp = np.arange(N_TIME) + 1j
    if store_states:
        states = [np.eye(2, dtype=complex) for _ in range(N_TIME)]
        return exc, sp, states
    return exc, sp


def dummy_eval(state, theta, phi, method, tls_idx):
    """Return a grid whose constant value is Re(tr(state))."""
    val = float(np.trace(np.asarray(state)).real)
    return np.ones((len(theta), len(phi))) * val
