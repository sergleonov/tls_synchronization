"""Shared fixtures and test configuration.

All solver fixtures use deterministic TLS frequencies and deliberately tiny
time/frequency grids so the full suite runs quickly, while still exercising
every code path (operators, Hamiltonian, drive, dissipators, pickling, the
parallel sweep, correlations and Husimi-Q evaluation).
"""
import os
import multiprocessing as mp

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")          # headless: no display needed in CI
import matplotlib.pyplot as plt  # noqa: E402

from _config import BASE  # noqa: E402  (constants shared with the test modules)

# Let CI force a specific multiprocessing start method (e.g. TLS_MP_START=spawn
# on Linux) so the pickling/serialization paths are exercised on every platform.
_START = os.environ.get("TLS_MP_START")
if _START:
    try:
        mp.set_start_method(_START, force=True)
    except (RuntimeError, ValueError):
        pass


@pytest.fixture(autouse=True)
def _mpl_clean():
    """Keep LaTeX rendering off (no system LaTeX in CI) and close figures."""
    plt.rcParams["text.usetex"] = False
    yield
    plt.close("all")


# --- individual solver fixtures (imported lazily so import errors surface per test) ---
@pytest.fixture
def lindblad():
    from tls_sync import Lindblad
    return Lindblad(**BASE)


@pytest.fixture
def tiered():
    from tls_sync import TieredSolver
    return TieredSolver(**BASE, g=0.02, omega_c=3.75, Nb=2)


@pytest.fixture
def heom_drude():
    from tls_sync import HEOM
    return HEOM(**BASE, gamma_bath=0.5, Nk=3, max_depth=3, sd_type="drude")


@pytest.fixture
def heom_power():
    from tls_sync import HEOM
    return HEOM(**BASE, gamma_bath=0.5, Nk=3, max_depth=3, sd_type="power", ohmicity=1.0)


@pytest.fixture
def tempo():
    from tls_sync import TEMPO
    return TEMPO(**BASE, gamma_bath=0.5, zeta=1, cutoff_type="exponential",
                 tcut=1.0, epsrel=1e-3)


# --- parametrized "give me every solver" fixtures ---
QUTIP_SOLVERS = ["lindblad", "tiered", "heom_drude", "heom_power"]
ALL_SOLVERS = QUTIP_SOLVERS + ["tempo"]


@pytest.fixture(params=ALL_SOLVERS)
def any_solver(request):
    """Every solver backend, one per parametrization."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=QUTIP_SOLVERS)
def qutip_solver(request):
    """QuTiP-object solvers only (states are Qobj, not raw arrays)."""
    return request.getfixturevalue(request.param)
