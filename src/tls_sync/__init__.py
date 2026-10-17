"""TLS synchronization solver package.
 
Public API: solver classes are exposed at the top level, e.g.
 
    from tls_sync import MarkovianSolver
 
Everything else stays under its own module, reached as ``tls_sync.<module>``:
 
    tls_sync.backend.QutipBackend
    tls_sync.model.TLSChainModel
    tls_sync.model.Bath
    tls_sync.model.SD_TYPES
    tls_sync.helpers.Dynamics
    tls_sync.plotting, tls_sync.utils
"""

from .solver import Solver              # abstract base, for typing / subclassing
from .markovian import MarkovianSolver
from .heom import HeomSolver
from .tempo import TempoSolver
from .semiclassical import SemiclassicalSolver
# The tiered (TLS + quantized cavity) case has no dedicated solver: run a
# tls_sync.model.TLSCavityModel with MarkovianSolver (the cavity is just another
# subsystem for qutip.mesolve).
from . import plotting, utils, backend, model, helpers

__all__ = [
    # solvers
    "Solver",
    "MarkovianSolver",
    "HeomSolver",
    "TempoSolver",
    "SemiclassicalSolver",
    # modules
    "backend",
    "model",
    "helpers",
    "plotting",
    "utils",
]