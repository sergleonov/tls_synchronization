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
# Additional concrete solvers get added here as they land, e.g.:
# from .tempo import TempoSolver
# from .tiered import TieredSolver
# from .semiclassical import SemiclassicalRK4Solver
from . import plotting, utils, backend, model, helpers
# `correlations` is planned (Pearson/PLV/mutual-info over a Dynamics) but is not
# a module yet; re-enable this import and its __all__ entry once it lands.
# from . import correlations

__all__ = [
    # solvers
    "Solver",
    "MarkovianSolver",
    "HeomSolver",
    # modules
    "backend",
    "model",
    "helpers",
    "plotting",
    "utils",
]