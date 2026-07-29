"""Package import surface: the public API is importable and constants are sane."""
import tls_sync
from tls_sync import (
    Solver, Lindblad, TieredSolver, HEOM, TEMPO,
    SOLVERS, SD_TYPES, HUSIMI_EVAL_METHODS, plotting, utils,
)


def test_all_exports_present():
    for name in tls_sync.__all__:
        assert hasattr(tls_sync, name), f"{name} missing from package namespace"


def test_solver_registry_constants():
    assert set(SOLVERS) == {"Markovian", "Tiered", "HEOM", "TEMPO"}
    assert set(SD_TYPES) == {"power", "drude"}
    assert set(HUSIMI_EVAL_METHODS) == {"avg", "ptrace", "diff"}


def test_classes_are_solver_subclasses():
    for cls in (Lindblad, TieredSolver, HEOM, TEMPO):
        assert issubclass(cls, Solver)


def test_submodules_have_expected_callables():
    assert callable(utils.compute_fft)
    assert callable(plotting.plot_exc_map)
