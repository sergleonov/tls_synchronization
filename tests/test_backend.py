"""Tests for :mod:`tls_sync.backends`.

The module exposes three interchangeable representations of the same quantum
operators/states -- a QuTiP ``Qobj`` backend, a dense NumPy backend, and an
OQuPy adapter that subclasses the NumPy one. The guiding property of the suite
is therefore *cross-backend agreement*: whatever representation a backend uses
internally, the numbers it produces must match a single, backend-independent
reference (and each other).

Notes
-----
* Import path assumes the source lives at ``tls_sync/backends.py`` (the package
  ``module-name`` declared in ``pyproject.toml``). Adjust the import below if
  your layout differs.
* ``qutip`` and ``oqupy`` are hard dependencies of the module (both are imported
  at module top level), so if either is missing the whole file is skipped
  cleanly rather than erroring during collection.
"""

import numpy as np
import pytest
import qutip as qt 

from tls_sync.backend import (
    Backend,
    QutipBackend,
    NumpyBackend,
    OqupyBackend,
)


# --------------------------------------------------------------------------- #
# Reference data + helpers
# --------------------------------------------------------------------------- #

# The single source of truth for the ladder/Pauli convention the whole package
# agrees on:  sigma('+') = |0><1| = [[0,1],[0,0]]  (i.e. qt.sigmap).
REF = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "+": np.array([[0, 1], [0, 0]], dtype=complex),
    "-": np.array([[0, 0], [1, 0]], dtype=complex),
}
AXES = list(REF)

RHO0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
RHO1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|


def dense(op) -> np.ndarray:
    """Coerce any backend's operator/state into a plain dense NumPy array."""
    if isinstance(op, qt.Qobj):
        return op.full()
    return np.asarray(op)


def basis_states(backend):
    """Return (|0>, |1>, |0><0|) in the representation `backend` expects."""
    if backend.name == "qutip":
        k0, k1 = qt.basis(2, 0), qt.basis(2, 1)
        return k0, k1, qt.ket2dm(k0)
    k0 = np.array([1, 0], dtype=complex)
    k1 = np.array([0, 1], dtype=complex)
    return k0, k1, np.outer(k0, k0.conj())


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(params=[QutipBackend, NumpyBackend, OqupyBackend],
                ids=lambda c: c.name)
def backend(request):
    """Every concrete backend, one at a time."""
    return request.param()


@pytest.fixture(params=[NumpyBackend, OqupyBackend], ids=lambda c: c.name)
def array_backend(request):
    """Only the dense/array backends (share NumPy state conventions)."""
    return request.param()


# --------------------------------------------------------------------------- #
# The abstract base class
# --------------------------------------------------------------------------- #

class TestBackendABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Backend()  # abstract methods are unimplemented

    def test_names(self):
        assert QutipBackend().name == "qutip"
        assert NumpyBackend().name == "numpy"
        assert OqupyBackend().name == "oqupy"

    def test_oqupy_is_a_numpy_backend(self):
        # OQuPy reuses NumPy's linear algebra; only construction + expect differ.
        assert issubclass(OqupyBackend, NumpyBackend)
        assert isinstance(OqupyBackend(), NumpyBackend)


# --------------------------------------------------------------------------- #
# sigma()
# --------------------------------------------------------------------------- #

class TestSigma:
    @pytest.mark.parametrize("axis", AXES)
    def test_matches_reference(self, backend, axis):
        assert np.allclose(dense(backend.sigma(axis)), REF[axis])

    @pytest.mark.parametrize("bad", ["q", "X", "", "0", "sigma_x"])
    def test_unknown_axis_raises_value_error(self, backend, bad):
        with pytest.raises(ValueError):
            backend.sigma(bad)

    @pytest.mark.parametrize("axis", AXES)
    def test_backends_agree(self, axis):
        a = dense(QutipBackend().sigma(axis))
        b = dense(NumpyBackend().sigma(axis))
        c = dense(OqupyBackend().sigma(axis))
        assert np.allclose(a, b)
        assert np.allclose(b, c)

    def test_numpy_returns_a_copy_not_the_table(self):
        # The NumpyBackend caches its Pauli table; sigma() must hand back a copy
        # so a caller mutating the result cannot corrupt later calls.
        be = NumpyBackend()
        s = be.sigma("x")
        s[0, 0] = 42.0
        assert be.sigma("x")[0, 0] == 0.0

    def test_hermiticity_of_paulis(self, backend):
        for axis in ("x", "y", "z"):
            op = dense(backend.sigma(axis))
            assert np.allclose(op, op.conj().T)


# --------------------------------------------------------------------------- #
# identity()
# --------------------------------------------------------------------------- #

class TestIdentity:
    @pytest.mark.parametrize("dim", [1, 2, 3, 5])
    def test_is_identity(self, backend, dim):
        assert np.allclose(dense(backend.identity(dim)), np.eye(dim))

    @pytest.mark.parametrize("dim", [2, 4])
    def test_backends_agree(self, dim):
        for be in (NumpyBackend(), OqupyBackend()):
            assert np.allclose(dense(QutipBackend().identity(dim)),
                               dense(be.identity(dim)))


# --------------------------------------------------------------------------- #
# destroy()
# --------------------------------------------------------------------------- #

class TestDestroy:
    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_annihilation_matrix(self, backend, dim):
        expected = np.diag(np.sqrt(np.arange(1, dim)), k=1).astype(complex)
        assert np.allclose(dense(backend.destroy(dim)), expected)

    def test_lowers_fock_states(self, backend):
        # a|n> = sqrt(n) |n-1>
        dim = 4
        a = dense(backend.destroy(dim))
        for n in range(1, dim):
            ket_n = np.eye(dim, dtype=complex)[n]
            ket_nm1 = np.eye(dim, dtype=complex)[n - 1]
            assert np.allclose(a @ ket_n, np.sqrt(n) * ket_nm1)

    def test_annihilates_ground_state(self, backend):
        a = dense(backend.destroy(3))
        ground = np.array([1, 0, 0], dtype=complex)
        assert np.allclose(a @ ground, 0)

    @pytest.mark.parametrize("dim", [2, 4])
    def test_backends_agree(self, dim):
        for be in (NumpyBackend(), OqupyBackend()):
            assert np.allclose(dense(QutipBackend().destroy(dim)),
                               dense(be.destroy(dim)))


# --------------------------------------------------------------------------- #
# tensor()
# --------------------------------------------------------------------------- #

class TestTensor:
    def test_two_operators_is_kron(self, backend):
        x = backend.sigma("x")
        z = backend.sigma("z")
        assert np.allclose(dense(backend.tensor([x, z])),
                           np.kron(REF["x"], REF["z"]))

    def test_three_operators(self, backend):
        ops = [backend.sigma("x"), backend.identity(2), backend.sigma("z")]
        expected = np.kron(np.kron(REF["x"], np.eye(2)), REF["z"])
        assert np.allclose(dense(backend.tensor(ops)), expected)

    def test_single_operator_roundtrips(self, backend):
        x = backend.sigma("x")
        assert np.allclose(dense(backend.tensor([x])), REF["x"])

    def test_empty_raises_value_error(self, backend):
        with pytest.raises(ValueError):
            backend.tensor([])

    def test_dimensions(self, backend):
        # 2 (x) 3 (x) 2 -> 12-dimensional operator
        ops = [backend.sigma("x"), backend.identity(3), backend.sigma("z")]
        assert dense(backend.tensor(ops)).shape == (12, 12)


# --------------------------------------------------------------------------- #
# mul() -- must be matrix product, never elementwise
# --------------------------------------------------------------------------- #

class TestMul:
    def test_pauli_x_squared_is_identity(self, backend):
        x = backend.sigma("x")
        assert np.allclose(dense(backend.mul(x, x)), np.eye(2))

    def test_ladder_product(self, backend):
        # sigma+ sigma- = |0><0|
        sp = backend.sigma("+")
        sm = backend.sigma("-")
        assert np.allclose(dense(backend.mul(sp, sm)), RHO0)

    def test_matches_reference_matmul(self, backend):
        x, y = backend.sigma("x"), backend.sigma("y")
        assert np.allclose(dense(backend.mul(x, y)), REF["x"] @ REF["y"])

    def test_not_elementwise_for_arrays(self, array_backend):
        # The whole reason mul() exists: numpy `*` is elementwise. Guard it.
        x = array_backend.sigma("x")
        assert not np.allclose(dense(array_backend.mul(x, x)),
                               np.asarray(x) * np.asarray(x))


# --------------------------------------------------------------------------- #
# dag()
# --------------------------------------------------------------------------- #

class TestDag:
    def test_ladder_conjugates_swap(self, backend):
        assert np.allclose(dense(backend.dag(backend.sigma("+"))), REF["-"])
        assert np.allclose(dense(backend.dag(backend.sigma("-"))), REF["+"])

    def test_hermitian_operator_is_fixed(self, backend):
        y = backend.sigma("y")
        assert np.allclose(dense(backend.dag(y)), REF["y"])

    def test_destroy_dagger_is_creation(self, backend):
        a = backend.destroy(4)
        assert np.allclose(dense(backend.dag(a)), dense(a).conj().T)

    def test_double_dagger_is_identity_op(self, backend):
        a = backend.destroy(3)
        assert np.allclose(dense(backend.dag(backend.dag(a))), dense(a))


# --------------------------------------------------------------------------- #
# commutator() -- concrete helper on the ABC, exercised through each backend
# --------------------------------------------------------------------------- #

class TestCommutator:
    def test_pauli_algebra(self, backend):
        # [sigma_x, sigma_y] = 2i sigma_z
        x, y = backend.sigma("x"), backend.sigma("y")
        assert np.allclose(dense(backend.commutator(x, y)), 2j * REF["z"])

    def test_cyclic(self, backend):
        # [sigma_y, sigma_z] = 2i sigma_x
        y, z = backend.sigma("y"), backend.sigma("z")
        assert np.allclose(dense(backend.commutator(y, z)), 2j * REF["x"])

    def test_self_commutator_is_zero(self, backend):
        x = backend.sigma("x")
        assert np.allclose(dense(backend.commutator(x, x)), 0)


# --------------------------------------------------------------------------- #
# expect() -- single state, all backends
# --------------------------------------------------------------------------- #

class TestExpectSingleState:
    def test_sigmaz_eigenstates(self, backend):
        z = backend.sigma("z")
        k0, k1, rho0 = basis_states(backend)
        assert backend.expect(z, k0) == pytest.approx(1.0)
        assert backend.expect(z, k1) == pytest.approx(-1.0)
        assert backend.expect(z, rho0) == pytest.approx(1.0)

    def test_returns_complex_scalar(self, backend):
        z = backend.sigma("z")
        k0, _, _ = basis_states(backend)
        assert isinstance(backend.expect(z, k0), complex)

    def test_superposition(self, backend):
        # |+x> = (|0>+|1>)/sqrt2  =>  <sigma_z> = 0, <sigma_x> = 1
        x, z = backend.sigma("x"), backend.sigma("z")
        if backend.name == "qutip":
            plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        else:
            plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert backend.expect(z, plus) == pytest.approx(0.0, abs=1e-9)
        assert backend.expect(x, plus) == pytest.approx(1.0)


class TestExpectArrayMultiState:
    """The dense backends also accept collections of states."""

    def test_stack_of_density_matrices(self, array_backend):
        # A 3-D stack survives np.asarray, so both NumPy and OQuPy handle it.
        z = array_backend.sigma("z")
        stack = np.stack([RHO0, RHO1])
        out = array_backend.expect(z, stack)
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, [1, -1])

    def test_list_of_kets_numpy_only(self):
        # NumpyBackend explicitly special-cases list/tuple inputs.
        be = NumpyBackend()
        z = be.sigma("z")
        out = be.expect(z, [np.array([1, 0]), np.array([0, 1])])
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, [1, -1])


# --------------------------------------------------------------------------- #
# ptrace() -- signatures differ between the qutip and dense backends
# --------------------------------------------------------------------------- #

class TestPtraceQutip:
    def _bell(self):
        ket = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
               + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
        return qt.ket2dm(ket)

    def test_bell_reduced_state_is_maximally_mixed(self):
        be = QutipBackend()
        reduced = be.ptrace(self._bell(), [0])
        assert np.allclose(dense(reduced), 0.5 * np.eye(2))

    def test_product_state_reduces_to_its_factor(self):
        be = QutipBackend()
        rho = qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.basis(2, 1)))
        assert np.allclose(dense(be.ptrace(rho, [0])), RHO0)
        assert np.allclose(dense(be.ptrace(rho, [1])), RHO1)


class TestPtraceDense:
    """NumpyBackend.ptrace is a static method taking (rho, dims, keep)."""

    def _bell_rho(self):
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        return np.outer(bell, bell.conj())

    def test_bell_reduced_state_is_maximally_mixed(self, array_backend):
        reduced = array_backend.ptrace(self._bell_rho(), [2, 2], [0])
        assert np.allclose(reduced, 0.5 * np.eye(2))

    def test_product_state_reduces_to_its_factor(self, array_backend):
        prod = np.kron([1, 0], [0, 1]).astype(complex)
        rho = np.outer(prod, prod.conj())
        assert np.allclose(array_backend.ptrace(rho, [2, 2], [0]), RHO0)
        assert np.allclose(array_backend.ptrace(rho, [2, 2], [1]), RHO1)

    def test_keep_both_returns_full_state(self, array_backend):
        rho = self._bell_rho()
        assert np.allclose(array_backend.ptrace(rho, [2, 2], [0, 1]), rho)

    def test_unequal_subsystem_dimensions(self, array_backend):
        # 2 (x) 3 product state; tracing out the qubit leaves the qutrit intact.
        qubit = np.array([1, 0], dtype=complex)
        qutrit = np.array([0, 1, 0], dtype=complex)
        psi = np.kron(qubit, qutrit)
        rho = np.outer(psi, psi.conj())
        reduced = array_backend.ptrace(rho, [2, 3], [1])
        expected = np.outer(qutrit, qutrit.conj())
        assert reduced.shape == (3, 3)
        assert np.allclose(reduced, expected)

    def test_agrees_with_qutip(self):
        rho = self._bell_rho()
        q_reduced = dense(QutipBackend().ptrace(
            qt.Qobj(rho, dims=[[2, 2], [2, 2]]), [0]))
        n_reduced = NumpyBackend.ptrace(rho, [2, 2], [0])
        assert np.allclose(q_reduced, n_reduced)


# --------------------------------------------------------------------------- #
# to_density_matrix()
# --------------------------------------------------------------------------- #

class TestToDensityMatrix:
    def test_ket_becomes_projector(self, backend):
        k0, _, _ = basis_states(backend)
        assert np.allclose(dense(backend.to_density_matrix(k0)), RHO0)

    def test_density_matrix_passthrough(self, backend):
        _, _, rho0 = basis_states(backend)
        assert np.allclose(dense(backend.to_density_matrix(rho0)), RHO0)

    def test_result_is_a_valid_density_matrix(self, backend):
        # |+x> -> rho should be Hermitian, unit trace, and idempotent (pure).
        if backend.name == "qutip":
            state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        else:
            state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = dense(backend.to_density_matrix(state))
        assert np.allclose(rho, rho.conj().T)
        assert np.trace(rho) == pytest.approx(1.0)
        assert np.allclose(rho @ rho, rho)

    def test_qutip_bra_is_accepted(self):
        be = QutipBackend()
        bra = qt.basis(2, 0).dag()
        assert np.allclose(dense(be.to_density_matrix(bra)), RHO0)

    def test_numpy_column_vector(self, array_backend):
        col = np.array([[1], [0]], dtype=complex)
        assert np.allclose(array_backend.to_density_matrix(col), RHO0)

    def test_numpy_1d_and_column_agree(self, array_backend):
        row = array_backend.to_density_matrix(np.array([0, 1], dtype=complex))
        col = array_backend.to_density_matrix(np.array([[0], [1]], dtype=complex))
        assert np.allclose(row, col)
        assert np.allclose(row, RHO1)


# --------------------------------------------------------------------------- #
# NumpyBackend.to_qobj() -- bridge back into QuTiP land
# --------------------------------------------------------------------------- #

class TestToQobj:
    def test_returns_qobj_with_expected_dims_and_data(self, array_backend):
        ket = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        q = array_backend.to_qobj(ket, [2, 2])
        assert isinstance(q, qt.Qobj)
        assert q.dims == [[2, 2], [2, 2]]
        expected = np.outer(ket, ket.conj())
        assert np.allclose(q.full(), expected)

    def test_accepts_a_density_matrix(self, array_backend):
        q = array_backend.to_qobj(RHO0, [2])
        assert isinstance(q, qt.Qobj)
        assert np.allclose(q.full(), RHO0)


# --------------------------------------------------------------------------- #
# OqupyBackend.expect() -- the Dynamics-object quirk
# --------------------------------------------------------------------------- #

class _FakeDynamics:
    """Stand-in for an oqupy Dynamics object exposing `expectations`."""

    def __init__(self, times, values):
        self._times = np.asarray(times)
        self._values = np.asarray(values)
        self.calls = []

    def expectations(self, operator, real=False):
        self.calls.append({"operator": np.asarray(operator), "real": real})
        return self._times, self._values


class TestOqupyExpect:
    def test_reads_expectations_off_a_dynamics_object(self):
        be = OqupyBackend()
        values = np.array([0.1, 0.2, 0.3])
        dyn = _FakeDynamics(times=[0.0, 1.0, 2.0], values=values)
        out = be.expect(be.sigma("z"), dyn)
        # Current behaviour returns the *whole* time series, not just the last
        # point (despite the docstring mentioning "final-time"). Pin the
        # implemented behaviour so a future intentional change is deliberate.
        assert np.allclose(out, values)

    def test_requests_complex_valued_expectations(self):
        be = OqupyBackend()
        dyn = _FakeDynamics(times=[0.0], values=[1.0])
        be.expect(be.sigma("z"), dyn)
        assert dyn.calls[0]["real"] is False

    def test_falls_back_to_numpy_for_plain_states(self):
        be = OqupyBackend()
        z = be.sigma("z")
        assert be.expect(z, np.array([1, 0], dtype=complex)) == pytest.approx(1.0)
        assert be.expect(z, RHO1) == pytest.approx(-1.0)