from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Sequence

import qutip as qt
import oqupy as oq
import numpy as np


class Backend(ABC):
    """Operations the rest of the package needs, in one consistent representation."""

    name: str

    # --- operator construction (used by Model.build_*) ---------------------- #
    @abstractmethod
    def sigma(self, axis: str) -> Any:
        """Single-TLS Pauli/ladder operator for axis in {'x','y','z','+','-'}."""

    @abstractmethod
    def identity(self, dim: int) -> Any:
        """Identity operator of dimension `dim`."""

    @abstractmethod
    def destroy(self, dim: int) -> Any:
        """Bosonic annihilation operator on a `dim`-level mode (cavity Fock)."""

    @abstractmethod
    def tensor(self, ops: Sequence[Any]) -> Any:
        """Tensor product over the listed single-subsystem operators."""

    @abstractmethod
    def mul(self, a: Any, b: Any) -> Any:
        """Operator (matrix) product a·b. (numpy `*` is elementwise, hence this.)"""

    @abstractmethod
    def dag(self, op: Any) -> Any:
        """Hermitian conjugate."""

    # --- analysis primitives (used by Dynamics / correlations) -------------- #
    @abstractmethod
    def expect(self, op: Any, state: Any) -> complex:
        """Expectation value <op> in a single state (ket or density matrix)."""

    @abstractmethod
    def ptrace(self, state: Any, keep: Sequence[int],
               dims: Sequence[int] | None = None) -> Any:
        """Partial trace keeping the listed subsystem indices.

        `dims` gives the subsystem dimensions (subsystem 0 most significant),
        ordered as the tensor/kron factors. It is optional for backends whose
        state already carries its tensor structure (QuTiP `Qobj`) and required
        for backends whose dense arrays do not (numpy, oqupy)."""

    @abstractmethod
    def to_density_matrix(self, state: Any) -> Any:
        """Coerce a ket / raw array into a density matrix in this representation."""

    # --- concrete helper built on the primitives above --------------------- #
    def commutator(self, a: Any, b: Any) -> Any:
        """[a, b] = a·b - b·a, expressed via `mul`."""
        return self.mul(a, b) - self.mul(b, a)


# Shared convention for the ladder operators so every backend agrees:
#   sigma('+') = |0><1| = [[0, 1], [0, 0]]   (matches qt.sigmap)
#   sigma('-') = |1><0| = [[0, 0], [1, 0]]   (matches qt.sigmam)
_LADDER_AXES = {"x", "y", "z", "+", "-"}


class QutipBackend(Backend):
    """QuTiP `Qobj` representation (sigmax(), qt.expect, qt.ptrace, ...)."""

    name = "qutip"

    _SIGMA = {
        "x": qt.sigmax,
        "y": qt.sigmay,
        "z": qt.sigmaz,
        "+": qt.sigmap,
        "-": qt.sigmam,
    }

    def sigma(self, axis: str) -> Any:
        try:
            return self._SIGMA[axis]()
        except KeyError:
            raise ValueError(
                f"unknown Pauli/ladder axis {axis!r}; expected one of {sorted(_LADDER_AXES)}"
            )

    def identity(self, dim: int) -> Any:
        return qt.qeye(dim)

    def destroy(self, dim: int) -> Any:
        return qt.destroy(dim)

    def tensor(self, ops: Sequence[Any]) -> Any:
        ops = list(ops)
        if not ops:
            raise ValueError("tensor() needs at least one operator")
        return qt.tensor(ops)

    def mul(self, a: Any, b: Any) -> Any:
        # Qobj.__mul__ is the operator (matrix) product, not elementwise.
        return a * b

    def dag(self, op: Any) -> Any:
        return op.dag()

    def expect(self, op: Any, state: Any) -> complex:
        # qt.expect returns a real float for Hermitian ops; complex() is safe either way.
        return complex(qt.expect(op, state))

    def ptrace(self, state: Any, keep: Sequence[int],
               dims: Sequence[int] | None = None) -> Any:
        # Qobj.ptrace(sel) keeps the selected subsystem indices. The Qobj
        # already carries its tensor structure, so `dims` is accepted for
        # signature uniformity but not required.
        return state.ptrace(list(keep))

    def to_density_matrix(self, state: Any) -> Any:
        if state.isoper:
            return state
        if state.isbra:
            state = state.dag()
        return qt.ket2dm(state)


class NumpyBackend(Backend):
    """Dense numpy representation (kron, einsum, matmul); the semiclassical dialect."""

    name = "numpy"

    _PAULI = {
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "z": np.array([[1, 0], [0, -1]], dtype=complex),
        "+": np.array([[0, 1], [0, 0]], dtype=complex),  # |0><1|
        "-": np.array([[0, 0], [1, 0]], dtype=complex),  # |1><0|
    }

    def sigma(self, axis: str) -> Any:
        try:
            return self._PAULI[axis].copy()  # copy: callers must not mutate the table
        except KeyError:
            raise ValueError(
                f"unknown Pauli/ladder axis {axis!r}; expected one of {sorted(_LADDER_AXES)}"
            )

    def identity(self, dim: int) -> Any:
        return np.eye(dim, dtype=complex)

    def destroy(self, dim: int) -> Any:
        # sqrt(1..dim-1) on the first superdiagonal
        return np.diag(np.sqrt(np.arange(1, dim, dtype=float)), k=1).astype(complex)

    def tensor(self, ops: Sequence[Any]) -> Any:
        ops = [np.asarray(o) for o in ops]
        if not ops:
            raise ValueError("tensor() needs at least one operator")
        return reduce(np.kron, ops)

    def mul(self, a: Any, b: Any) -> Any:
        # `@` is matmul; numpy `*` would be elementwise (the trap the ABC warns about).
        return np.asarray(a) @ np.asarray(b)

    def dag(self, op: Any) -> Any:
        return np.asarray(op).conj().T

    def expect(self, op: Any, state: Any) -> Any:
        op = np.asarray(op)
        if isinstance(state, (list, tuple)):
            states = [np.asarray(s) for s in state]
        elif np.ndim(state) >= 3:            # stack of density matrices
            states = list(np.asarray(state))
        else:                                # one ket or density matrix
            return self._expect_one(op, state)
        return np.array([self._expect_one(op, s) for s in states], dtype=complex)

    @staticmethod
    def _expect_one(op: Any, state: Any) -> complex:
        op = np.asarray(op)
        state = np.asarray(state)
        if state.ndim == 1 or (state.ndim == 2 and 1 in state.shape):
            psi = state.reshape(-1)
            return complex(np.vdot(psi, op @ psi))  # <psi|op|psi>
        return complex(np.trace(op @ state))  # Tr(op rho)

    def to_density_matrix(self, state: Any) -> Any:
        state = np.asarray(state, dtype=complex)
        if state.ndim == 1:
            return np.outer(state, state.conj())
        if state.ndim == 2 and 1 in state.shape:
            psi = state.reshape(-1, 1)
            return psi @ psi.conj().T
        return state  # already a (square) density matrix

    def to_qobj(self, state: Any, dims: Sequence[int]) -> Any:
        rho = self.to_density_matrix(state)
        return qt.Qobj(rho, dims=[dims, dims])

    def ptrace(self, state: Any, keep: Sequence[int],
               dims: Sequence[int] | None = None) -> np.ndarray:
        """General dense partial trace via einsum.

        `dims` orders subsystems as the kron factors (subsystem 0 most
        significant) and is required here: a dense matrix carries no tensor
        structure to infer them from."""
        if dims is None:
            raise ValueError(
                f"{type(self).__name__}.ptrace requires `dims` (the subsystem "
                "dimensions); a dense array carries no tensor structure to infer them."
            )
        n = len(dims)
        keep = sorted(keep)
        rho = np.asarray(state).reshape(list(dims) + list(dims))
        # row labels 0..n-1; column labels reuse the row label for traced
        # subsystems (→ summed) and take a fresh label for kept ones.
        row = list(range(n))
        col = [(n + i) if i in keep else i for i in range(n)]
        out = [i for i in keep] + [n + i for i in keep]
        result = np.einsum(rho, row + col, out)
        d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
        return result.reshape(d_keep, d_keep)


class OqupyBackend(NumpyBackend):
    """Adapter for oqupy arrays / Dynamics-object expectations (TEMPO).

    oqupy operators *are* dense numpy arrays, so construction and the linear
    algebra reduce to the numpy backend; only `expect` carries a genuine quirk.
    """

    name = "oqupy"

    def sigma(self, axis: str) -> Any:
        if axis not in _LADDER_AXES:
            raise ValueError(
                f"unknown Pauli/ladder axis {axis!r}; expected one of {sorted(_LADDER_AXES)}"
            )
        return np.asarray(oq.operators.sigma(axis), dtype=complex)

    def identity(self, dim: int) -> Any:
        return np.asarray(oq.operators.identity(dim), dtype=complex)

    def destroy(self, dim: int) -> Any:
        return np.asarray(oq.operators.destroy(dim), dtype=complex)

    def expect(self, op: Any, state: Any) -> complex:
        # The oqupy subtelty: expectation values are read off a Dynamics-like
        # object, not computed from a bare state. When handed one, ask it
        # directly and return the full expectation time series it yields;
        # otherwise treat `state` as a plain density matrix and defer to the
        # numpy implementation.
        if hasattr(state, "expectations"):
            _times, values = state.expectations(np.asarray(op), real=False)
            return values
        return super().expect(op, np.asarray(state))