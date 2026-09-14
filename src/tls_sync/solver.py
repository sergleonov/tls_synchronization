import numpy as np
from abc import ABC, abstractmethod
from tls_sync.backend import Backend
from tls_sync.helpers import Dynamics
from tls_sync.model import Model

import multiprocessing
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

class Solver(ABC):
    """What a solver *is*: build the model in a representation, then integrate.
 
    ``__init__`` builds the observation grid and, by uniform delegation to the
    Model, the operators, static Hamiltonian, and initial state -- so every
    solver gets those for free. The genuinely per-solver primitives are
    ``_prepare`` (frequency-independent setup) and ``_run_one`` (integrate one
    drive frequency); the base provides ``single_run`` and the parallel
    ``sweep`` on top of them.
 
    Subclasses that support a structured bath declare `SUPPORTED_SD` and call
    `_require_bath()` in their bath-rendering step; the default accepts any
    known family (a Markovian solver reads only coupling + temperature).
    """
 
    #: spectral-density families this solver can represent (see model.SD_TYPES)
    SUPPORTED_SD: tuple[str, ...] = ()
 
    def __init__(self, model: Model, backend: Backend, *,
                 T_total: float, dt: float) -> None:
        self.model = model
        self.backend = backend
        self.T_total = T_total
        self.dt = dt

        self.times = np.arange(0.0, self.T_total + 0.5 * self.dt, self.dt)
 
        # common construction: uniform delegation to the model, so every solver
        # gets operators, static Hamiltonian, and initial state for free.
        self.ops = self.model.build_operators(self.backend)      
        self.H = self.model.build_hamiltonian(self.backend, self.ops)
        self.rho0 = self.model.initial_state(self.backend, self.ops)
 
    def _require_bath(self):
        """Return the model's bath, or raise if it is missing / unsupported.
 
        Solvers that must have a structured bath (HEOM, TEMPO) call this in
        their bath-rendering step; a permissive solver (Markovian) can ignore it.
        """
        bath = self.model.bath
        if bath is None:
            raise ValueError(f"{type(self).__name__} requires a bath on the model.")
        if self.SUPPORTED_SD and bath.sd_type not in self.SUPPORTED_SD:
            raise ValueError(
                f"{type(self).__name__} supports {self.SUPPORTED_SD} spectral "
                f"densities, but the model's bath is {bath.sd_type!r}.")
        return bath

    @abstractmethod
    def _prepare(self):
        """Per-run setup shared across all drive frequencies (abstract).
 
        A solver with an expensive, frequency-independent object -- collapse
        operators (Markovian), bath coefficients (HEOM), a process tensor
        (TEMPO) -- builds it here once; the result is passed to every
        `_run_one` call. Every solver must implement it.
        """
 
    @abstractmethod
    def _run_one(self, omega_d, prepared, e_ops, store_states) -> tuple[list[np.ndarray], list[Any] | None]:
        """Integrate ONE drive frequency (the per-solver primitive).
 
        Returns ``(expect, states)`` where ``expect[j]`` is the time series for
        ``e_ops[j]`` and ``states`` is the per-time trajectory (or None). This
        runs inside a worker process during a sweep, so it must rebuild any
        unpicklable objects (e.g. drive-coefficient closures) itself.
        """
 
    def _default_e_ops(self):
        """Observables used when a run is given no `e_ops` (collective exc, S+)."""
        return [self.ops.collective_exc, self.ops.collective_sp]
 
    def _collect(self, omegas, results, store_states) -> Dynamics:
        """Assemble per-frequency results into one Dynamics.
 
        With ``store_states`` each result is ``(expect, states)``; otherwise it is
        just ``expect`` (a list of per-operator time series). Expectations end up
        as a single ``(n_ops, n_omega, n_time)`` array, so ``expectations[j]`` is
        the ``(n_omega, n_time)`` heatmap grid for the j-th operator.
        """
        if store_states:
            expects, states = zip(*results)
            states = list(states)
        else:
            expects, states = results, None
 
        grid = np.array(expects)                          # (n_omega, n_ops, n_time)
        expectations = grid.transpose(1, 0, 2) if grid.ndim == 3 else None
        return Dynamics(backend=self.backend, times=self.times,
                        omegas=np.asarray(omegas, dtype=float),
                        expectations=expectations, states=states, extra=None)
 
    def single_run(self, omega_d, *, e_ops=None, store_states=True) -> Dynamics:
        """Run a single drive frequency, returning states and expectations."""
        e_ops = list(e_ops) if e_ops is not None else self._default_e_ops()
        result = self._run_one(omega_d, self._prepare(), e_ops, store_states)
        return self._collect([omega_d], [result], store_states)
 
    def sweep(self, omega_d_vals, *, e_ops=None, max_workers=None) -> Dynamics:
        """Run the drive-frequency sweep in parallel, storing expectations only.
 
        Each frequency is an independent `_run_one` dispatched to a worker
        process. States are not stored (use `single_run` for a full trajectory).
        A solver that integrates the whole sweep more efficiently as one batch
        may override this.
        """
        omega_d_vals = np.asarray(omega_d_vals, dtype=float)
        e_ops = list(e_ops) if e_ops is not None else self._default_e_ops()
        worker = partial(self._run_one, prepared=self._prepare(),
                         e_ops=e_ops, store_states=False)
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(worker, omega_d_vals),
                                total=len(omega_d_vals),
                                desc=f"{type(self).__name__} sweep"))
        return self._collect(omega_d_vals, results, store_states=False)