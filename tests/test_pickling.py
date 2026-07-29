"""Pickling / serialization round-trips.

Every solver is pickled per task by ProcessPoolExecutor during a sweep, so
these are the most important consistency guards. We check that:
  * pickle.dumps/loads reconstructs an equivalent solver (__getstate__ matches),
  * the reconstruction rebuilt its operators/Hamiltonian,
  * the payloads that actually cross the process boundary are picklable
    (HEOM bath coefficients; TEMPO process tensor).
"""
import pickle
from functools import partial

import numpy as np
import pytest

from _helpers import assert_params_equal, to_dense


def test_getstate_is_picklable_dict(any_solver):
    state = any_solver.__getstate__()
    assert isinstance(state, dict)
    pickle.loads(pickle.dumps(state))   # no exception


def test_solver_roundtrip_preserves_params(any_solver):
    s = any_solver
    s2 = pickle.loads(pickle.dumps(s))
    assert type(s2) is type(s)
    assert_params_equal(s.__getstate__(), s2.__getstate__())


def test_solver_roundtrip_rebuilds_operators(any_solver):
    s = any_solver
    s2 = pickle.loads(pickle.dumps(s))
    assert s2.H is not None
    assert to_dense(s2.H).shape == to_dense(s.H).shape
    assert len(s2.sx) == len(s.sx)
    # drive behaviour is identical after a round-trip
    args = {"omega": 4.0}
    assert np.isclose(s.drive_coeff(1.0, args), s2.drive_coeff(1.0, args))


def test_heom_bath_coefficients_are_picklable(heom_drude):
    bath = heom_drude._build_bath()
    coeffs = heom_drude._bath_to_coeffs(bath)
    restored = pickle.loads(pickle.dumps(coeffs))
    for a, b in zip(coeffs[:4], restored[:4]):
        assert np.array_equal(a, b)
    assert coeffs[4] == restored[4]   # temperature


def test_heom_worker_partial_is_picklable(heom_drude):
    # This is exactly what run() hands to the pool.
    bath = heom_drude._build_bath()
    coeffs = heom_drude._bath_to_coeffs(bath)
    worker = partial(heom_drude._worker, bath_coeffs=coeffs, store_states=False)
    pickle.loads(pickle.dumps(worker))   # must not raise


@pytest.mark.slow
def test_tempo_process_tensor_worker_is_picklable(tempo):
    import oqupy
    pt = oqupy.pt_tempo_compute(
        bath=tempo.bath, start_time=0.0, end_time=tempo.T_total,
        parameters=tempo.tempo_params,
    )
    worker = partial(tempo._worker, process_tensor=pt, store_states=False)
    pickle.loads(pickle.dumps(worker))   # process tensor must survive pickling
