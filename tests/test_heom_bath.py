"""HEOM bath: coefficient extraction and worker-side reconstruction.

The whole point of ``_bath_to_coeffs`` / ``_rebuild_bath`` is to ship a bath
across a process boundary without pickling the unpicklable environment object.
Correctness means the rebuilt bath reproduces the original's correlation
function exactly (same exponents, just re-wrapped).
"""
import pickle

import numpy as np
import pytest

from tls_sync.heom import _rebuild_bath


def _correlation(bath, tlist):
    """Return C(t) for a qutip environment, tolerant of minor API differences."""
    if hasattr(bath, "correlation_function"):
        return np.asarray(bath.correlation_function(tlist))
    pytest.skip("bath object exposes no correlation_function(t) in this qutip version")


@pytest.mark.parametrize("fixture_name", ["heom_drude", "heom_power"])
def test_bath_coeffs_extract_finite_arrays(request, fixture_name):
    solver = request.getfixturevalue(fixture_name)
    bath = solver._build_bath()
    ck_r, vk_r, ck_i, vk_i, T = solver._bath_to_coeffs(bath)
    for arr in (ck_r, vk_r, ck_i, vk_i):
        assert arr.dtype == complex
        assert np.isfinite(arr).all()
    assert (T is None) or np.isfinite(T)


@pytest.mark.parametrize("fixture_name", ["heom_drude", "heom_power"])
def test_bath_coeffs_pickle_roundtrip(request, fixture_name):
    solver = request.getfixturevalue(fixture_name)
    coeffs = solver._bath_to_coeffs(solver._build_bath())
    restored = pickle.loads(pickle.dumps(coeffs))
    for a, b in zip(coeffs[:4], restored[:4]):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("fixture_name", ["heom_drude", "heom_power"])
def test_rebuilt_bath_matches_original_correlation(request, fixture_name):
    solver = request.getfixturevalue(fixture_name)
    bath = solver._build_bath()
    rebuilt = _rebuild_bath(solver._bath_to_coeffs(bath))

    t = np.asarray(solver.tlist)
    c_orig = _correlation(bath, t)
    c_new = _correlation(rebuilt, t)
    # Same exponents, re-wrapped -> correlation functions agree to precision.
    assert np.allclose(c_orig, c_new, atol=1e-8, rtol=1e-6)
