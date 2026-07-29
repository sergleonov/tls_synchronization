"""Operator building, Hamiltonian, drive coefficient, initial state, dissipators."""
import numpy as np
import pytest

from _config import BASE, EXPECTED_N_TIME, EXPECTED_N_FREQS
from _helpers import is_hermitian, is_density_matrix, to_dense


def test_time_and_frequency_grids(any_solver):
    s = any_solver
    assert s.n_time == EXPECTED_N_TIME
    assert len(s.tlist) == EXPECTED_N_TIME
    assert len(s.omega_d_vals) == EXPECTED_N_FREQS
    assert s.tlist[0] == 0.0
    assert np.isclose(s.tlist[-1], BASE["T_total"])


def test_operators_and_hamiltonian_built(any_solver):
    s = any_solver
    assert len(s.sx) == s.n_tls
    assert len(s.sy) == s.n_tls
    assert len(s.sz) == s.n_tls
    assert s.H is not None
    assert s.collective_sp is not None
    assert s.collective_sm is not None
    assert s.collective_exc is not None


def test_hamiltonian_is_hermitian(any_solver):
    assert is_hermitian(any_solver.H)


def test_initial_state_is_valid_density_matrix(any_solver):
    assert is_density_matrix(any_solver.rho0)


def test_operator_dimensions(any_solver):
    s = any_solver
    dim = 2 ** s.n_tls
    if s._name == "Tiered":
        dim *= s.Nb            # TLS space tensored with the cavity mode
    m = to_dense(s.sx[0])
    assert m.shape == (dim, dim)
    assert to_dense(s.H).shape == (dim, dim)


def test_tempo_initial_state_dimension():
    # regression guard: psi0 must live in 2**n_tls, not 2*n_tls
    from tls_sync import TEMPO
    for n_tls, freqs in [(2, [3.5, 4.0]), (3, [3.4, 3.7, 4.1])]:
        kw = dict(BASE)
        kw.update(n_tls=n_tls, tls_freqs=freqs)
        s = TEMPO(**kw, gamma_bath=0.5, tcut=1.0, epsrel=1e-3)
        assert s.rho0.shape == (2 ** n_tls, 2 ** n_tls)
        assert np.isclose(np.trace(s.rho0).real, 1.0)


@pytest.mark.parametrize("omega", [3.0, 4.2])
def test_drive_coeff_on_and_off(any_solver, omega):
    s = any_solver
    args = {"omega": omega}
    # within the drive window
    t = 1.0
    expected = 0.5 * s.Omega_amp * np.cos(omega * t)
    assert np.isclose(s.drive_coeff(t, args), expected)
    # after the drive window -> off
    assert s.drive_coeff(s.T_drive + 1.0, args) == 0.0
    # at t = 0
    assert np.isclose(s.drive_coeff(0.0, args), 0.5 * s.Omega_amp)


def test_lindblad_collapse_operator_count(lindblad):
    # two operators (up/down) per TLS
    assert len(lindblad.c_ops) == 2 * lindblad.n_tls


def test_tiered_collapse_operator_count(tiered):
    # two per TLS plus two for the cavity mode
    assert len(tiered.c_ops) == 2 * tiered.n_tls + 2


def test_random_tls_freqs_when_unspecified():
    from tls_sync import Lindblad
    kw = dict(BASE)
    kw.pop("tls_freqs")
    s = Lindblad(tls_freqs=None, **kw)
    assert len(s.omega_tls) == s.n_tls
    assert np.all((s.omega_tls >= 3.0) & (s.omega_tls <= 5.0))


def test_mismatched_tls_freqs_raises():
    from tls_sync import Lindblad
    kw = dict(BASE)
    kw.update(tls_freqs=[3.5, 4.0, 4.5])   # 3 freqs but n_tls stays 2
    with pytest.raises(ValueError):
        Lindblad(**kw)
