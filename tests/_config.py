"""Shared configuration constants for the test suite."""

# Fixed, non-random TLS frequencies -> deterministic, reproducible tests.
TLS_FREQS = [3.5, 4.0]

# Tiny but non-degenerate grid:
#   tlist = [0,1,2,3,4,5,6]  (n_time = 7)
#   omega_d_vals = [3.0, 5.0] (n_freqs = 2)
#   drive on for the first half.
BASE = dict(
    tls_freqs=TLS_FREQS,
    J=0.02,
    Omega_amp=0.1,
    lam=0.02,
    T=0.5,
    T_total=6.0,
    T_drive=3.0,
    dt=1.0,
    n_tls=2)

EXPECTED_N_TIME = 7
EXPECTED_N_FREQS = 2
