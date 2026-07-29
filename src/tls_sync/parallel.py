import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

def run_parallel(omega_d_vals, worker, n_time, store_states, desc, max_workers=None):
    """Run a collection of simulation workers in parallel with progress feedback."""
    exc = np.zeros((len(omega_d_vals), n_time))
    sp = np.zeros((len(omega_d_vals), n_time), dtype=complex)
    states = np.zeros((len(omega_d_vals), n_time), dtype=object) if store_states else None

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if store_states:
            for idx, result in enumerate(tqdm(executor.map(worker, omega_d_vals),
                                              total=len(omega_d_vals),
                                              desc=desc)):
                exc_res, sp_res, states_res = result
                exc[idx, :] = exc_res
                sp[idx, :] = sp_res
                states_slice = states_res.states if hasattr(states_res, "states") else states_res
                states[idx, :] = list(states_slice)
            return exc, sp, states

        for idx, (exc_res, sp_res) in enumerate(tqdm(executor.map(worker, omega_d_vals),
                                                       total=len(omega_d_vals),
                                                       desc=desc)):
            exc[idx, :] = exc_res
            sp[idx, :] = sp_res

    return exc, sp


def parallel_eval_husimi(states, eval_husimi, theta, phi, method, tls_idx, desc, max_workers=None):
    """Evaluate Husimi-Q for each state in parallel."""
    Qt = np.zeros((len(states), len(theta), len(phi)))
    eval_husimi_partial = partial(eval_husimi,
                                  theta=theta,
                                  phi=phi,
                                  method=method,
                                  tls_idx=tls_idx)
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for t_idx, Q in enumerate(tqdm(executor.map(eval_husimi_partial, states),
                                        total=len(states),
                                        desc=desc)):
            Qt[t_idx] = Q

    return Qt