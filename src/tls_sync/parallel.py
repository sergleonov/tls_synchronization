import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm


def _default_max_workers(max_workers):
    if max_workers is None:
        return max(1, multiprocessing.cpu_count() - 1)
    return max_workers


def run_parallel(omega_d_vals, worker, n_time, store_states, desc, state_postprocess=None, max_workers=None):
    """Run a collection of simulation workers in parallel with progress feedback."""
    exc = np.zeros((len(omega_d_vals), n_time))
    sp = np.zeros((len(omega_d_vals), n_time), dtype=complex)
    states = np.zeros((len(omega_d_vals), n_time), dtype=object) if store_states else None

    max_workers = _default_max_workers(max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if store_states:
            for idx, result in enumerate(tqdm(executor.map(worker, omega_d_vals),
                                              total=len(omega_d_vals),
                                              desc=desc)):
                exc_res, sp_res, states_res = result
                exc[idx, :] = exc_res
                sp[idx, :] = sp_res
                state_row = state_postprocess(states_res) if state_postprocess else states_res
                states[idx, :] = list(state_row)
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
    max_workers = _default_max_workers(max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for t_idx, Q in enumerate(tqdm(executor.map(eval_husimi_partial, states),
                                        total=len(states),
                                        desc=desc)):
            Qt[t_idx] = Q

    return Qt
