import numpy as np
import qutip as qt


HUSIMI_EVAL_METHODS = ("avg", "ptrace", "diff")

def eval_husimi(solver, rho, theta, phi, sel=None, method="avg"):
    """Evaluate the Husimi Q-function for a state or reduced TLS state.

    Parameters
    ----------
    solver : Solver
        Solver instance used to solve for dynamics
    rho : Qobj or ndarray
        State or density matrix to evaluate.
    theta : array_like
        polst angle grid for the Husimi function.
    phi : array_like
        azimuthal angle grid for the Husimi function.
    sel : int or array_like or None
        TLS index used to partially trace out TLS state when method is 'ptrace'.
    method : {'avg', 'ptrace', 'diff'}
        Evaluation method for the Husimi Q-function.

    Returns
    -------
    ndarray
        Evaluated Husimi Q-function on the requested grid.
    """
    if method not in HUSIMI_EVAL_METHODS:
        raise ValueError("Error: Invalid husimi evaluation method")

    n_tls = solver.model.n_tls

    if not isinstance(rho, qt.Qobj):
        dims = [2 for _ in range(n_tls)]
        rho = qt.Qobj(rho, dims=dims)

    if rho.isket or rho.isbra:
        rho = qt.ket2dm(rho)

    if sel:
        sel = np.asarray(sel)

    j = 1/2 # spin of TLS
    prefactor = (2 * j + 1) / (4 * np.pi) # husimi prefactor
    match method:
        case "ptrace":
            if sel is None:
                raise ValueError("Error: Index for the partial trace is None")
            rho_partial = qt.ptrace(rho, sel)
            Q, _, _ = qt.spin_q_function(rho_partial, theta, phi)
            return prefactor * Q
        case "avg":
            Qs = []
            for i in range(n_tls):
                rho_partial = qt.ptrace(rho, i)
                Q, _, _ = qt.spin_q_function(rho_partial, theta, phi)
                Qs.append(Q)
            Q_res = np.mean(Qs, axis=0)
            return prefactor * Q_res
        case "diff":
            rho_1, rho_2 = qt.ptrace(rho, sel[0]), qt.ptrace(rho, sel[1])
            Q1, _, _ = qt.spin_q_function(rho_1, theta, phi)
            Q2, _, _ = qt.spin_q_function(rho_2, theta, phi)
            return prefactor * (Q1 - Q2)
        case _:
            raise ValueError("Error: Invalid Husimi-Q evaluation method.")


    def final_time_pearson(solver, states, window_size):
        """Compute the final rolling Pearson correlation from a state trajectory."""
        n_tls = solver.model.n_tls

        if n_tls < 2:
            raise ValueError("Correlation requires at least two TLSs")

        if len(states) < window_size:
            raise ValueError("window_size cannot exceed the number of stored states")

        exp_xs = []
        for i in range(n_tls):
            sx = solver.ops.sx[i]
            exp_x = solver.backend.expect(sx, states)
            exp_xs.append(np.real(exp_x))
        x = np.asarray(exp_xs[0])
        y = np.asarray(exp_xs[1])

        return np.corrcoef(x[-window_size:], y[-window_size:])[1, 0]

    def final_time_plv(solver, states, window_size):
        """Compute the final rolling Phase Locking Value from a state trajectory."""
        n_tls = solver.model.n_tls

        if n_tls < 2:
            raise ValueError("Correlation requires at least two TLSs")

        if len(states) < window_size:
            raise ValueError("Window size cannot exceed the number of stored states")

        exp_sms = []
        for i in range(n_tls):
            sm = solver.ops.sm[i]
            exp_sm = solver.backend.expect(sm, states)
            exp_sms.append(exp_sm)
        x = np.asarray(exp_sms[0])
        y = np.asarray(exp_sms[1])
        phi1, phi2 = np.angle(x), np.angle(y)
        phase_exp = np.exp(1j * (phi1 - phi2))

        return np.abs(np.mean(phase_exp[-window_size:]))

    def compute_pearson(x, y, window_size):
        """Compute a rolling Pearson correlation between two signals."""
        if len(x) != len(y):
            raise ValueError("Arrays must be of equal length.")
        C_t = np.zeros(len(x))
        for i in range(window_size, len(x)):
            C_t[i] = np.corrcoef(x[i-window_size:i], y[i-window_size:i])[1, 0]
        return C_t

    def compute_plv(x, y, window_size):
        """Compute a rolling phase locking value between two phase signals."""
        if len(x) != len(y):
            raise ValueError("Arrays must be of equal length.")
        phi1, phi2 = np.angle(x), np.angle(y)
        phase_exp = np.exp(1j * (phi1 - phi2))

        plv_t = np.zeros(len(x))
        for i in range(window_size, len(x)):
            plv_t[i] = np.abs(np.mean(phase_exp[i-window_size:i]))

        return plv_t

    def compute_entropy(states, n_tls, i, j):
        """Compute mutual information evolution between two TLS."""
        entropy_t = np.zeros(len(states))
        for idx, state in enumerate(states):
            if not isinstance(state, qt.Qobj):
                dims = [2 for _ in range(n_tls)]
                state = qt.Qobj(state, dims=dims)
            entropy_t[idx] = qt.entropy_mutual(state, i, j)
        return entropy_t

#TODO: these methods should probably also go to jupyter notebook, but undecided yet

    def _phase_sim_helper(states):
        """Compute instantaneous TLS phases from the system states."""
        phases = []

        for i in range(self.n_tls):

            if self.is_qutip_solver:
                e_ops = [self.sx[i], self.sy[i]]
                exp_x, exp_y = qt.expect(e_ops, states)
            else:
                t, exp_x = states.expectations(self.sx[i], real=True)
                t, exp_y = states.expectations(self.sy[i], real=True)

            phases.append(np.arctan2(np.real(exp_y), np.real(exp_x)))

        return phases, self.tlist

    def _cor_sim_helper(states, corr_name, window_size, overlap):
        """Compute correlation trajectories for the requested metric.

        Parameters
        ----------
        states : sequence
            Time-series states or dynamics objects.
        corr_name : str
            Correlation type: ``pearson``, ``plv``, ``connected``, or ``entropy``.
        window_size : int
            Sliding window size for time-dependent correlations.
        overlap : int
            Overlap between windows.

        Returns
        -------
        tuple
            A tuple containing a mapping of TLS pair labels to correlation arrays
            and the shared time axis ``tlist``.
        """

        if corr_name.lower() == "entropy":
            return self._entropy_evolution(states)

        corrs = {}
        exp_xs = [] # expectations in x

        for i in range(self.n_tls):
            if self.is_qutip_solver:
                exp_x = qt.expect(self.sx[i], states)
            else:
                t, exp_x = states.expectations(self.sx[i], real=True)
            exp_xs.append(exp_x)

        if corr_name.lower() == "connected":
            if self.is_qutip_solver:
                exp_xs_all = qt.expect(np.prod(self.sx), states)
            else:
                e_op = self.sx[0] # product of sigma Xs
                for i in range(1, self.n_tls):
                    e_op = np.matmul(e_op, self.sx[i])
                t, exp_xs_all = states.expectations(e_op, real=True)

        if corr_name.lower() == "plv":
            exp_sms = []
            for i in range(self.n_tls):
                if self.is_qutip_solver:
                    exp_sm = qt.expect(self.sm[i], states)
                else:
                    t, exp_sm = states.expectations(self.sm[i])
                exp_sms.append(exp_sm)

        for i in range(0, self.n_tls):
            for j in range(i+1, self.n_tls):
                match corr_name.lower():
                    case "plv":
                        corrs[f"TLS {self.omega_tls[i]}, {self.omega_tls[j]}"] = self._plv_evolution(exp_sms[i], exp_sms[j], window_size=window_size, overlap=overlap)
                    case "pearson":
                        corrs[f"TLS {self.omega_tls[i]}, {self.omega_tls[j]}"] = self._pearson_evolution(np.real(exp_xs[i]), np.real(exp_xs[j]), window_size=window_size, overlap=overlap)
                    case "connected":
                        if self.is_qutip_solver:
                            q_corr = (exp_xs_all - exp_xs[i] * exp_xs[j]) / np.sqrt((qt.variance(self.sx[i], states) * qt.variance(self.sx[j], states)))
                        else: # handle TEMPO
                            def var(op):
                                t, exp_op = states.expectations(op, real=True)
                                t, exp_op_sq = states.expectations(np.matmul(op, op), real=True)
                                return exp_op_sq - exp_op**2
                            q_corr = (exp_xs_all - exp_xs[i] * exp_xs[j]) / np.sqrt((var(self.sx[i]) * var(self.sx[j])))
                        corrs[f"TLS {self.omega_tls[i]}, {self.omega_tls[j]}"] = q_corr
                    case _:
                        raise ValueError("Error: Invalid correlation name.")

        return corrs, self.tlist


#TODO: move these to jupyter notebook

    def husimi_sim(omega_d, theta, phi, method, tls_idx=None):
        """Compute Husimi-Q functions for a single-frequency run."""
        states = self._state_sequence(self._get_states(omega_d))
        return parallel_eval_husimi(
            states,
            self.eval_husimi,
            theta,
            phi,
            method,
            tls_idx,
            desc=f"{self._name} Husimi-Q Computation",
        )

    def phase_sim(omega_d):
        """Compute TLS phase differences from solver states."""
        states = self._get_states(omega_d)
        return self._phase_sim_helper(states)

    def pearson_sim(omega_d, window_size, overlap):
        """Compute rolling Pearson correlations from solver states."""
        states = self._get_states(omega_d)
        return self._cor_sim_helper(states, "pearson", window_size, overlap)

    def plv_sim(omega_d, window_size, overlap):
        """Compute rolling phase locking values from solver states."""
        states = self._get_states(omega_d)
        return self._cor_sim_helper(states, "plv", window_size, overlap)

    def phase_corr_sim(omega_d, corr_names, window_size=None, overlap=None):
        """Compute phase differences and requested correlations from solver states."""
        states = self._get_states(omega_d)
        phases, t = self._phase_sim_helper(states)
        if isinstance(corr_names, list):
            corrs = []
            for corr_name in corr_names:
                corr, _ = self._cor_sim_helper(states, corr_name, window_size, overlap)
                corrs.append(corr)
            return phases, corrs, t

        corr, _ = self._cor_sim_helper(states, corr_names, window_size, overlap)
        return phases, corr, t
