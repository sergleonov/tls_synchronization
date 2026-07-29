import numpy as np
import oqupy
import qutip as qt

SOLVERS = ["Markovian", "Tiered", "HEOM", "TEMPO"]
SD_TYPES = ["power", "drude"]
HUSIMI_EVAL_METHODS = ["avg", "ptrace", "diff"]

class Solver:
    """Base solver class for TLS dynamics and observables.

    This class provides common initialization, operator building, Hamiltonian
    construction, and correlation utilities shared by different solver backends.
    """

    def __init__(self,
                 tls_freqs=None, 
                 J=0.02, 
                 Omega_amp=0.1, 
                 lam=0.02, 
                 T=0.5, 
                 T_total=1600, 
                 T_drive=100.0, 
                 dt=0.5, 
                 n_tls=2,
                 n_freqs=300,
                 is_qutip_solver=None,
                 name="Abstract"):
        """Initialize common TLS solver parameters.

        Parameters
        ----------
        tls_freqs : array_like or None
            TLS eigenfrequencies. If None, random frequencies are generated.
        J : float
            TLS interaction strength.
        Omega_amp : float
            Drive amplitude.
        lam : float
            System-bath coupling strength.
        T : float
            Bath temperature.
        T_total : float
            Total simulation time.
        T_drive : float
            Drive duration.
        dt : float
            Time step.
        n_tls : int
            Number of two-level systems.
        n_freqs : int
            Number of drive frequency points.
        is_qutip_solver : bool or None
            Whether the solver uses QuTiP objects.
        name : str
            Solver name used for validation and string formatting.
        """

        if name not in SOLVERS:
            raise ValueError("Error: Invalid solver name")
        self._name = name
        
        self.J = J # interaction strength
        self.Omega_amp = Omega_amp # drive strength

        # bath parameters
        self.lam = lam # coupling strength
        self.T = T # temperature

        # time parameters
        self.T_total = T_total # ns
        self.T_drive = T_drive   # ns
        self.dt = dt # ns

        self.n_tls = n_tls # number of TLS in the system
        self.is_qutip_solver = is_qutip_solver

        # time list and drive frequencies
        self.tlist = np.arange(0, self.T_total+self.dt, self.dt)
        self.n_time = len(self.tlist)
        self.n_freqs = n_freqs
        self.omega_d_vals = np.linspace(3.0, 5.0, self.n_freqs)

        # tls frequencies
        if tls_freqs is not None:
            self.omega_tls = tls_freqs
        else:
            self.omega_tls = np.random.uniform(3.0, 5.0, self.n_tls) # GHz
        
        if self.n_tls != len(self.omega_tls):
            raise ValueError("Error: n_tls must equal the number of provided tls_freqs")

    def __getstate__(self):
        """Return the picklable state of the solver.

        The returned dictionary contains constructor parameters required to
        recreate the solver instance during unpickling.
        """
        d = {
            "tls_freqs":self.omega_tls, 
            "J":self.J, 
            "Omega_amp":self.Omega_amp, 
            "lam":self.lam, 
            "T":self.T, 
            "T_total":self.T_total, 
            "T_drive":self.T_drive, 
            "dt":self.dt, 
            "n_tls":self.n_tls,
            "n_freqs":self.n_freqs,
            "name":self._name
        }
        return d

    def __setstate__(self, d):
        """Reconstruct the solver from a saved state dictionary."""
        self.__init__(tls_freqs=d["tls_freqs"], 
                    J=d["J"], 
                    Omega_amp=d["Omega_amp"], 
                    lam=d["lam"], 
                    T=d["T"], 
                    T_total=d["T_total"], 
                    T_drive=d["T_drive"], 
                    dt=d["dt"], 
                    n_tls=d["n_tls"],
                    n_freqs=d["n_freqs"],
                    name=d["name"])

    def __str__(self):
        """Return a string summary of the solver configuration."""
        return(f"{self._name}_J_{self.J}_Omega_amp_{self.Omega_amp}_" + 
            f"lam_{self.lam}_T_{self.T}_" + 
            f"T_total_{self.T_total}_dt_{self.dt}_N_TLS_{self.n_tls}") 
    
    def _tensor(self, mats: list):
        """Return the tensor product of a list of matrices or QuTiP objects."""
        res = mats[0]
        if type(res) == qt.Qobj:
            return qt.tensor(mats)
        
        for i in range(1, len(mats)):
            res = np.kron(res, mats[i])
        return res
    
    def _build_c_ops(self):
        """Construct collapse operators for lindblad-based solvers."""
        # collapse ops with temperature dependence
        n_th = []
        for i in range(self.n_tls):
            n_th.append(1 / (np.exp(self.omega_tls[i] / self.T) - 1))
        self.c_ops = []
        for i in range(self.n_tls):
            self.c_ops.append(np.sqrt(self.lam * (n_th[i] + 1)) * sum(self.sm))
            self.c_ops.append(np.sqrt(self.lam * n_th[i]) * sum(self.sp))
        if self._name == "Tiered":
            self.a = self._tensor([qt.qeye(2), qt.qeye(2), qt.destroy(self.Nb)])
            n_th_mode = 1 / (np.exp(self.omega_c / self.T) - 1)
            self.c_ops.append(np.sqrt(self.lam * (n_th_mode + 1)) * (self.a))
            self.c_ops.append(np.sqrt(self.lam * (n_th_mode)) * (self.a.dag()))

    def build_operators(self):
        """Build TLS operators and system observables.

        The method constructs Pauli operators for each TLS and sets up the
        collective excitation and spin operators used by solver backends.
        """
        assert self._name in SOLVERS, "Error: Invalid solver name"

        sx_tls = []
        sy_tls = []
        sz_tls = []
        sm_tls = []
        sp_tls = []

        for i in range(self.n_tls):
            op_list = [qt.qeye(2) if self.is_qutip_solver else np.eye(2) for _ in range(self.n_tls)]
            op_list[i] = qt.sigmax() if self.is_qutip_solver else oqupy.operators.sigma("x")
            sx_tls.append(self._tensor(op_list))

            op_list[i] = qt.sigmay() if self.is_qutip_solver else oqupy.operators.sigma("y")
            sy_tls.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmaz() if self.is_qutip_solver else oqupy.operators.sigma("z")
            sz_tls.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmam() if self.is_qutip_solver else oqupy.operators.sigma("-")
            sm_tls.append(self._tensor(op_list))
            
            op_list[i] = qt.sigmap() if self.is_qutip_solver else oqupy.operators.sigma("+")
            sp_tls.append(self._tensor(op_list))

        match self._name:
            case "Markovian":
                self.sx = sx_tls
                self.sy = sy_tls
                self.sz = sz_tls
                self.sp = sp_tls
                self.sm = sm_tls
                # collapse operators
                self._build_c_ops()

            case "Tiered":
                self.sx = []
                self.sy = []
                self.sz = []
                self.sp = []
                self.sm = []
                I_cav = qt.qeye(self.Nb) 
                for i in range(self.n_tls):
                    op_list = [sx_tls[i], I_cav]
                    self.sx.append(self._tensor(op_list))

                    op_list = [sy_tls[i], I_cav]
                    self.sy.append(self._tensor(op_list))

                    op_list = [sz_tls[i], I_cav]
                    self.sz.append(self._tensor(op_list))

                    op_list = [sp_tls[i], I_cav]
                    self.sp.append(self._tensor(op_list))

                    op_list = [sm_tls[i], I_cav]
                    self.sm.append(self._tensor(op_list))
                # annihilator and collapse operators
                self._build_c_ops()
        
            case _:
                self.sx = sx_tls
                self.sy = sy_tls
                self.sz = sz_tls
                self.sp = sp_tls
                self.sm = sm_tls
        
        # observables
        self.collective_sp = sum(self.sp)
        self.collective_sm = sum(self.sm)
        if self.is_qutip_solver:
            self.collective_exc = self.collective_sp * self.collective_sm
        else:
            self.collective_exc = np.matmul(self.collective_sp, self.collective_sm)

    def build_hamiltonian(self):
        """Construct the static system Hamiltonian."""
        if self._name not in SOLVERS:
            raise ValueError("Error: Invalid solver name")

        self.H = sum(0.5 * self.omega_tls[i] * self.sz[i] for i in range(self.n_tls))
        for i in range(self.n_tls):
            for j in range(i+1, self.n_tls):
                if self.is_qutip_solver:
                    self.H += self.J * self.sz[i] * self.sz[j]
                else:
                    self.H += self.J * np.matmul(self.sz[i], self.sz[j])

        if self._name == "Tiered":
            self.H += self.omega_c * self.a.dag() * self.a # cavity hamiltonian
            self.H += self.g * sum(self.sx) * (self.a.dag() + self.a) # system-bath hamiltonian
    
    def drive_coeff(self, t, args):
        """Return the time-dependent drive coefficient for the Hamiltonian."""
        if 0.0 <= t <= self.T_drive:
            return 0.5 * self.Omega_amp * np.cos(args["omega"] * t)
        else:
            return 0.0
        
    def eval_husimi(self, rho, theta, phi, tls_idx=None, method="avg"):
        """Evaluate the Husimi Q-function for a state or reduced TLS state.

        Parameters
        ----------
        rho : Qobj or ndarray
            State or density matrix to evaluate.
        theta : array_like
            Azimuthal angle grid for the Husimi function.
        phi : array_like
            Polar angle grid for the Husimi function.
        tls_idx : int or None
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

        if self._name == "TEMPO":
            dims = [2 for _ in range(self.n_tls)]
            rho = qt.Qobj(rho, dims=[dims, dims])

        if rho.isket:
            rho = qt.ket2dm(rho)
        
        j = 1/2 # spin of TLS
        prefactor = (2 * j + 1) / (4 * np.pi) # husimi prefactor
        match method:
            case "ptrace":
                if tls_idx is None:
                    raise ValueError("Error: Index for the partial trace is None")
                rho_partial = qt.ptrace(rho, tls_idx)
                Q, _, _ = qt.spin_q_function(rho_partial, theta, phi)
                return prefactor * np.transpose(Q)
            case "avg":
                Qs = []
                for i in range(self.n_tls):
                    rho_partial = qt.ptrace(rho, i)
                    Q, _, _ = qt.spin_q_function(rho_partial, theta, phi)
                    Qs.append(Q)
                Q_res = np.mean(Qs, axis=0)
                return prefactor * np.transpose(Q_res)
            case "diff":
                if self.n_tls != 2: raise ValueError("Error: Husimi Difference is only supported for 2 TLSs.")
                rho_1, rho_2 = qt.ptrace(rho, 0), qt.ptrace(rho, 1)
                Q1, _, _ = qt.spin_q_function(rho_1, theta, phi)
                Q2, _, _ = qt.spin_q_function(rho_2, theta, phi)
                return prefactor * np.transpose(Q1 - Q2)
            case _:
                raise ValueError("Error: Invalid Husimi-Q evaluation method.")
        
    def _pearson_evolution(self, x, y, window_size, overlap=1):
        """Compute a rolling Pearson correlation between two signals."""
        step = window_size - overlap
        if step <= 0:
            raise ValueError("Error: overlap must be smaller than window_size")
        # TODO: add overlap control
        C_t = np.zeros(self.n_time)
        for i in range(window_size, self.n_time):
            C_t[i] = np.corrcoef(x[i-window_size:i], y[i-window_size:i])[1, 0]
        return C_t

    def final_corr_from_states(self, states, corr_name, window_size):
        """Compute the final rolling Pearson correlation from a state trajectory."""

        if self.n_tls < 2:
            raise ValueError("Correlation requires at least two TLSs")

        if corr_name.lower() == "plv":
            exp_sms = []
            for i in range(self.n_tls):
                if self.is_qutip_solver:
                    exp_sm = qt.expect(self.sm[i], states)
                else: 
                    t, exp_sm = states.expectations(self.sm[i])
                exp_sms.append(exp_sm)
            x = np.asarray(exp_sms[0])
            y = np.asarray(exp_sms[1])
            phi1, phi2 = np.angle(x), np.angle(y)
            phase_exp = np.exp(1j * (phi1 - phi2))

            if len(x) < window_size:
                raise ValueError("window_size cannot exceed the number of stored states")
    
            return np.abs(np.mean(phase_exp[-window_size:])) 

        if corr_name.lower() == "pearson":
            exp_xs = []
            for i in range(self.n_tls):
                if self.is_qutip_solver:
                    exp_x = qt.expect(self.sx[i], states)
                else:
                    t, exp_x = states.expectations(self.sx[i])
                exp_xs.append(np.real(exp_x))
            x = np.asarray(exp_xs[0])
            y = np.asarray(exp_xs[1])

            if len(x) < window_size:
                raise ValueError("window_size cannot exceed the number of stored states")

            return np.corrcoef(x[-window_size:], y[-window_size:])[1, 0]

        raise ValueError("Error: Invalid correlation name")

    def _plv_evolution(self, x, y, window_size, overlap=1):
        """Compute a rolling phase locking value between two phase signals."""
        step = window_size - overlap
        if step <= 0:
            raise ValueError("Error: overlap must be smaller than window_size")
        # TODO: add overlap control
        phi1, phi2 = np.angle(x), np.angle(y)
        phase_exp = np.exp(1j * (phi1 - phi2))

        plv_t = np.zeros(self.n_time)
        for i in range(window_size, self.n_time):
            plv_t[i] = np.abs(np.mean(phase_exp[i-window_size:i])) 

        return plv_t

    def _entropy_evolution(self, states):
        """Compute mutual information evolution between TLS pairs.

        Parameters
        ----------
        states : sequence
            Time series of system states or dynamics objects.

        Returns
        -------
        dict
            Mapping of TLS pair labels to entropy trajectories.
        """
        if len(states) != self.n_time:
            raise ValueError("Error: states length must equal n_time")

        if self._name == "TEMPO":
            states = states.states

        res_dict = {}
        for i in range(self.n_tls):
            for j in range(i+1, self.n_tls):
                entropy_t = np.zeros(self.n_time)
                for idx, state in enumerate(states):
                    if self._name == "Tiered":
                        # trace out environment
                        state = qt.ptrace(state, [i for i in range(self.n_tls)]) 
                        entropy_t[idx] = qt.entropy_mutual(state, i, j)
                    elif self._name == "TEMPO":
                        dims = [2 for _ in range(self.n_tls)]
                        rho = qt.Qobj(state, dims=[dims, dims])
                        entropy_t[idx] = qt.entropy_mutual(rho, i, j)
                    else:
                        entropy_t[idx] = qt.entropy_mutual(state, i, j)
                res_dict[f"TLS {self.omega_tls[i]}, {self.omega_tls[j]}"] = entropy_t

        return res_dict, self.tlist      
    
    def _phase_sim_helper(self, states):
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
    
    def _cor_sim_helper(self, states, corr_name, window_size, overlap):
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

    def _get_states(self, omega_d):
        """Return stored states or dynamics from a single-frequency run."""
        raise NotImplementedError("Solver subclasses must implement _get_states()")

    def phase_sim(self, omega_d):
        """Compute TLS phase differences from solver states."""
        states = self._get_states(omega_d)
        return self._phase_sim_helper(states)

    def pearson_sim(self, omega_d, window_size, overlap):
        """Compute rolling Pearson correlations from solver states."""
        states = self._get_states(omega_d)
        return self._cor_sim_helper(states, "pearson", window_size, overlap)

    def plv_sim(self, omega_d, window_size, overlap):
        """Compute rolling phase locking values from solver states."""
        states = self._get_states(omega_d)
        return self._cor_sim_helper(states, "plv", window_size, overlap)

    def phase_corr_sim(self, omega_d, corr_names, window_size=None, overlap=None):
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