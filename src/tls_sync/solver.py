import numpy as np
import oqupy
import qutip as qt
from abc import ABC, abstractmethod
from functools import partial

from .parallel import run_parallel, parallel_eval_husimi

SOLVERS = ["Markovian", "Tiered", "HEOM", "TEMPO"]
SD_TYPES = ["power", "drude"]
HUSIMI_EVAL_METHODS = ["avg", "ptrace", "diff"]

class Solver(ABC):
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
            "name":self._name
        }
        return d

    def __setstate__(self, d):
        """Reconstruct the solver from a saved state dictionary."""
        raise NotImplementedError("Reconstruction from pickled state is not implemented.")

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

    # ------------------------------------------------------------------ #
    # Model-definition hooks. Override these in a subclass to change the  #
    # physical model (extra modes, dissipators, Hamiltonian terms)        #
    # without editing the shared construction code.                       #
    # ------------------------------------------------------------------ #
    def _embed_operators(self, sx, sy, sz, sp, sm):
        """Embed single-TLS operators into the full Hilbert space.

        Default is the identity: the TLS operators already act on the full
        space. Subclasses that add extra subsystems (e.g. a cavity) override
        this to tensor the TLS operators with the extra-mode identities.
        """
        return sx, sy, sz, sp, sm

    def _build_dissipators(self):
        """Populate ``self.c_ops`` for Lindblad-form solvers.

        Default builds the standard per-TLS thermal collapse operators.
        Subclasses with extra channels append to ``self.c_ops`` after calling
        ``super()._build_dissipators()``; purely non-Markovian backends that
        never use collapse operators may override this with a no-op.
        """
        self._build_c_ops()

    def build_operators(self):
        """Build TLS operators and system observables.

        The method constructs Pauli operators for each TLS and sets up the
        collective excitation and spin operators used by solver backends.
        """
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

        # Embed the single-TLS operators into the full Hilbert space.
        # Default is a no-op; subclasses that add extra subsystems (e.g. a
        # cavity) override _embed_operators to tensor them in.
        self.sx, self.sy, self.sz, self.sp, self.sm = self._embed_operators(
            sx_tls, sy_tls, sz_tls, sp_tls, sm_tls
        )

        # observables
        self.collective_sp = sum(self.sp)
        self.collective_sm = sum(self.sm)
        if self.is_qutip_solver:
            self.collective_exc = self.collective_sp * self.collective_sm
        else:
            self.collective_exc = np.matmul(self.collective_sp, self.collective_sm)

        # Dissipators: standard TLS collapse operators by default. Solvers with
        # extra channels (or none) override _build_dissipators.
        self._build_dissipators()

    def build_hamiltonian(self):
        """Construct the static system Hamiltonian."""
        self.H = sum(0.5 * self.omega_tls[i] * self.sz[i] for i in range(self.n_tls))
        for i in range(self.n_tls):
            for j in range(i+1, self.n_tls):
                if self.is_qutip_solver:
                    self.H += self.J * self.sz[i] * self.sz[j]
                else:
                    self.H += self.J * np.matmul(self.sz[i], self.sz[j])

        # Extra model terms (default: none; e.g. Tiered adds a cavity + coupling).
        extra = self._model_hamiltonian()
        if extra is not None:
            self.H = self.H + extra

    def _model_hamiltonian(self):
        """Return extra static Hamiltonian terms for the model, or None."""
        return None
    
    def drive_coeff(self, t, args):
        """Return the time-dependent drive coefficient for the Hamiltonian."""
        if 0.0 <= t <= self.T_drive:
            return 0.5 * self.Omega_amp * np.cos(args["omega"] * t)
        else:
            return 0.0

    # ------------------------------------------------------------------ #
    # State-normalization hooks. These convert a solver's stored states    #
    # into a uniform QuTiP representation so the analysis code below does  #
    # not need to know which backend produced them.                       #
    # ------------------------------------------------------------------ #
    def _to_density_matrix(self, state):
        """Return ``state`` as a full-system QuTiP density matrix.

        Accepts a QuTiP ket/operator or a raw array. Raw arrays are wrapped as
        an ``n_tls``-qubit operator (used by array-based backends such as
        TEMPO). Subclasses with a different Hilbert-space layout may override.
        """
        if not isinstance(state, qt.Qobj):
            dims = [2 for _ in range(self.n_tls)]
            state = qt.Qobj(state, dims=[dims, dims])
        if state.isket:
            state = qt.ket2dm(state)
        return state

    def _reduce_to_tls(self, rho):
        """Reduce a full-system density matrix to the TLS subsystems.

        Default traces out any trailing extra subsystems (e.g. a cavity) so
        only the ``n_tls`` two-level systems remain; a state that already
        contains only the TLSs is returned unchanged.
        """
        if len(rho.dims[0]) > self.n_tls:
            rho = qt.ptrace(rho, list(range(self.n_tls)))
        return rho

    def _state_sequence(self, states):
        """Return an indexable sequence of stored states.

        Default assumes ``states`` is already a sequence; backends that wrap
        their trajectory in a container override this to unwrap it.
        """
        return states

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

        rho = self._to_density_matrix(rho)

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
        states = self._state_sequence(states)

        if len(states) != self.n_time:
            raise ValueError("Error: states length must equal n_time")

        res_dict = {}
        for i in range(self.n_tls):
            for j in range(i+1, self.n_tls):
                entropy_t = np.zeros(self.n_time)
                for idx, state in enumerate(states):
                    rho = self._reduce_to_tls(self._to_density_matrix(state))
                    entropy_t[idx] = qt.entropy_mutual(rho, i, j)
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

    @abstractmethod
    def _worker(self, omega_d, store_states=False):
        """Evolve the system at a single drive frequency.

        Any drive-frequency-independent objects built by ``_prepare`` are
        injected as extra keyword arguments. Returns ``(exc, sp)``, or
        ``(exc, sp, states)`` when ``store_states`` is True.
        """

    def _prepare(self):
        """Return per-run, drive-frequency-independent worker kwargs.

        Default is empty. Solvers that build an expensive object once per run
        (e.g. a bath expansion or a process tensor) override this to return it
        keyed by the argument name their ``_worker`` expects.
        """
        return {}

    def run(self, omega_d_vals, store_states=False):
        """Run the solver across the given drive frequencies.

        Returns ``(exc, sp)`` or ``(exc, sp, states)`` when ``store_states``.
        """
        worker = partial(self._worker, store_states=store_states, **self._prepare())
        return run_parallel(
            omega_d_vals=omega_d_vals,
            worker=worker,
            n_time=self.n_time,
            store_states=store_states,
            desc=f"{self._name} simulations",
        )

    def _get_states(self, omega_d):
        """Return the stored state trajectory from a single-frequency run."""
        _, _, states = self._worker(omega_d, store_states=True, **self._prepare())
        return states

    def husimi_sim(self, omega_d, theta, phi, method, tls_idx=None):
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