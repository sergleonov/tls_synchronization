===========
Development
===========

Guidance for working on the package itself: repository layout, environment
setup, building the documentation, and conventions for extending the code.

.. contents:: On this page
   :local:
   :depth: 1

Repository layout
=================

.. code-block:: text

   tls_synchronization/
   ├── src/tls_sync/        # the installable package (src layout)
   ├── examples/            # BCTDS / correlation  example scripts
   ├── tests/               # pytest suite
   ├── docs/                # Sphinx documentation (this site)
   ├── config.json          # shared solver defaults
   ├── pyproject.toml       # packaging + tooling metadata
   └── setup.cfg            # additional packaging metadata

Development environment
=======================

Install everything, including the ``dev`` group (pytest + Sphinx):

.. code-block:: bash

   uv sync --all-groups

Because the project is installed into the managed environment, edits to files
under ``src/tls_sync`` are picked up on the next ``uv run`` without reinstalling.

Building the documentation
==========================

The documentation uses Sphinx with the Read the Docs theme. Build the HTML
locally from the ``docs`` directory:

.. code-block:: bash

   cd docs
   uv run make html          # macOS / Linux
   # or:  uv run make.bat html   (Windows)

The rendered site is written to ``docs/build/html/``; open
``docs/build/html/index.html`` in a browser.

To build without the ``make`` wrapper:

.. code-block:: bash

   uv run sphinx-build -b html docs/source docs/build/html

Regenerating the API stubs
--------------------------

The :doc:`api` page uses ``automodule`` directives and updates automatically
from docstrings. If you prefer to regenerate per-module stub files (the
``modules.rst`` / ``tls_sync.rst`` style output), use ``sphinx-apidoc``:

.. code-block:: bash

   uv run sphinx-apidoc -o docs/source src/tls_sync --force

Coding conventions
==================

- **Keep solver state picklable.** Parameter sweeps ship solver objects to
  worker processes; non-picklable attributes will break :mod:`tls_sync.parallel`.
  ``tests/test_pickling.py`` guards this.
- **Docstrings drive the API docs.** Write NumPy- or Google-style docstrings on
public classes, methods, and functions so the :doc:`api` page stays complete.

Adding a new solver
===================

The :class:`~tls_sync.solver.Solver` base class is a template: it owns the
shared model construction, the parallel drive-frequency sweep (``run``), the
single-frequency state retrieval (``_get_states``), the Husimi routine
(``husimi_sim``), and all of the phase, correlation, and entropy analysis. A
new solver supplies only what is specific to it — one required primitive plus a
few optional hooks, each with a sensible default. ``lindblad.py`` is the
minimal example; ``tiered.py`` exercises every hook.

What every solver must provide
------------------------------

1. **Create the module and subclass** ``Solver``. Add
   ``src/tls_sync/my_solver.py`` defining ``class MySolver(Solver)`` and export
   it from ``__init__.py`` to simplify imports.
2. **Wire up** ``__init__``. Store any solver-specific parameters, call
   ``super().__init__(..., is_qutip_solver=<bool>, name="MySolver")``, then
   ``self.build_operators()`` and ``self.build_hamiltonian()``, and finally
   build the initial state (``self.rho0``).
3. **Register the name.** Add ``"MySolver"`` to the ``SOLVERS`` list in
   ``solver.py``; the base ``__init__`` validates ``name`` against it. This is
   the only edit a new solver makes to the base module.
4. **Implement** ``_worker(self, omega_d, store_states=False)``. This is the
   sole abstract method: evolve the system at one drive frequency and return
   ``(exc, sp)``, or ``(exc, sp, states)`` when ``store_states`` is True. The
   inherited ``run`` and ``_get_states`` drive this method for you.
5. **Implement pickling.** Extend ``__getstate__`` with any new constructor
   parameters and reconstruct via ``self.__init__(**d)`` in ``__setstate__``.
   Sweeps ship the solver to worker processes, so this is required, not
   optional (see ``tests/test_pickling.py`` and *Coding conventions* above).

Everything else — ``run``, ``_get_states``, ``husimi_sim``, ``phase_sim``,
``pearson_sim``, ``plv_sim``, ``phase_corr_sim`` — is inherited unchanged.

Optional: precomputing a per-run object
---------------------------------------

If the solver builds one expensive, drive-frequency-independent object per run
(a bath expansion, a process tensor), override ``_prepare`` to return it as a
dict keyed by the argument name your ``_worker`` expects::

   def _prepare(self):
       return {"bath_coeffs": self._bath_to_coeffs(self._build_bath())}

The inherited ``run`` and ``_get_states`` splat this into every worker call, so
the object is built once and reused. ``heom.py`` and ``tempo.py`` are the
examples; the default returns an empty dict (no extra arguments).

Optional: changing the physical model
-------------------------------------

Override these only if the model differs from the default TLS chain. Each has a
default, so leave the ones you do not need untouched:

``_embed_operators(sx, sy, sz, sp, sm)``
   Embed the single-TLS operators into the full Hilbert space. Default is the
   identity; override to tensor in an extra subsystem (see ``tiered.py``, which
   adds a cavity mode).

``_model_hamiltonian()``
   Return extra static Hamiltonian terms, or ``None`` (the default) for none.

``_build_dissipators()``
   Populate ``self.c_ops``. The default builds the standard per-TLS collapse
   operators; call ``super()._build_dissipators()`` and append to add channels,
   or override with a no-op for a solver that uses none (``heom.py``,
   ``tempo.py``).

Optional: non-standard stored states
------------------------------------

The analysis code expects a uniform state representation. Override these only
if ``_worker`` stores states in an unusual form:

``_to_density_matrix(state)``
   Convert a stored state to a QuTiP density matrix. The default accepts QuTiP
   kets or operators and raw arrays (interpreted as ``n_tls`` qubits).

``_reduce_to_tls(rho)``
   Reduce to the TLS subsystems. The default traces out any trailing extra
   subsystem (e.g. a cavity).

``_state_sequence(states)``
   Return an indexable sequence of states. The default assumes ``states`` is
   already one; ``tempo.py`` overrides it to unwrap an oqupy ``Dynamics``.

Finishing up
------------

1. Add tests under ``tests/`` mirroring the existing files, and mark full
   end-to-end runs (see :doc:`testing`).
2. Add an ``automodule`` block for the module in ``docs/source/api.rst``.

Adding a utility follows the same first and last steps: drop the module under
``src/tls_sync/``, export it from ``__init__.py``, and document it in
``docs/source/api.rst``.

Contributing workflow
=====================

1. Create a feature branch.
2. Make the change with accompanying tests and docstrings.
3. Run the test suite (``uv run pytest"``) and build the docs.
4. Open a pull request describing the change and any new parameters or scripts.