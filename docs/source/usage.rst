==========
User guide
==========

This guide describes the package's structure, the available solvers and how they
differ, and the shared workflow for running simulations and analyzing results.

.. contents:: On this page
   :local:
   :depth: 1


Importing
=========

Import solver classes from ``tls_sync`` and analysis helpers
from ``plotting``:

.. code-block:: python

   from tls_sync import HEOM, Lindblad, TieredSolver, TEMPO
   from tls_sync.plotting import generate_husimi_anim, plot_correlations

Choosing a solver
=================

Each solver trades accuracy against cost differently. As a rough guide:

**Lindblad** (:class:`~tls_sync.lindblad.Lindblad`)
   A Markovian master equation. Cheapest and most robust; appropriate when the
   bath is memoryless (weak coupling, fast bath). Use it as a baseline for the non-Markovian solvers.

**HEOM** (:class:`~tls_sync.heom.HEOM`)
   Numerically exact for a Gaussian bath with a given spectral density,
   capturing non-Markovian memory through an auxiliary-density operator hierarchy.
   Accuracy is controlled by the Matsubara count (``Nk``) and the hierarchy
   truncation (``max_depth``); cost grows quickly with both.

**TEMPO** (:class:`~tls_sync.tempo.TEMPO`)
   A tensor-network approach that represents the influence functional as a
   matrix product operator. Handles strong coupling and long memory times
   efficiently; accuracy is set by the memory cutoff and bond-dimension
   tolerances.

**Tiered solver** (:class:`~tls_sync.tiered.TieredSolver`)
   Splits the environment into an explicitly modeled single mode plus a residual
   thermal bath. Useful when one bath mode dominates and should be treated
   non-perturbatively while the rest is treated more coarsely.

A common workflow
=================

Regardless of solver, the pattern is the same:

1. **Configure** the solver via its constructor (see :doc:`configuration` for
   sharing defaults through ``config.json``).
2. **Run** the dynamics to obtain a time-resolved result.
3. **Analyze / visualize** with :mod:`tls_sync.plotting`.

.. code-block:: python

   from tls_sync.solver import HEOM
   from tls_sync.plotting import plot_correlations

   heom = HEOM(
       tls_freqs=[3.75, 3.82], J=0.02, Omega_amp=0.1, lam=0.002,
       gamma_bath=0.05, T=0.5, Nk=3, max_depth=5, T_total=400,
       T_drive=100.0, dt=0.5, n_tls=2, n_freqs=300, sd_type="drude",
       ohmicity=None,
   )

   # result = heom.run()
   # plot_correlations(result)

Plotting and analysis
=====================

:mod:`tls_sync.plotting` provides the visualization scripts used throughout the
examples, including Husimi Q-function animations
(:func:`~tls_sync.plotting.generate_husimi_anim`), correlation plots
(:func:`~tls_sync.plotting.plot_correlations`), along with phase-V visualization for
spectral analysis. See the :doc:`api` for the complete list.

Parameter sweeps
===============

For scans over a parameter (for example the coupling ``J`` or the drive frequency
``omega_d``), :mod:`tls_sync.parallel` provides helpers to distribute independent runs
across processes. The ``phonon_bath/heom_sweep.py`` and
``husimi_function/correlation_J_sweep.py`` scripts demonstrate this pattern; see
:doc:`examples`.

.. note::

   Objects that are dispatched to worker processes must be picklable. The test
   suite includes ``test_pickling.py`` and ``test_parallel.py`` to guard this;
   keep new solver state serializable when extending the package.