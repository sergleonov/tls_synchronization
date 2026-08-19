==========
Quickstart
==========

This page walks through a first simulation from an installed environment. If you
have not installed the package yet, start with :doc:`installation`.

.. contents:: On this page
   :local:
   :depth: 1

Importing the package
=====================

Solvers and plotting utilities are exposed through two convenience modules:

.. code-block:: python

   from tls_sync.solver import HEOM, Lindblad, TieredSolver, TEMPO
   from tls_sync.plotting import generate_husimi_anim, plot_correlations

Building a solver
=================

Every solver is configured through its constructor. The example below sets up a
two-TLS HEOM simulation with a driven Drude spectral density:

.. code-block:: python

   from tls_sync.solver import HEOM

   heom = HEOM(
       tls_freqs=[3.75, 3.82],   # bare frequencies of the two TLS
       J=0.02,                   # TLS–TLS coupling strength
       Omega_amp=0.1,            # drive amplitude
       lam=0.002,                # system–bath coupling (reorganization energy)
       gamma_bath=0.05,          # bath cutoff / relaxation rate
       T=0.5,                    # temperature
       Nk=3,                     # number of Matsubara terms
       max_depth=5,              # HEOM hierarchy truncation depth
       T_total=400,              # total simulation time
       T_drive=100.0,            # duration of the drive
       dt=0.5,                   # time step
       n_tls=2,                  # number of two-level systems
       n_freqs=300,              # frequency-grid resolution
       sd_type="drude",          # spectral-density model
       ohmicity=None,            # ohmicity exponent (for ohmic spectral densities)
   )

.. note::

   Parameter names and defaults are documented alongside each solver in the
   :doc:`api`. Use that reference as the source of truth for the exact
   constructor signatures.

Running and visualizing
=======================

Once constructed, run the solver's dynamics and pass the result to a plotting
utility. A typical Husimi animation looks like this:

.. code-block:: python

   from tls_sync.plotting import generate_husimi_anim

   # result = heom.run()            # evolve the system
   # generate_husimi_anim(result)   # produce a Husimi Q-function animation

Correlation plots follow the same pattern:

.. code-block:: python

   from tls_sync.plotting import plot_correlations

   # plot_correlations(result)

Where to go next
================

- :doc:`usage` — a fuller tour of the solvers and utilities.
- :doc:`examples` — ready-to-run scripts in ``husimi_function/`` and
  ``phonon_bath/``.
- :doc:`configuration` — solver defaults via ``config.json``.
- :doc:`theory` — background on BCTDS, TLS, and the solver methods.