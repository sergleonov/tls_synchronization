========
Examples
========

The repository bundles runnable example scripts in two directories. They show
how to drive the solvers end to end and how to turn results into figures,
animations, and saved data.

.. contents:: On this page
   :local:
   :depth: 1

Running the scripts
===================

Run every script **from the repository root** so that the installed package
resolves correctly and no ``sys.path`` manipulation is needed:

.. code-block:: bash

   uv run python husimi_function/husimi_animation.py
   uv run python phonon_bath/heom_bctds.py

Outputs (figures, animations, and ``.npz`` data) are written into the
corresponding output directories, for example ``phonon_bath/bctds_data`` and
``phonon_bath/bctds_figures``.

Husimi and correlation examples
===============================

Scripts in ``husimi_function/`` focus on phase-space (Husimi) visualization and
on correlation / phase-synchronization diagnostics.

``husimi_animation.py``
   Produces an animated Husimi Q-function of the driven TLS as the system
   evolves.

``husimi_snapshots.py``
   Renders static Husimi snapshots at selected times, useful for figures where
   an animation is impractical.

``correlation_plots.py``
   Plots time-resolved correlations between the two-level systems.

``correlation_heatmap.py``
   Computes and renders a two-dimensional correlation heatmap.

``correlation_J_sweep.py``
   Sweeps the TLS–TLS coupling ``J`` and plots how correlations respond — a
   direct probe of synchronization onset.

``plot_saved_correlation_heatmap.py``
   Re-renders a correlation heatmap from previously saved data without rerunning
   the simulation.

``phase_plots.py``
   Plots the phase evolution of each TLS over time.

``phase_corr_plots.py``
   Plots phase-correlation diagnostics between the TLS.

Phonon-bath / BCTDS examples
============================

Scripts in ``phonon_bath/`` set up phonon-bath models and run BCTDS-style
simulations across the different solvers.

``bosonic_bath.py``
   Constructs and inspects a bosonic (phonon) bath / spectral density used by
   the other scripts.

``heom_bctds.py``
   Runs a BCTDS simulation with the HEOM solver.

``tempo_bctds.py``
   Runs the same class of BCTDS simulation with the TEMPO solver.

``tiered_bctds.py``
   Runs a BCTDS simulation with the tiered (single mode + thermal bath) solver.

``tiered_heom_bctds.py``
   Runs a combined tiered + HEOM treatment of the BCTDS problem.

``heom_sweep.py``
   Performs a parameter sweep using the HEOM solver, distributing runs with the
   helpers in :mod:`tls_sync.parallel`.

``heom_sweep_plot.py``
   Plots the aggregated results produced by ``heom_sweep.py``.

A minimal end-to-end example
============================

The following mirrors what the scripts do, condensed to the essentials:

.. code-block:: python

   from tls_sync.solver import HEOM
   from tls_sync.plotting import generate_husimi_anim, plot_correlations

   heom = HEOM(
       tls_freqs=[3.75, 3.82], J=0.02, Omega_amp=0.1, lam=0.002,
       gamma_bath=0.05, T=0.5, Nk=3, max_depth=5, T_total=400,
       T_drive=100.0, dt=0.5, n_tls=2, n_freqs=300, sd_type="drude",
       ohmicity=None,
   )

   # result = heom.run()
   # plot_correlations(result)
   # generate_husimi_anim(result)

.. tip::

   Start from ``heom_bctds.py`` for a single representative run, then move to
   ``heom_sweep.py`` when you want to scan a parameter. Use
   ``plot_saved_correlation_heatmap.py`` and ``heom_sweep_plot.py`` to iterate
   on figures without paying for the simulation again.