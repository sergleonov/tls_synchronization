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
   Produces an evolution of Husimi-Q function for 2 interacting TLSs.

``husimi_snapshots.py``
   Renders static Husimi snapshots at selected times, useful for figures where
   an animation is impractical.

``correlation_plots.py``
   Plots time-resolved correlations between the two-level systems.

``correlation_heatmap.py``
   Computes and plots final time correlation values between TLSs for a chosen metric.

``correlation_J_sweep.py``
   Sweeps the TLS–TLS coupling ``J`` and TLS frequency ratio and plots the heatmap.  

``phase_plots.py``
   Simulates the phase evolution and phase differences between TLSs over time.

``phase_corr_plots.py``
   Plots phase difference and time-dependent correlations between TLSs.

Phonon-bath / BCTDS examples
============================

Scripts in ``phonon_bath/`` set up phonon-bath models and run BCTDS-style
simulations across the different solvers.

``heom_bctds.py``
   Runs a BCTDS simulation with the HEOM solver comparing against Markovian solver.

``tempo_bctds.py``
   Runs a BCTDS simulation with the TEMPO solver comparing against Markovian solver.

``tiered_bctds.py``
   Runs a BCTDS simulation with the Tiered solver comparing against Markovian solver.

``tiered_heom_bctds.py``
   Compares the BCTDS response of Tiered and HEOM solvers to Markovian simulation.

``heom_sweep.py``
   Performs a power law spectral density order sweep for the HEOM solver.

A minimal example
============================

The following mirrors what the scripts do, condensed to the essentials:

.. code-block:: python

   from tls_sync import HEOM
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