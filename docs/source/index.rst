.. TLS Synchronization documentation master file.

===================
TLS Synchronization
===================

**TLS Synchronization** is a Python package for simulating the dynamics of
open quantum systems in the context of *Broadband Cryogenic Transient
Dielectric Spectroscopy* (BCTDS). Its goal is to help understand the phase
synchronization of two-level defects (two-level systems, or TLS) in amorphous
materials.

The package provides a common interface over several open quantum system solvers and a
set of plotting and analysis utilities for producing Husimi visualizations,
correlation and phase-evolution plots, and spin-boson dynamics.

.. contents:: On this page
   :local:
   :depth: 1

Features
========

- **HEOM** (:mod:`tls_sync.heom`) — hierarchical equations of motion.
- **TEMPO** (:mod:`tls_sync.tempo`) — time-evolving matrix product operators.
- **Tiered solver** (:mod:`tls_sync.tiered`) — a tiered environment combining a
  single mode with a thermal bath.
- **Lindblad** (:mod:`tls_sync.lindblad`) — Lindblad master-equation dynamics.
- **Plotting & FFT utilities** (:mod:`tls_sync.plotting`) — Husimi animations,
  correlation plots, and spectral analysis.
- **Parallel helpers** (:mod:`tls_sync.parallel`) — utilities for running
  parameter sweeps concurrently.

At a glance
===========

.. code-block:: python

   from tls_sync.solver import HEOM
   from tls_sync.plotting import generate_husimi_anim

   heom = HEOM(
       tls_freqs=[3.75, 3.82],
       J=0.02,
       Omega_amp=0.1,
       lam=0.002,
       gamma_bath=0.05,
       T=0.5,
       Nk=3,
       max_depth=5,
       T_total=400,
       T_drive=100.0,
       dt=0.5,
       n_tls=2,
       n_freqs=300,
       sd_type="drude",
       ohmicity=None,
   )

   # Run methods on the solver and visualize the results.

See :doc:`quickstart` for a complete first run, and :doc:`examples` for the
bundled example scripts.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   configuration

.. toctree::
   :maxdepth: 2
   :caption: User guide

   usage
   examples
   theory

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

.. toctree::
   :maxdepth: 2
   :caption: Development

   development
   testing
   changelog

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`