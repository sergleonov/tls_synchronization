=======================
Background and methods
=======================

This page gives a high-level orientation to the physics the package models and
to the numerical methods behind each solver. It is meant as onboarding context,
not a derivation; consult the cited literature for full treatments.

.. contents:: On this page
   :local:
   :depth: 1

Physical context
================

Two-level systems (TLS)
-----------------------

Amorphous (glassy) materials host low-energy excitations well described as
*two-level systems*: microscopic degrees of freedom that tunnel between two
nearly degenerate configurations. They dominate dielectric and acoustic loss at
low temperature and are a leading source of decoherence in superconducting
devices.

Broadband Cryogenic Transient Dielectric Spectroscopy (BCTDS)
-------------------------------------------------------------

BCTDS probes the dielectric response of such materials at cryogenic
temperatures over a broad frequency range and in the transient (time-resolved)
regime. This package simulates the open-system dynamics relevant to that
experiment, with a focus on whether and how the phases of coupled TLS
**synchronize** under driving and shared bath coupling.

Synchronization
---------------

Synchronization is the tendency of coupled oscillators to lock their phases.
Here the relevant knobs are the direct TLS–TLS coupling (``J``), the drive
(``Omega_amp``, ``T_drive``), and the shared environment. The correlation and
phase-plot utilities are the diagnostics used to detect synchronization onset
as these parameters vary.

Open quantum systems
====================

A TLS is never fully isolated; it couples to a phonon bath. The reduced state
of the system obeys a dynamical map obtained by tracing out the environment.
The solvers differ in how they represent that environment and its memory.

Lindblad master equation
-------------------------

The Lindblad equation is the most general *Markovian* (memoryless) generator of
completely positive, trace-preserving dynamics. It is inexpensive and a good
baseline, but it discards bath memory and is only quantitatively reliable in the
weak-coupling / fast-bath limit.

Hierarchical equations of motion (HEOM)
---------------------------------------

HEOM is numerically exact for a Gaussian bath with a specified spectral density.
It introduces a hierarchy of *auxiliary density operators* that encode bath
correlations, decomposing the bath correlation function into exponential terms
(Matsubara and/or Padé). Convergence is governed by the number of expansion
terms (``Nk``) and the truncation depth of the hierarchy (``max_depth``).

TEMPO
-----

TEMPO (Time-Evolving Matrix Product Operators) represents the Feynman–Vernon
influence functional as a matrix product operator and contracts it efficiently
with a tensor network. It handles strong coupling and long memory times where
HEOM's hierarchy becomes expensive, with accuracy controlled by the memory
cutoff and singular-value / bond-dimension tolerances. In this package TEMPO is
backed by ``oqupy``.

Tiered environment
------------------

The tiered solver splits the environment into an explicitly modeled single mode
(treated non-perturbatively, e.g. a dominant phonon mode) plus a residual
thermal bath treated more coarsely. This is efficient when a single mode
dominates the system–environment interaction.

Spectral densities
==================

The bath is characterized by its spectral density :math:`J(\omega)`. The
solvers accept a spectral-density type (``sd_type``, e.g. ``"drude"``) and, for
ohmic families, an ``ohmicity`` exponent. The Drude (overdamped Brownian) form
is a common choice for phonon baths and admits a convenient exponential
decomposition for HEOM.

Phase-space and the Husimi function
===================================

The Husimi Q-function is a smoothed, everywhere-non-negative quasi-probability
distribution over phase space. It is convenient for visualizing the state and,
in particular, its phase — which is exactly what the synchronization diagnostics
in ``husimi_function/`` exploit.

References
==============

- Müller, Clemens, Jared H. Cole, and Jürgen Lisenfeld. "Towards understanding two-level-systems in amorphous solids: insights from quantum circuits." Reports on Progress in Physics 82, no. 12 (2019): 124501. https://iopscience.iop.org/article/10.1088/1361-6633/ab3a7e

- Strathearn, Aidan, Peter Kirton, Dainius Kilda, Jonathan Keeling, and Brendon William Lovett. "Efficient non-Markovian quantum dynamics using time-evolving matrix product operators." Nature communications 9, no. 1 (2018): 3322. https://www.nature.com/articles/s41467-018-05617-3

- Fux, Gerald E., Piper Fowler-Wright, Joel Beckles, Eoin P. Butler, Paul R. Eastham, Dominic Gribben, Jonathan Keeling et al. "OQuPy: A Python package to efficiently simulate non-Markovian open quantum systems with process tensors." The Journal of Chemical Physics 161, no. 12 (2024). https://arxiv.org/abs/2406.16650

- Johansson, J. Robert, Paul D. Nation, and Franco Nori. "QuTiP: An open-source Python framework for the dynamics of open quantum systems." Computer physics communications 183, no. 8 (2012): 1760-1772. https://doi.org/10.1016/j.cpc.2012.02.021

- Tanimura, Yoshitaka. "Numerically “exact” approach to open quantum dynamics: The hierarchical equations of motion (HEOM)." The Journal of chemical physics 153, no. 2 (2020). https://pubs.aip.org/aip/jcp/article/153/2/020901/76291

- Settimo, Federico, and Bassano Vacchini. “Synchronization Effects in a Periodically Driven Two-Level System.” Physical Review A 113, no. 2 (2026): 022213. https://doi.org/10.1103/cswq-l3c8.

- Lambert, Neill, Tarun Raheja, Simon Cross, et al. “QuTiP-BoFiN: A Bosonic and Fermionic Numerical Hierarchical-Equations-of-Motion Library with Applications in Light-Harvesting, Quantum Control, and Single-Molecule Electronics.” Physical Review Research 5, no. 1 (2023): 013181. https://doi.org/10.1103/PhysRevResearch.5.013181.

- Zhang, Liyun, Zhao Wang, Yucheng Wang, et al. “Quantum Synchronization of a Single Trapped-Ion Qubit.” Physical Review Research 5, no. 3 (2023): 033209. https://doi.org/10.1103/PhysRevResearch.5.033209.

- Wang, Qianxu, Juan S. Salcedo-Gallo, Sara Magdalena Gómez, et al. “Probing the Dynamics of Two-Level System Defect Ensembles via Broadband Cryogenic Transient Dielectric Spectroscopy.” arXiv:2505.18263. Preprint, arXiv, July 22, 2026. https://arxiv.org/abs/2505.18263.