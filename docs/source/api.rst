=============
API reference
=============

This page documents the public interface of ``tls_sync``. Signatures and
descriptions are generated directly from the source docstrings, so this is the
authoritative reference for constructor arguments and method behavior.

.. contents:: On this page
   :local:
   :depth: 1

Solvers
=======

.. currentmodule:: tls_sync

All solvers share the :class:`Solver` base class, so the analysis methods
(``run``, ``pearson_sim``, ``husimi_sim``, ``phase_sim``, ``phase_corr_sim``,
``correlation_sim``, ``final_corr_from_states``, ``get_name``, ...) are
inherited. They are documented on :class:`Solver` and, thanks to
``:inherited-members:``, repeated on each concrete solver so every solver page
is self-contained.

Solver
------

.. autoclass:: Solver
   :members:
   :undoc-members:
   :show-inheritance:

HEOM
----

.. autoclass:: HEOM
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

TEMPO
-----

.. autoclass:: TEMPO
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Tiered solver
-------------

.. autoclass:: TieredSolver
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Lindblad
--------

.. autoclass:: Lindblad
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Plotting and analysis
=====================

.. automodule:: tls_sync.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Parallel execution
==================

.. automodule:: tls_sync.parallel
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
=========

.. automodule:: tls_sync.utils
   :members:
   :undoc-members:
   :show-inheritance: