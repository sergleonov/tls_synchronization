=======
Testing
=======

The package is covered by a ``pytest`` suite under ``tests/``. This page
describes how to run it and how it is organized.

.. contents:: On this page
   :local:
   :depth: 1

Running the tests
=================

Run the whole suite in the managed environment:

.. code-block:: bash

   uv run pytest

Run a single file or test:

.. code-block:: bash

   uv run pytest tests/test_operators.py
   uv run pytest tests/test_operators.py::test_name

Configuration
=============

Test settings live in ``pyproject.toml`` under ``[tool.pytest.ini_options]``:

- ``testpaths = ["tests"]`` — collection is rooted at ``tests/``.
- ``pythonpath = ["tests"]`` — the ``tests`` directory is importable, so shared
  helpers such as ``_config``, ``_helpers``, and ``_workers`` can be imported by
  test modules.
- ``addopts = "-ra"`` — show a short summary for all non-passing outcomes.
- ``filterwarnings`` — ``DeprecationWarning`` and ``PendingDeprecationWarning``
  are ignored to keep output readable.

Suite overview
==============

The suite mixes fast unit tests with heavier integration tests:

===================================  ================================================
Test module                          Focus
===================================  ================================================
``test_imports.py``                  Package and public API import cleanly.
``test_operators.py``                Operator construction and algebra.
``test_utils.py``                    Utility / helper functions.
``test_correlations.py``             Correlation calculations.
``test_husimi.py``                   Husimi Q-function generation.
``test_plotting.py``                 Plotting utilities.
``test_freq_sweep.py``               Frequency-sweep behavior.
``test_heom_bath.py``                HEOM bath / spectral-density handling.
``test_solver_phase_correlations.py`` Phase-correlation results from the solvers.
``test_parallel.py``                 Parrallel parameter sweeps.
``test_pickling.py``                 Solver objects remain picklable for workers.
===================================  ================================================

Shared fixtures and helpers live in ``conftest.py`` and the underscore-prefixed
modules (``_config.py``, ``_helpers.py``, ``_workers.py``).