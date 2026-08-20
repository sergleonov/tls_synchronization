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

The suite mixes fast unit tests with heavier integration tests. Shared fixtures and helpers live in ``conftest.py`` and the underscore-prefixed
modules (``_config.py``, ``_helpers.py``, ``_workers.py``).