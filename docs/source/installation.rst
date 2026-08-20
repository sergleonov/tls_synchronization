============
Installation
============

.. contents:: On this page
   :local:
   :depth: 1

Prerequisites
=============

- **Python 3.11 or newer** (declared as ``requires-python = ">=3.11"``).
- **uv** — the package and environment manager.
- **git** — one dependency, ``oqupy``, is installed directly from its GitHub
  repository.

Installing uv
=============

If you do not already have ``uv`` installed:

.. code-block:: bash

   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

See the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_
for alternatives (Homebrew, pipx, etc.).

Installing the project
======================
Clone the repo into your working directory:
.. code-block:: bash

   git clone https://github.com/sergleonov/tls_synchronization.git

From the repository root, install the project together with its development
dependency groups:

.. code-block:: bash

   uv sync --all-groups

This creates a managed virtual environment in ``.venv`` and installs:

- the runtime dependencies (``numpy``, ``scipy``, ``matplotlib``, ``qutip``,
  ``mpmath``, ``pyqt5``, ``tqdm``, and ``oqupy`` from git), and
- the ``dev`` group (``pytest``, ``sphinx``, ``sphinx-rtd-theme``).

If you only need the runtime environment, omit the flag:

.. code-block:: bash

   uv sync

Running code in the environment
===============================

Prefix commands with ``uv run`` to execute them inside the managed
environment without activating it manually:

.. code-block:: bash

   uv run python path/to/script.py

Alternatively, activate the environment directly:

.. code-block:: bash

   # macOS / Linux
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

Verifying the installation
==========================

Confirm the package imports cleanly:

.. code-block:: bash

   uv run python -c "from tls_sync.solver import HEOM, Lindblad, TieredSolver, TEMPO; print('ok')"

And run the fast portion of the test suite:

.. code-block:: bash

   uv run pytest -m "not slow"

See :doc:`testing` for more on the test suite.

Notes on ``oqupy``
==================

``oqupy`` is pinned to the upstream TEMPO Collaboration repository rather than a
PyPI release:

.. code-block:: toml

   [tool.uv.sources]
   oqupy = { git = "https://github.com/tempoCollaboration/OQuPy.git" }

Because of this, the first ``uv sync`` requires network access to GitHub. If you
work behind a proxy or an air-gapped environment, mirror or vendor this
dependency before syncing.