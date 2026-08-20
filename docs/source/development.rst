===========
Development
===========

Guidance for working on the package itself: repository layout, environment
setup, building the documentation, and conventions for extending the code.

.. contents:: On this page
   :local:
   :depth: 1

Repository layout
=================

.. code-block:: text

   tls_synchronization/
   ├── src/tls_sync/        # the installable package (src layout)
   ├── phonon_bath/         # BCTDS / phonon-bath example scripts + data
   ├── husimi_function/     # Husimi & correlation example scripts + output
   ├── tests/               # pytest suite
   ├── docs/                # Sphinx documentation (this site)
   ├── config.json          # shared solver defaults
   ├── pyproject.toml       # packaging + tooling metadata
   └── setup.cfg            # additional packaging metadata

Development environment
======================

Install everything, including the ``dev`` group (pytest + Sphinx):

.. code-block:: bash

   uv sync --all-groups

Because the project is installed into the managed environment, edits to files
under ``src/tls_sync`` are picked up on the next ``uv run`` without reinstalling.

Building the documentation
=========================

The documentation uses Sphinx with the Read the Docs theme. Build the HTML
locally from the ``docs`` directory:

.. code-block:: bash

   cd docs
   uv run make html          # macOS / Linux
   # or:  uv run make.bat html   (Windows)

The rendered site is written to ``docs/build/html/``; open
``docs/build/html/index.html`` in a browser.

To build without the ``make`` wrapper:

.. code-block:: bash

   uv run sphinx-build -b html docs/source docs/build/html

Regenerating the API stubs
-------------------------

The :doc:`api` page uses ``automodule`` directives and updates automatically
from docstrings. If you prefer to regenerate per-module stub files (the
``modules.rst`` / ``tls_sync.rst`` style output), use ``sphinx-apidoc``:

.. code-block:: bash

   uv run sphinx-apidoc -o docs/source src/tls_sync --force

Coding conventions
==================

- **Docstrings drive the API docs.** Write NumPy- or Google-style docstrings on
  public classes, methods, and functions so the :doc:`api` page stays complete.
  The Napoleon extension parses both styles.
- **Keep solver state picklable.** Parameter sweeps ship solver objects to
  worker processes; non-picklable attributes will break :mod:`tls_sync.parallel`.
  ``tests/test_pickling.py`` guards this.
- **Keep example scripts runnable from the repository root** so they rely on the
  installed package rather than ``sys.path`` edits.
- **Route shared defaults through** ``config.json`` rather than hard-coding them
  across scripts (see :doc:`configuration`).

Adding a new solver or utility
=============================

1. Add the module under ``src/tls_sync/`` (for example
   ``src/tls_sync/my_solver.py``) and to ``__init__.py`` to simplify imports.
2. Add tests under ``tests/`` mirroring the existing files, and mark full
   end-to-end runs (see :doc:`testing`).
3. Add an ``automodule`` block for it in ``docs/source/api.rst``.

Contributing workflow
=====================

1. Create a feature branch.
2. Make the change with accompanying tests and docstrings.
3. Run the test suite (``uv run pytest"``) and build the docs.
4. Open a pull request describing the change and any new parameters or scripts.