=========
Changelog
=========

All notable changes to this project are documented here. The format is based on
`Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_, and the project aims
to follow `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
==========

Added
-----

- Sphinx / Read the Docs documentation set: getting-started, user guide,
  API reference, and development pages.

0.1.0
=====

Initial release.

Added
-----

- ``src/tls_sync`` package with HEOM, TEMPO, tiered, and Lindblad solvers.
- Plotting and FFT utilities (:mod:`tls_sync.plotting`), including Husimi
  animations and correlation plots.
- Parallel sweep helpers (:mod:`tls_sync.parallel`).
- Example scripts under ``husimi_function/`` and ``phonon_bath/``.
- ``pytest`` suite with a ``slow`` marker for full end-to-end simulations.
- ``config.json`` for shared solver defaults.