=============
Configuration
=============

The repository ships a ``config.json`` file at its root that holds solver
defaults. Centralizing defaults keeps example scripts terse and makes it easy to
reproduce a run by editing a single file rather than many call sites.

.. contents:: On this page
   :local:
   :depth: 1

Purpose
=======

``config.json`` is intended for values that are shared across scripts and that
you tune between experiments, for example:

- default physical parameters (frequencies, couplings, temperature),
- default numerical parameters (time step, total time, truncation depth,
  frequency-grid resolution),
- output locations for figures and data.

Structure
=========

The file is standard JSON. A representative layout is shown below — adapt the
keys to match the defaults your scripts actually read.

.. code-block:: json

   {
     "solver_defaults": {
       "tls_freqs": [3.75, 3.82],
       "J": 0.02,
       "Omega_amp": 0.1,
       "lam": 0.002,
       "gamma_bath": 0.05,
       "T": 0.5,
       "Nk": 3,
       "max_depth": 5,
       "T_total": 400,
       "T_drive": 100.0,
       "dt": 0.5,
       "n_tls": 2,
       "n_freqs": 300,
       "sd_type": "drude",
       "ohmicity": null
     },
     "output": {
       "data_dir": "phonon_bath/bctds_data",
       "figure_dir": "phonon_bath/bctds_figures"
     }
   }

Loading the configuration
==========================

Read the file with the standard library and unpack it into a solver:

.. code-block:: python

   import json
   from pathlib import Path

   from tls_sync.solver import HEOM

   config = json.loads(Path("config.json").read_text())
   heom = HEOM(**config["solver_defaults"])

Overriding defaults
===================

Because the solver is constructed from keyword arguments, you can load the
shared defaults and override individual values per run:

.. code-block:: python

   params = {**config["solver_defaults"], "J": 0.05, "T": 1.0}
   heom = HEOM(**params)
