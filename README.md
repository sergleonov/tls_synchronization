# TLS Synchronization

A Python package for simulating the dynamics of open quantum systems in the
context of **Broadband Cryogenic Transient Dielectric Spectroscopy (BCTDS)**,
used to study the phase synchronization of two-level defects (TLS) in amorphous
materials.

The package wraps several open-system solvers behind a common interface and
provides analysis utilities for Husimi visualizations, correlation and
phase-evolution plots, and phonon-bath dynamics.

---

## Features

| Solver | Module | Method |
| --- | --- | --- |
| `HEOM` | `tls_sync.heom` | Hierarchical equations of motion (numerically exact for a Gaussian bath) |
| `TEMPO` | `tls_sync.tempo` | Time-evolving matrix product operators (via [OQuPy](https://github.com/tempoCollaboration/OQuPy)) |
| `TieredSolver` | `tls_sync.tiered` | Tiered environment: explicit single mode + thermal bath |
| `Lindblad` | `tls_sync.lindblad` | Markovian master-equation dynamics (baseline) |

Plus `tls_sync.plotting` (Husimi animations, correlation plots, FFT utilities),
`tls_sync.parallel` (concurrent parameter sweeps), and `tls_sync.utils` (shared
operators and helpers).

## Requirements

- **Python ≥ 3.12**
- [**uv**](https://docs.astral.sh/uv/) for environment and dependency management
- **git** — `oqupy` is installed directly from its upstream repository

## Installation

```bash
# from the repository root
uv sync --all-groups
```

This creates a `.venv` and installs runtime dependencies (`numpy`, `scipy`,
`matplotlib`, `qutip`, `oqupy`, `mpmath`, `pyqt5`, `tqdm`) plus the `dev` group
(`pytest`, `sphinx`, `sphinx-rtd-theme`). Use `uv sync` alone to skip dev tools.

Verify the install:

```bash
uv run python -c "from tls_sync.solver import HEOM, Lindblad, TieredSolver, TEMPO; print('ok')"
```

> **Note:** the first sync needs network access to GitHub to fetch `oqupy`.

## Quickstart

```python
from tls_sync.solver import HEOM
from tls_sync.plotting import generate_husimi_anim, plot_correlations

heom = HEOM(
    tls_freqs=[3.75, 3.82],   # bare TLS frequencies
    J=0.02,                   # TLS–TLS coupling
    Omega_amp=0.1,            # drive amplitude
    lam=0.002,                # system–bath coupling
    gamma_bath=0.05,          # bath cutoff / relaxation rate
    T=0.5,                    # temperature
    Nk=3,                     # Matsubara terms
    max_depth=5,              # HEOM hierarchy depth
    T_total=400,              # total simulation time
    T_drive=100.0,            # drive duration
    dt=0.5,                   # time step
    n_tls=2,                  # number of TLS
    n_freqs=300,              # frequency-grid resolution
    sd_type="drude",          # spectral-density model
    ohmicity=None,            # ohmicity exponent (ohmic baths)
)

# result = heom.run()
# plot_correlations(result)
# generate_husimi_anim(result)
```

Exact constructor arguments and methods for each solver are in the
[API reference](#documentation).

## Usage

Import solvers from the aggregate `solver` module and analysis helpers from
`plotting`:

```python
from tls_sync.solver import HEOM, Lindblad, TieredSolver, TEMPO
from tls_sync.plotting import generate_husimi_anim, plot_correlations
```

**Choosing a solver:** `Lindblad` is the cheap Markovian baseline; `HEOM` is
numerically exact for a Gaussian bath (cost grows with `Nk` and `max_depth`);
`TEMPO` handles strong coupling and long memory efficiently; `TieredSolver`
treats one dominant mode explicitly alongside a residual thermal bath.

## Example scripts

Run all scripts **from the repository root** so the installed package resolves
without `sys.path` edits:

```bash
uv run python phonon_bath/heom_bctds.py
uv run python husimi_function/husimi_animation.py
```

**`husimi_function/`** — phase-space and synchronization diagnostics:
`husimi_animation.py`, `husimi_snapshots.py`, `correlation_plots.py`,
`correlation_heatmap.py`, `correlation_J_sweep.py`,
`plot_saved_correlation_heatmap.py`, `phase_plots.py`, `phase_corr_plots.py`.

**`phonon_bath/`** — BCTDS runs and bath dynamics: `bosonic_bath.py`,
`heom_bctds.py`, `tempo_bctds.py`, `tiered_bctds.py`, `tiered_heom_bctds.py`,
`heom_sweep.py`, `heom_sweep_plot.py`.

Outputs (figures, animations, `.npz` data) are written to the corresponding
`bctds_data/` and `bctds_figures/` directories.

## Configuration

Shared solver defaults live in `config.json` at the repository root. Load and
unpack them into a solver, overriding individual values as needed:

```python
import json
from pathlib import Path
from tls_sync.solver import HEOM

config = json.loads(Path("config.json").read_text())
heom = HEOM(**{**config["solver_defaults"], "J": 0.05})
```

## Documentation

Full documentation (getting started, user guide, theory background, API
reference, and development notes) is built with Sphinx and the Read the Docs
theme.

```bash
cd docs
uv run make html          # output in docs/build/html/
```

To build without the wrapper:

```bash
uv run sphinx-build -b html docs/source docs/build/html
```

## Testing

```bash
uv run pytest                 # full suite
uv run pytest -m "not slow"   # skip end-to-end simulations
uv run pytest -m slow         # only the slow simulations
```

The `slow` marker tags full simulation runs. Because parameter sweeps ship
solver objects to worker processes, solver state must remain picklable —
`test_pickling.py` and `test_parallel.py` guard this.

## Project layout

```
tls_synchronization/
├── src/tls_sync/        # installable package (src layout)
├── phonon_bath/         # BCTDS / phonon-bath scripts + data
├── husimi_function/     # Husimi & correlation scripts + output
├── tests/               # pytest suite
├── docs/                # Sphinx documentation
├── config.json          # shared solver defaults
├── pyproject.toml       # packaging + tooling metadata
└── setup.cfg            # additional packaging metadata
```

## Development

- Package source lives in `src/tls_sync`; edits are picked up on the next
  `uv run` without reinstalling.
- Write NumPy- or Google-style docstrings — the API docs are generated from them.
- Re-export new solver classes from `tls_sync/solver.py` and add them to the
  docs API page.
- Keep example scripts runnable from the repository root and route shared
  defaults through `config.json`.

## Dependencies

`numpy`, `scipy`, `matplotlib`, `qutip`, `oqupy`, `mpmath`, `pyqt5`, `tqdm`.

## Author

Sergei Leonov — <sleonov27@amherst.edu>