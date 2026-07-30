# TLS Synchronization

A Python package for simulating dynamics of open quantum systems in the context of Broadband Cryogenic Transient Dielectric Spectroscopy (BCTDS) experiment for understanding phase synchronization of the two-level defects in amorphous materials. 

## Project overview

This repository provides a source-layout package under `src/tls_sync` with solver implementations for:

- HEOM (`tls_sync.heom`) for hierarchical equations of motion
- TEMPO (`tls_sync.tempo`) for time-evolving matrix product operators
- Tiered solver (`tls_sync.tiered`) for tiered environment with single mode and thermal bath
- Lindblad master equation dynamics (`tls_sync.lindblad`)
- FFT and plotting utilities in `tls_sync.plotting`

Example scripts in `husimi_function/` and `phonon_bath/` show how to use the package to generate Husimi visualizations, correlation plots, phase evolution, and bath dynamics.

## Installation

Install the package in editable mode from the repo root:

```bash
python3 -m pip install -e .
```

Install runtime dependencies manually if needed:

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Once installed, import the package from anywhere in your environment:

```python
from tls_sync.solver import HEOM, Lindblad, TieredSolver, TEMPO
from tls_sync.plotting import generate_husimi_anim, plot_correlations
```

### Example

```python
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

# Run methods on the solver and visualize results
```

## Example scripts

The repository includes the following example scripts:

- `husimi_function/correlation_J_sweep.py`
- `husimi_function/correlation_plots.py`
- `husimi_function/husimi_animation.py`
- `husimi_function/phase_corr_plots.py`
- `husimi_function/phase_plots.py`
- `phonon_bath/heom_bctds.py`
- `phonon_bath/heom_sweep_plot.py`
- `phonon_bath/heom_sweep.py`
- `phonon_bath/tempo_bctds.py`
- `phonon_bath/tiered_bctds.py`
- `phonon_bath/tiered_heom_bctds.py`

Run them from the repository root after installing the package to avoid modifying `sys.path`.

## Configuration

A simple configuration file, `config.json`, is included for solver defaults and can be extended as needed.

## Development

- Package source is under `src/tls_sync`
- Use `setup.cfg` and `pyproject.toml` for packaging metadata
- Add new solver modules or plotting utilities inside `src/tls_sync`

## Notes

- This repo is intended for numerical experiments with TLS synchronization and bath dynamics.
- The package uses `numpy`, `matplotlib`, `scipy`, `qutip`, `oqupy`, and `tqdm`.
