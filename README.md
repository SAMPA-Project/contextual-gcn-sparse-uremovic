# hybrid-context-gcrnn

Official implementation accompanying:

> Niko Uremović, Domen Mongus, Aleksander Pur, Niko Lukač. **Contextualized
> spatio-temporal graph-based method for forecasting sparse geospatial sensor
> networks.** *Expert Systems with Applications*, vol. 294, p. 128779, 2025.

Sparse sensor networks (weather stations, traffic counters, air-quality
monitors, ...) leave large unmeasured gaps between stations. The usual fix
forecasts at the real stations and spatially interpolates those forecasts
outward — treating the forecasting model itself as blind to anything beyond
the observed graph. This work instead embeds **virtual nodes** — unobserved
locations — directly into the graph and trains the GCRNN to forecast them
end-to-end alongside the real stations, conditioned on the same raster/
contextual side-information (terrain, land cover, ...) as the real ones,
rather than smoothing predictions after the fact:

![Interpolating forecasts between stations vs. this work's virtual nodes embedded in the graph and forecast directly](img/concept_viz.png)

Three domains are covered, each with the same modeling pipeline:

| domain  | data                                   | scripts (suffix) |
|---------|-----------------------------------------|-------------------|
| meteo   | weather station network (Slovenia)      | `_meteo` |
| traffic | motorway traffic counters               | `_traffic` |
| air     | air-quality monitoring stations         | `_air` |

## Repository layout

```
data_handling/       loading & preprocessing: graphs, timeseries, grids, interpolation baselines,
                     graph virtualization (masking/reconnecting virtual nodes),
                     and generate_expw_graphs.py (one-off script to precompute
                     reconnected, exponentially-weighted graphs)
models_impl/         model definitions (GCRNNBase, GCRNNVirtual, ConvLSTM context baseline)
training/            training loop(s)
eval_vis/            inference helpers
interpolation/       classical spatial-interpolation grid-search baselines (IDW / kriging / TPS)

hyperparams*.py   hyperparameter sweep for the base (no virtual nodes) GCRNN
reference_methods*.py      classical interpolation baselines (IDW/kriging/TPS/raster-ConvLSTM) vs. the base model
virtual*.py                the virtual-node GCRNN (main model of this work)
```

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
```

This installs the pinned dependencies from `pyproject.toml` (PyTorch, PyTorch
Geometric, PyTorch Geometric Temporal, NumPy, pandas, NetworkX, natsort,
pyproj, matplotlib) and puts the repository on your Python path in editable
mode, so every script — including the ones inside `interpolation/` and
`data_handling/` — can be run directly regardless of your current working
directory, e.g.:

```bash
python virtual_meteo.py
python interpolation/interpolation_air.py
python data_handling/generate_expw_graphs.py
```

Alternatively, if you don't want an editable install:

```bash
pip install -r requirements.txt
```

but then always run scripts from the repository root (so the top-level
modules and the `data_handling`/`models_impl`/`training`/`eval_vis` packages
resolve on `sys.path`).

**GPU**: training and inference are hardcoded to `torch.device(0)` (CUDA
device 0). A CUDA-capable GPU is required as-is; CPU-only use would need
that hardcoded device swapped for `torch.device('cpu')` in
`training/training.py` and `eval_vis/interence.py`.

## Data

None of the underlying datasets are included in this repository. See
[DATA.md](DATA.md) for the exact `data/` folder layout every script expects,
and what to place in it.

## Running

Every script takes CLI arguments (`--help` lists them) and writes results
under `results/experiments/...`. Two conventions apply, depending on the
script family:

- **`hyperparams*.py`** (hyperparameter sweeps): with no arguments,
  every combination of `--filters`/`--khops`/`--layers` in the default grid
  is run in nested loops. Passing any of these flags pins *that* dimension
  to the given single value while the others keep sweeping their default
  list.

  ```bash
  python hyperparams_meteo.py                        # full grid sweep
  python hyperparams_meteo.py --filters 256           # khops/layers still swept
  python hyperparams_meteo.py --filters 256 --khops 2 --layers 3  # single run
  ```

- **`reference_methods*.py`** and **`virtual*.py`** (the actual experiments,
  run across k-fold splits): with no arguments, every parameter defaults to
  its original value from the source project (fold count `--k`, model size
  `--filters`/`--khops`/`--layers`, `--virtual-ratio`, and — for `virtual*.py`
  — `--variant` {ctx, noctx}), and every fold `0..k(-1)` is run. Passing
  `--fold` restricts the run to that one fold; passing any other flag
  overrides just that value for all folds run.

  ```bash
  python virtual_air.py                          # original config, every fold
  python virtual_air.py --fold 2 --filters 32      # fold 2 only, filters=32
  python reference_methods_meteo.py --fold 0       # just fold 0
  ```

Each script's module docstring documents its exact defaults; function
docstrings on `base_model`/`virtual`/`virtual_noctx`/`base_interp`/
`interp_base`/`interp_raster` document their parameters.

## Citation

If you use this code, please cite:

```bibtex
@article{uremovic2025contextualized,
  title={Contextualized spatio-temporal graph-based method for forecasting sparse geospatial sensor networks},
  author={Uremovi{\'c}, Niko and Mongus, Domen and Pur, Aleksander and Luka{\v{c}}, Niko},
  journal={Expert Systems with Applications},
  volume={294},
  pages={128779},
  year={2025},
  publisher={Elsevier}
}
```
