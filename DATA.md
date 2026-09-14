# Data layout

No data ships with this repository. Every path below is read relative to
the repository root, so create a `data/` folder there matching this layout
exactly (paths in **bold** are the ones the code opens directly; everything
else is folder structure).

None of this is anonymized/synthetic — it is real station/counter data, so
treat it accordingly (do not commit it; keep it out of version control,
e.g. via `.gitignore`).

The layout is unified across all three datasets:

```
data/<dataset>/graph/...        edgelists (.nx), flat files
data/<dataset>/contextual/...   metadata JSONs, raster npys, context CSVs
data/<dataset>/timeseries/...   per-node timeseries CSVs
```

and file naming follows the same convention everywhere: the full node set is
always `stations.json` / `stations*.nx`; a k-fold subset is always
`<fold>_real.json` / `<fold>_virtual.json` (metadata) or `<fold>_real*.nx`
(graph, real nodes only — there's no separate "virtual" graph, since virtual
nodes live inside the full graph). Graph files take the same base name as
their corresponding metadata, plus a weighting-scheme suffix wherever more
than one graph variant exists for the same node set (`_inv` = inverse
road-distance weights, `_expw` = exponential road-distance weights, `_hp` =
the separate graph used only for meteo's hyperparameter sweep — see the note
below).

If you still have the original `tools/meteo_slo/`, `tools/dars_traffic/`,
`tools/arso_air/` tree from the source project, run
[`scripts/collect_zenodo_data.py`](scripts/collect_zenodo_data.py) from
there — it copies exactly the files below into this layout and zips the
result (for a Zenodo upload, say), reporting anything it can't find rather
than failing silently:

```bash
python collect_zenodo_data.py --source-root tools --dest-root data
```

If you already have a `data/` folder built by an older version of that
script (the `stations_curated_sorted*.json` / `graph/<name>/edgelist.nx` /
`metadata/k_fold/...` layout), migrate it in place instead of re-fetching —
[`scripts/repack_data.py`](scripts/repack_data.py) moves everything into the
layout below and deletes any `data/paper/`, `data/mag/` (unrelated to this
project):

```bash
python repack_data.py --root data          # add --dry-run to preview first
```

## `data/meteo/` — weather stations (9-fold split, `k=8` → folds `i=0..8`)

- **`data/meteo/contextual/stations.json`** — full station metadata (keys
  are `id_XXXX` matching the timeseries filenames; each entry needs at least
  `x`, `y` projected coordinates and an `idx`).
- **`data/meteo/contextual/{0..8}_real.json`** and
  **`{0..8}_virtual.json`** — per-fold metadata for the "real" (observed)
  and held-out ("virtual") station subsets, 9+9 files.
- **`data/meteo/timeseries/*.csv`** — one CSV per station, filename ending in
  the 4-digit station id (matched via `id_<last 4 chars of filename>`).
  Expected columns include `povp. T`, `povp. rel. vla.`, `količina padavin`,
  plus precomputed `daytime_sin/cos`, `yearday_sin/cos`.
- **`data/meteo/graph/stations.nx`** — the full graph (all stations); this
  is the "higher thresholds" graph the k-fold graphs below were actually
  virtualized from, and what `reference_methods_meteo.py`/`virtual_meteo.py`
  use as `graph_full`.
- **`data/meteo/graph/stations_hp.nx`** — a *different* full graph (looser
  thresholds), used only by `hyperparams_meteo.py`'s hyperparameter sweep —
  not part of the main fold structure.
- **`data/meteo/graph/{0..8}_real.nx`** — per-fold real-only graphs.
- **`data/meteo/contextual/dtm.npy`** — digital terrain model raster (elevation).
- **`data/meteo/contextual/sentinel2_data.npy`** — Sentinel-2 raster stack
  (channel 11 = SCL, channel 12 = NDVI source, per `data_handling/grid.py`).

## `data/traffic/` — motorway traffic counters (10-fold split, `k=10` → folds `i=0..9`)

- **`data/traffic/contextual/stations.json`** — full counter metadata.
- **`data/traffic/contextual/{0..9}_real.json`** and
  **`{0..9}_virtual.json`** — per-fold "real"/"virtual" counter metadata,
  10+10 files.
- **`data/traffic/timeseries/*.csv`** — one CSV per counter, filename ending
  in the 6-character counter id. Expected columns include `slow`, `fast`,
  `VAvg`, `VMin`, `VMax`, `Gap`, `Occ`, `Status`, calendar features
  (`yearday_sin/cos`, `weektime_sin/cos`, `daytime_sin/cos`), holiday flags
  (`praznik_<Country>`), weather (`povp. T`, `kolicina padavin`,
  `povp. rel. vla.`), and `event_*` one-hot event columns.
- **`data/traffic/graph/stations.nx`** and **`stations_expw.nx`** — full
  graph, base weights and exponential road-distance weights (the latter
  *generated*, not raw — see note below).
- **`data/traffic/graph/{0..9}_real.nx`** and **`{0..9}_real_expw.nx`** —
  per-fold graphs, same two weight variants.

Node/edge metadata also expects per-counter fields: `onehot_smer1/2`,
`speed_limit`, `onehot_AC`, `onehot_HC` (used as `METADATA_FEATURES`
context).

## `data/air/` — air-quality stations (8-fold split, `k=8` → folds `i=0..7`)

- **`data/air/contextual/stations.json`** — full station metadata.
- **`data/air/contextual/{0..7}_real.json`** and
  **`{0..7}_virtual.json`** — per-fold "real"/"virtual" station metadata,
  8+8 files.
- **`data/air/timeseries/*.csv`** — one CSV per station, filename ending in
  the 4-character station id. Expected columns include `NO2`, `O3`, `PM10`,
  weather (`povp. T`, `kolicina padavin`, `povp. rel. vla.`, `hitrost_vetra`,
  `povp_tlak`), and calendar features.
- **`data/air/graph/stations_inv.nx`** — the full graph (inverse
  road-distance weights — the only weighting scheme used for air).
- **`data/air/graph/{0..7}_real_inv.nx`** — per-fold graphs.
- **`data/air/contextual/contextual_features_normalized.csv`** — per-station
  contextual feature table indexed by station id, providing the
  `METADATA_FEATURES` columns (settlement distance/elevation/population,
  large combustion plant — `lcp_*` — proximity/emissions, NDVI, building
  coverage, plus rolling per-pollutant stats `<pollutant>_mean/std/p5/p95/daily_fluct`).

## Notes / things to double check when sourcing this from the larger project

- `hyperparams_air.py`'s `__main__` block and the rest of the "air"
  scripts used two *different* paths for what looks like the same full
  station-metadata file (`arso_air/data/metadata/arso_air_postaje_curated.json`
  vs. `tools/arso_air/arso_air_postaje_curated.json`). Both were normalized
  here to `data/air/contextual/stations.json` — please confirm that's
  actually the same file before copying data over, otherwise
  `hyperparams_air.py` needs pointing at a second file.
- `interpolation_meteo.py` imported a `phd_hyperparams` module that does not
  exist anywhere in this codebase (only `phd_hyperparams_residual.py` did,
  itself since renamed to `hyperparams_meteo.py`). It's been repointed to
  `hyperparams_meteo.py`, assuming that was always the intent — flag if that
  assumption is wrong.
- The meteo `stations.nx` vs. `stations_hp.nx` split (see above) is a
  judgment call: the original two graphs were named
  `..._fullHigherBenchmarks` and a bare (no-suffix) name, with no explicit
  documentation of which was "canonical". `stations.nx` was chosen as the
  one actually used by the fold-based experiments; `stations_hp.nx` is only
  ever used standalone by the hyperparameter sweep. Flag if that's backwards.
- The `data/traffic/graph/*_expw.nx` files are *generated* (not raw data) by
  `data_handling/generate_expw_graphs.py` from `stations.nx` +
  `contextual/stations.json`. You only need to source the base `.nx` graphs
  and can regenerate the `_expw` variants by running that script once the
  base data is in place.
