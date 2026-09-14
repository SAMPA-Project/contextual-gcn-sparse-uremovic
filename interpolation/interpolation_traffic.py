import numpy as np
import os
import json
from dataclasses import dataclass, field
from typing import Literal

# --- Config dataclasses ---

@dataclass(frozen=True)
class IDWConfig:
    method: Literal['idw'] = 'idw'
    power: float = 2
    k: int = 8

    def label(self):
        return f'power={self.power}_k={self.k}'


@dataclass(frozen=True)
class OKConfig:
    method: Literal['ok'] = 'ok'
    variogram: str = 'gaussian'
    range_: float = 120_000
    sill: float = 1.0
    nugget: float = 1e-5
    k: int = 12

    def label(self):
        return f'variogram={self.variogram}_range={self.range_}_sill={self.sill}_nugget={self.nugget}_k={self.k}'


@dataclass(frozen=True)
class TPSConfig:
    method: Literal['tps'] = 'tps'
    reg: float = 1e-5
    k: int = 16

    def label(self):
        return f'reg={self.reg}_k={self.k}'

@dataclass(frozen=True)
class NNConfig:
    method: Literal['nn'] = 'nn'
    k: int = 1
 
    def label(self):
        return f'k={self.k}'

# --- Core interp function ---

def interp(graph_real, metadata_real, graph_full, metadata_full, configs):
    """
    Run spatial interpolation for all provided configs.

    Args:
        graph_real:      graph identifier for the reduced (real) station set G-
        metadata_real:   path to metadata JSON for G-
        graph_full:      graph identifier for the full station set G
        metadata_full:   path to metadata JSON for G
        configs:         list of IDWConfig | OKConfig | TPSConfig instances

    Returns:
        indices_virtual: list of indices of virtual nodes in G
        y_full:          ground-truth labels at virtual nodes, shape (T, V, F, H)
        results:         dict mapping config -> predicted labels at virtual nodes
                         keys are the config objects themselves
    """
    from hyperparams_traffic import base_model
    from data_handling.interp_graph import idw_graph as idw, tps_graph as tps, ok_graph as ok, nn_graph as nn

    filters, layers, khops = 4, 1, 1

    y, _, _, _ = base_model(graph_real, metadata_real, filters=filters, khops=khops, layers=layers,
                      epochs=1, save=False, skiptrain=True)
    y_full, _, edge_weight, edge_index = base_model(graph_full, metadata_full, filters=filters, khops=khops, layers=layers,
                           epochs=1, save=False, skiptrain=True)

    from natsort import natsorted

    with open(metadata_full, 'r') as fp:
        metadataf = json.load(fp)

    with open(metadata_real, 'r') as fp:
        metadatar = json.load(fp)

    # enumerate full set once — this defines the canonical index space
    full_keys_sorted = natsorted(metadataf.keys())
    real_keys_sorted = natsorted(metadatar.keys())
    real_keys_set    = set(real_keys_sorted)

    # full index space: all nodes (observed + virtual)
    metaf_reindex = {
        i: {'x': metadataf[n]['x'], 'y': metadataf[n]['y']}
        for i, n in enumerate(full_keys_sorted)
    }

    # real nodes: same indices as in full space (gaps where virtual nodes are)
    metar_reindex = {
        i: {'x': metadataf[n]['x'], 'y': metadataf[n]['y']}
        for i, n in enumerate(full_keys_sorted)
        if n in real_keys_set
    }

    # virtual node positions in the full index space
    indices_virtual = [
        i for i, n in enumerate(full_keys_sorted)
        if n not in real_keys_set
    ]

    results = {}
    for cfg in configs:
        if isinstance(cfg, IDWConfig):
            pred = idw(y, metar_reindex, metaf_reindex, edge_weight, edge_index, power=cfg.power, max_hops=cfg.k)
        elif isinstance(cfg, OKConfig):
            pred = ok(y, metar_reindex, metaf_reindex,edge_weight, edge_index,
                      variogram=cfg.variogram, range_=cfg.range_,
                      nugget=cfg.nugget, max_hops=cfg.k)
        elif isinstance(cfg, TPSConfig):
            pred = tps(y, metar_reindex, metaf_reindex,edge_weight, edge_index, reg=cfg.reg, max_hops=cfg.k)
        elif isinstance(cfg, NNConfig):
            pred = nn(y, metar_reindex, metaf_reindex,edge_weight, edge_index)
        else:
            raise ValueError(f'Unknown config type: {type(cfg)}')

        results[cfg] = pred[:, indices_virtual]

    return indices_virtual, y_full[:, indices_virtual], results


# --- Aggregation helpers ---

def _make_accumulator(T, total_virtual, F, H):
    return np.zeros((T, total_virtual, F, H), dtype=np.float32)


def _fill_fold(acc, indices, values):
    acc[:, indices] = values


# --- Main ---

if __name__ == '__main__':
    # --- Define config grid ---
    # =============================================================================
    # EXPANDED GRID-SEARCH CONFIG LIST
    # =============================================================================
    #
    # Grounded in the literature above.  Parameter ranges are kept symmetric
    # around the "standard" values so the grid search is interpretable.
    #
    # IDW:   p in {1, 2, 3},  k in {4, 8, 12, 16}          → 12 configs
    # OK:    model in {gaussian, spherical, exponential}
    #        range in {60k, 100k, 150k, 200k} m
    #        sill in {0.8, 1.0, 1.2},  nugget in {1e-5, 1e-4, 1e-3}
    #        k in {8, 12, 16}                                → 3×4×3×3×3 = 324 configs
    #        (reduce to a practical subset below)
    # TPS:   reg in {1e-2, 1e-3, 1e-4, 1e-5, 1e-7},
    #        k in {8, 12, 16, 20}                            → 20 configs
    # =============================================================================
    
    from itertools import product
    
    # --- paste your dataclass definitions here or import them ---
    
    configs = []
    
    # ── IDW ──────────────────────────────────────────────────────────────────────
    # p=2 is the classical default (Shepard 1968; Lu & Wong 2008).
    # p=1 gives smooth regional surfaces; p=3 strongly localises prediction.
    # k=4–16 covers the range used in traffic-network interpolation (Ly et al. 2011).
    #for p, k in product([1, 1.5, 2, 2.5, 3], [1,2,3]):
    for p,k in product([2], [12]):
        configs.append(IDWConfig(power=p, k=k))
    
    # ── Ordinary Kriging ─────────────────────────────────────────────────────────
    # Variogram shapes: all three standard models (Goovaerts 1997, ch. 4).
    # Range: 60–200 km brackets typical country-scale correlation lengths
    #        (Verworn & Haberlandt 2011; Frazier et al. 2022).
    # Sill: normalised to sample variance (≈1.0) ±20 % (Goovaerts 1997).
    # Nugget: three decades covering micro-scale noise to moderate measurement error.
    # k=8–16 follows practice for station-sparse networks (Ly et al. 2011).
    OK_variograms = []#['spherical', 'exponential']
    OK_ranges     = []#[5_000, 20_000, 50_000]
    OK_sills      = []#[1.0] # inferred from the data anyway, keeping for compatibility
    OK_nuggets    = []#[1e-5, 1e-4, 1e-3]
    OK_ks         = []#[1,2,3]
    
    for variogram, range_, sill, nugget, k in product(
            OK_variograms, OK_ranges, OK_sills, OK_nuggets, OK_ks):
        configs.append(OKConfig(
            variogram=variogram,
            range_=range_,
            sill=sill,
            nugget=nugget,
            k=k,
        ))
    
    # ── TPS ───────────────────────────────────────────────────────────────────────
    # λ controls the smoothing–interpolation trade-off (Wahba 1990).
    # λ→0  : exact interpolation (can over-fit noisy data).
    # λ~1e-5: near-exact, slight smoothing — typical for dense networks
    #          (Hutchinson 1995, 1998).
    # λ~1e-2: significant smoothing — appropriate when data is noisy.
    # GCV-optimal λ lies empirically in [1e-5, 1e-2] for traffic-network fields
    # (Hutchinson 1995; Rossiter 2013).
    # k=8–20 controls the local support radius.
    TPS_regs = []#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7]
    TPS_ks   = []#[1,2,3]
    
    for reg, k in product(TPS_regs, TPS_ks):
        configs.append(TPSConfig(reg=reg, k=k))
    
    configs.append(NNConfig(k=1))
    
    # Summary
    idw_configs = [c for c in configs if isinstance(c, IDWConfig)]
    ok_configs  = [c for c in configs if isinstance(c, OKConfig)]
    tps_configs = [c for c in configs if isinstance(c, TPSConfig)]
    nn_configs = [c for c in configs if isinstance(c, NNConfig)]
    
    print(f'Total configs: {len(configs)}')
    print(f'  IDW: {len(idw_configs)}')
    print(f'  OK : {len(ok_configs)}')
    print(f'  TPS: {len(tps_configs)}')
    print(f'  NN: {len(nn_configs)}')

    K = 10

    # Per-fold accumulators; keyed by config, filled after first fold reveals shape
    fold_indices = []
    fold_y_full = []
    fold_results = {cfg: [] for cfg in configs}  # cfg -> list of per-fold arrays

    for i in range(K):
        print(f'K-fold: {i}/{K}')
        graph_real     = f'data/traffic/graph/{i}_real_expw.nx'
        metadata_real  = f'data/traffic/contextual/{i}_real.json'
        graph_full     = 'data/traffic/graph/stations_expw.nx'
        metadata_full  = 'data/traffic/contextual/stations.json'

        iv, yf, results = interp(graph_real, metadata_real, graph_full, metadata_full, configs)

        fold_indices.append(iv)
        fold_y_full.append(yf)
        for cfg in configs:
            fold_results[cfg].append(results[cfg])

    # --- Merge folds into flat virtual-node axis ---
    total_virtual = sum(len(iv) for iv in fold_indices)
    T, _, F, H = fold_y_full[0].shape

    y_merged = _make_accumulator(T, total_virtual, F, H)
    for iv, yf in zip(fold_indices, fold_y_full):
        _fill_fold(y_merged, iv, yf)

    merged_preds = {}
    for cfg in configs:
        acc = _make_accumulator(T, total_virtual, F, H)
        for iv, pred in zip(fold_indices, fold_results[cfg]):
            _fill_fold(acc, iv, pred)
        merged_preds[cfg] = acc

    # --- Compute errors & save ---
    for cfg in configs:
        pred = merged_preds[cfg]
        error = np.mean(np.abs(y_merged - pred), axis=(0, 1, 2))  # shape (H,)

        save_dir = os.path.join(
            'results/experiments/interp_traffic_expw',
            cfg.method,
            cfg.label(),
        )
        os.makedirs(save_dir, exist_ok=True)

        # Save ground truth once per method folder (not duplicated per config)
        method_dir = os.path.dirname(save_dir)
        y_path = os.path.join(method_dir, 'y.npy')
        if not os.path.exists(y_path):
            np.save(y_path, y_merged)

        np.save(os.path.join(save_dir, f'y_{cfg.method}.npy'), pred)
        np.save(os.path.join(save_dir, f'{str(error)}.npy'), error)

        print(f'[{cfg.method}] {cfg.label()} — mean error: {error.mean():.6f}')