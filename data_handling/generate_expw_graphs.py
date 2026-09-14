"""
One-off data-prep script: generates reconnected, exponentially-weighted
edge lists for the traffic dataset (full graph + one per k-fold split).

Weight convention: w(i,j) = exp(-d(i,j) / sigma)
  where d(i,j) is the Euclidean distance between nodes i and j,
  and sigma = median Euclidean edge distance over the full graph.

For each fold, virtual (held-out) nodes are removed from the real-only
graph and bridged over via reconnect_virtual_nodes() (data_handling/
graph_virtualization.py), which multiplies weights along the bridge:
w(u,w) = w(u,v) * w(v,w).

New files are saved with an _expw suffix so they don't overwrite the
existing (non-exponentially-weighted) graphs:
  full graph:  data/traffic/graph/stations_expw.nx
  fold graphs: data/traffic/graph/{fold}_real_expw.nx

Run with no arguments; edit the CONFIG constants below to point at a
different metadata/graph layout or fold count.

    python data_handling/generate_expw_graphs.py
"""

import json
import os

import numpy as np
from natsort import natsorted

from data_handling.graph import load_graph_traffic
from data_handling.graph_virtualization import reconnect_virtual_nodes

# =============================================================================
# CONFIG
# =============================================================================

GRAPH_REAL    = 'data/traffic/graph/stations.nx'
METADATA_REAL = 'data/traffic/contextual/stations.json'

KFOLD_META_DIR = 'data/traffic/contextual'
GRAPH_DIR      = 'data/traffic/graph'
N_FOLDS        = 10


# =============================================================================
# HELPERS
# =============================================================================

def save_graph(ei_np, ew_np, path, idx_to_key):
    """Write a weighted edgelist (networkx-readable) to `path`, mapping node
    indices back to their original metadata keys via `idx_to_key`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for k in range(ei_np.shape[1]):
            src_key = idx_to_key[int(ei_np[0, k])]
            dst_key = idx_to_key[int(ei_np[1, k])]
            f.write(f"{src_key} {dst_key} {ew_np[k]:.10f}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── load topology (edge_index only, ignore existing weights) ────────────
    _, ei_full = load_graph_traffic(GRAPH_REAL, METADATA_REAL)
    ei_np   = ei_full.numpy() if hasattr(ei_full, 'numpy') else np.array(ei_full)
    n_nodes = int(ei_np.max()) + 1
    print(f"full graph: {n_nodes} nodes, {ei_np.shape[1]} edges")

    # ── load metadata: natsorted index <-> key, and x/y coords ──────────────
    with open(METADATA_REAL, 'r') as f:
        meta_full = json.load(f)
    full_keys_sorted = natsorted(meta_full.keys())
    key_to_idx = {k: i for i, k in enumerate(full_keys_sorted)}
    idx_to_key = {i: k for k, i in key_to_idx.items()}

    xs = np.array([meta_full[k]['x'] for k in full_keys_sorted])
    ys = np.array([meta_full[k]['y'] for k in full_keys_sorted])

    # ── compute Euclidean distances for all edges ────────────────────────────
    src_x, src_y = xs[ei_np[0]], ys[ei_np[0]]
    dst_x, dst_y = xs[ei_np[1]], ys[ei_np[1]]
    euclid_dists = np.sqrt((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2)

    sigma = np.median(euclid_dists)
    print(f"  Euclidean distance range: {euclid_dists.min():.1f} — {euclid_dists.max():.1f} m")
    print(f"  sigma (median):           {sigma:.1f} m")

    # ── convert distances to exponential weights ─────────────────────────────
    ew_expw = np.exp(-euclid_dists / sigma)
    print(f"  weight range: {ew_expw.min():.4f} — {ew_expw.max():.4f}")

    # ── save full weighted graph ──────────────────────────────────────────────
    full_out = os.path.join(GRAPH_DIR, 'stations_expw.nx')
    save_graph(ei_np, ew_expw, full_out, idx_to_key)
    print(f"\nfull graph saved -> {full_out}")

    # ── process each fold ─────────────────────────────────────────────────────
    for fold in range(N_FOLDS):
        virtual_path = os.path.join(KFOLD_META_DIR, f'{fold}_virtual.json')
        with open(virtual_path, 'r') as f:
            meta_virtual = json.load(f)

        ids_virtual     = sorted([key_to_idx[k] for k in meta_virtual.keys()])
        ids_virtual_set = set(ids_virtual)

        print(f"\nfold {fold}: {len(ids_virtual)} virtual nodes")

        # reconnect_virtual_nodes multiplies bridging weights
        ei_rec, ew_rec, _ = reconnect_virtual_nodes(
            ei_np.copy(), ew_expw.copy(), ids_virtual_set, n_nodes
        )
        print(f"  edges: {ei_rec.shape[1]}  (+{ei_rec.shape[1]-ei_np.shape[1]} bridging)")
        print(f"  weight range: {ew_rec.min():.4f} — {ew_rec.max():.4f}")

        # sanity check: no edge should still touch a virtual node
        leaked = [(u, v) for u, v in zip(ei_rec[0], ei_rec[1])
                  if int(u) in ids_virtual_set or int(v) in ids_virtual_set]
        assert len(leaked) == 0, f"fold {fold}: {len(leaked)} leaked edges!"

        out_path = os.path.join(GRAPH_DIR, f'{fold}_real_expw.nx')
        save_graph(ei_rec, ew_rec, out_path, idx_to_key)
        print(f"  saved -> {out_path}")

    print("\ndone — all folds saved")


if __name__ == '__main__':
    main()
